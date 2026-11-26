import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 让 matplotlib 能正常显示中文标题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

hide_size = 768
head_count = 12
per_head_hide_size = hide_size // head_count
batch_size = 5
max_len = 512
blok_count = 5

# ---------------- 训练超参数 ----------------
seq_len = 64            # 每条样本的字符数
epochs = 5              # 训练轮数
lr = 3e-4               # 学习率
max_vocab = 4000        # 词表大小
max_chars = 40_000      # 取多少字符训练
seed = 42

UNK_ID = 0            # <unk>
EOS_ID = 1            # <eos>，答案结束标记
IGNORE_INDEX = -100   # loss 中被忽略的位置（SFT 里 instruction 不参与训练）
EOS_CHAR = "\ue000"   # 用私有区字符当作 <eos>，单字符、语料里不会出现

DATA_PATH = Path(__file__).resolve().parent / "data" / "corpus_general.txt"
SFT_TRAIN_PATH = Path(__file__).resolve().parent / "data" / "sft_train.jsonl"   # SFT 训练语料
SFT_VAL_PATH = Path(__file__).resolve().parent / "data" / "sft_val.jsonl"       # SFT 验证语料
SAVE_PATH = Path(__file__).resolve().parent / "val_loss.png"     # 验证集 loss 曲线
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pt"   # 验证集最优时保存的权重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 有 GPU 就用 GPU

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(in_features=hide_size, out_features=hide_size)
        self.k = nn.Linear(in_features=hide_size, out_features=hide_size)
        self.v = nn.Linear(in_features=hide_size, out_features=hide_size)
        self.o = nn.Linear(in_features=hide_size, out_features=hide_size)
        self.layer_norm_1 = nn.LayerNorm(hide_size)
        self.layer_norm_2 = nn.LayerNorm(hide_size)

        self.gelu = nn.GELU()
        self.ffn_linear_1 = nn.Linear(in_features=hide_size, out_features=hide_size * 4)
        self.ffn_linear_2 = nn.Linear(in_features=hide_size * 4, out_features=hide_size)

    def multi_attention(self, x):
        """ 多头注意力 """
        # x= batch,seqLength,hide_size
        b, t, _ = x.shape
        # 张量拆分
        q = self.q(x).reshape(b, t, head_count, per_head_hide_size).transpose(1, 2)
        k_t = self.k(x).reshape(b, t, head_count, per_head_hide_size).transpose(1, 2).transpose(-2, -1)
        v = self.v(x).reshape(b, t, head_count, per_head_hide_size).transpose(1, 2)
        # 增加掩码，保留对角线（上三角置 -inf，第 i 位只能看到 0..i）
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        attention = (q @ k_t / math.sqrt(per_head_hide_size)).masked_fill(mask, float('-inf'))
        rel = torch.softmax(attention, dim=-1) @ v
        return self.o(rel.transpose(1, 2).reshape(b, t, -1))

    def ffn(self, x):
        """ 前馈神经网络 """
        return self.ffn_linear_2(self.gelu(self.ffn_linear_1(x)))

    def forward(self,x):
        x = self.multi_attention(self.layer_norm_1(x)) + x
        return self.ffn(self.layer_norm_2(x)) + x

class DecoderLM(nn.Module):
    def __init__(self, vcb_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vcb_size, embedding_dim=hide_size)
        self.position = nn.Embedding(num_embeddings=max_len, embedding_dim=hide_size)
        self.lm_head = nn.Linear(in_features=hide_size, out_features=vcb_size)
        self.blocks = nn.ModuleList(DecoderBlock() for _ in range(blok_count))
        self.layer_norm3 = nn.LayerNorm(hide_size)

    def forward(self, x):
        """ 前向传播 """
        # b,t->b,t,h  加上可训练的位置编码
        b, t = x.shape
        x = self.embedding(x) + self.position(torch.arange(t, device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.layer_norm3(x))

    @torch.no_grad()
    def generate(self, text, max_new=40):
        """给定开头文本，续写 max_new 个字。需要先设置 self.char_index / self.index_char。"""
        ids = [self.char_index.get(c, 0) for c in text][-seq_len:]
        out = []
        for _ in range(max_new):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = self(x)[0, -1, :].clone()
            logits[0] = float("-inf")            # 不生成 <unk>
            nxt = int(logits.argmax().item())
            if nxt == EOS_ID:
                break                       # 生成到 <eos> 就停
            ids.append(nxt)
            out.append(nxt)
            ids = ids[-seq_len:]
        return "".join(self.index_char[i] for i in out)


class CharDataset(Dataset):
    """字符级数据：x 是前 seq_len 个字符，y 是往后错一位的下一个字符"""
    def __init__(self, text, char_index):
        self.seq_len = seq_len
        ids = [char_index.get(c, 0) for c in text]   # 未登录字 -> 0 号 <unk>
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.seq_len + 1)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


def load_sft(path):
    """读取 SFT 的 jsonl 文件，把每条 instruction+output 拼成一段文本 sft需要和预训练的token id保持一致"""
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        segments.append(obj["instruction"] + "\n" + obj["output"] + EOS_CHAR)
    return "\n".join(segments)


def data_handel():
    """读预训练语料和 SFT 语料，建词表，划分训练集和验证集"""
    pretrain_text = DATA_PATH.read_text(encoding="utf-8")[:max_chars]
    sft_train_text = load_sft(SFT_TRAIN_PATH)
    sft_val_text = load_sft(SFT_VAL_PATH)

    # 词表从全部语料（预训练 + SFT）里统计，避免 SFT 里的字变成 <unk>
    counter = Counter(pretrain_text + sft_train_text + sft_val_text)
    # 过滤掉 EOS_CHAR，避免它既出现在 most_common 里、又作为特殊 token 重复进词表
    chars = ["<unk>", EOS_CHAR] + [c for c, _ in counter.most_common(max_vocab) if c != EOS_CHAR]
    char_index = {c: i for i, c in enumerate(chars)}
    index_char = {i: c for c, i in char_index.items()}

    # 预训练语料按行打乱，9:1 划分
    pretrain_lines = pretrain_text.splitlines()
    random.shuffle(pretrain_lines)
    split = int(len(pretrain_lines) * 0.9)
    pretrain_train = "\n".join(pretrain_lines[:split])
    pretrain_val = "\n".join(pretrain_lines[split:])

    # SFT 的 train / val 本来就已经分好，直接各自加入
    train_text = pretrain_train + "\n" + sft_train_text
    val_text = pretrain_val + "\n" + sft_val_text

    train_loader = DataLoader(CharDataset(train_text, char_index),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(CharDataset(val_text, char_index),
                            batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, char_index, index_char


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """验证集上的平均 loss（忽略 <unk> 位置）"""
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.transpose(1, 2), y)
        valid = int((y != 0).sum().item())
        total += loss.item() * valid
        n += valid
    return total / n if n else 0.0


def train():
    random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader, char_index, index_char = data_handel()
    model = DecoderLM(len(char_index)).to(device)
    model.char_index = char_index     # 让 model.generate() 能直接编解码
    model.index_char = index_char
    print(f"device: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)   # <unk>(id=0) 不参与 loss

    # 实时画验证集 loss 曲线：横轴 epoch，纵轴 val loss
    is_live = plt.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo")
    if is_live:
        plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_axis, val_losses = [], []

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                       # b,t,vcb
            # 自回归：logits 每一列预测的是下一个字，和错一位的 y 对齐算 loss
            loss = loss_fn(logits.transpose(1, 2), y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_loss = evaluate(model, val_loader, loss_fn)

        # 只要验证集 loss 比历史最低还低，就保存权重
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            saved = "  * 已保存权重"
        else:
            saved = ""

        epochs_axis.append(epoch + 1)
        val_losses.append(val_loss)

        ax.clear()
        ax.plot(epochs_axis, val_losses, "o-")
        ax.set_title("验证集 loss 变化")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val loss")
        if is_live:
            plt.draw()
            plt.pause(0.001)

        print(f"epoch {epoch + 1}/{epochs}  val_loss {val_loss:.4f}{saved}")

    fig.savefig(SAVE_PATH)
    print(f"验证集 loss 曲线已保存: {SAVE_PATH}")


if __name__ == "__main__":
    train()

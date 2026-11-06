"""

使用pytorch实现transformer架构的的编码器
以bert模型的维度来实现（字符级语言模型：根据前文预测下一个字符）

带因果掩码：第 i 个位置只能看到第 0..i 个字符，预测下一个字时看不到它自己，
所以最后一个位置的预测才是真正的"下一个字"，可以直接拿来做续写。
（掩码代码已注释掉，见 multi_attention；causal = True 时重新启用。）

训练结束后会把 loss / 准确率 曲线保存成 test4/train_curves.png，
把每个 epoch 的数值保存成 test4/train_history.csv。

checkpoint 里同时保存了模型权重和词表（char_index / index_char），
所以可以用 load_checkpoint() 直接还原出可用的模型，不会出现 id 对不上字的情况。

<unk>（id=0）只作为输入里的占位符，不参与 loss 和准确率的统计。

"""
import csv
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------- 超参数 ----------------
UNK_ID = 0              # <unk> 的词表下标，同时作为 ignore_index：不参与 loss / 准确率
batch_size = 4
max_len = 512           # 位置编码支持的最大长度
split_seq_len = 64      # 每条样本的字符数（= 训练窗口）
stride = 64             # 滑窗步长，等于 split_seq_len 表示样本之间不重叠
d = 768                 # 隐藏维度
head_d = 64             # 每个注意力头的维度
head_count = d // head_d
epochs = 6
max_vocab = 4000        # 词表最多保留多少个高频字符
max_chars = 40_000      # 从语料里取多少字符做实验（CPU 上跑得动；想认真训练就调大）
lr = 1e-3
causal = False          # 因果掩码已注释掉；改回 True 会启用 attention 里那段 mask
seed = 42
log_every = 25          # 每多少个 step 打印一次训练进度

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "corpus_general.txt"
CURVE_PATH = HERE / "train_curves.png"
HISTORY_PATH = HERE / "train_history.csv"
MODEL_PATH = HERE / "trans_lm.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharDataset(Dataset):
    def __init__(self, text, char_index):
        self.seq_len = split_seq_len
        self.stride = stride
        # 不在词表里的字统一映射到 UNK_ID，避免被直接丢掉导致文本不连续
        ids = [char_index.get(c, UNK_ID) for c in text]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 最后一条样本也必须是完整的 seq_len 个输入 + 1 个标签
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


def data_handel(max_chars=max_chars):
    text_str = DATA_PATH.read_text(encoding="utf-8")[:max_chars]

    counter = Counter(text_str)
    chars = ["<unk>"] + [c for c, _ in counter.most_common(max_vocab)]
    char_index = {c: i for i, c in enumerate(chars)}
    index_char = {i: c for c, i in char_index.items()}
    assert char_index["<unk>"] == UNK_ID

    # 打乱行顺序，按 9:1 划分训练集和验证集（按行切分可以避免两边出现重叠片段）
    texts = text_str.splitlines()
    random.shuffle(texts)
    split = int(len(texts) * 0.9)
    trans = "\n".join(texts[:split])
    tests = "\n".join(texts[split:])

    trains_dataset = CharDataset(trans, char_index)
    tests_dataset = CharDataset(tests, char_index)
    train_loader = DataLoader(trains_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(tests_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, char_index, index_char


# 定义模型
class Trans(nn.Module):
    def __init__(self, vocab_size, causal=causal):
        super().__init__()
        self.causal = causal

        # 词嵌入 + 位置编码：输入的 token id 必须先变成 d 维向量才能进 attention
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.dropout = nn.Dropout(0.1)

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.active = nn.GELU()
        self.q_linear = nn.Linear(d, d)
        self.k_linear = nn.Linear(d, d)
        self.v_linear = nn.Linear(d, d)
        self.o_linear = nn.Linear(d, d)
        self.f1_linear = nn.Linear(d, 4 * d)
        self.f2_linear = nn.Linear(4 * d, d)

        # 输出头：把 d 维隐藏状态映射回词表大小，得到每个字符的分数
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # 这里按照 batch, head_count, seq_len, head_d 来定义张量，因为 @ 只计算最后两个维度做矩阵乘
    def multi_attention(self, x):
        b, t, _ = x.shape                       # 用真实形状，不用全局 batch_size / seq_len
        q = self.q_linear(x).reshape(b, t, head_count, head_d).transpose(1, 2)
        k = self.k_linear(x).reshape(b, t, head_count, head_d).transpose(1, 2)
        v = self.v_linear(x).reshape(b, t, head_count, head_d).transpose(1, 2)

        score = q @ k.transpose(-1, -2) / math.sqrt(head_d)      # (b, head, t, t)
        rel = torch.softmax(score, dim=-1) @ v
        rel = rel.transpose(1, 2).contiguous().reshape(b, t, d)
        return self.o_linear(rel)

    def fnn(self, x):
        return self.f2_linear(self.active(self.f1_linear(x)))

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        x = self.ln1(x + self.dropout(self.multi_attention(x)))
        x = self.ln2(x + self.dropout(self.fnn(x)))
        return self.lm_head(x)                  # (b, t, vocab_size)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """在验证集上跑一遍，返回平均 loss 和 next-char 准确率（<unk> 不计入）。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # 只统计真实字符：y == UNK_ID 的位置既不算 loss 也不算对错
        mask = y != UNK_ID
        n_valid = int(mask.sum().item())
        if n_valid == 0:
            continue
        logits = model(x)
        loss = loss_fn(logits.transpose(1, 2), y)
        total_loss += loss.item() * n_valid
        correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
        total += n_valid
    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, correct / total


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    xs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(xs, history["train_loss"], "o-", label="train loss")
    axes[0].plot(xs, history["val_loss"], "s-", label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(xs, history["train_acc"], "o-", label="train acc")
    axes[1].plot(xs, history["val_acc"], "s-", label="val acc")
    axes[1].set_title("Next-char accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"曲线已保存: {path}")


def save_checkpoint(path, model, char_index, index_char):
    """把权重和词表一起存，缺了词表这个 checkpoint 就没法用。"""
    torch.save({
        "model_state": model.state_dict(),
        "char_index": char_index,
        "index_char": index_char,
        "config": {
            "vocab_size": len(char_index),
            "unk_id": UNK_ID,
            "d": d,
            "head_count": head_count,
            "head_d": head_d,
            "max_len": max_len,
            "split_seq_len": split_seq_len,
            "causal": bool(causal),
        },
    }, path)


def load_checkpoint(path, map_location="cpu"):
    """还原模型 + 词表，返回 (model, char_index, index_char)。"""
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    model = Trans(cfg["vocab_size"], causal=cfg.get("causal", causal))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["char_index"], ckpt["index_char"]


def main():
    random.seed(seed)
    torch.manual_seed(seed)

    train_loader, test_loader, char_index, index_char = data_handel()
    vocab_size = len(char_index)
    print(f"device={device}  vocab={vocab_size}  "
          f"train steps/epoch={len(train_loader)}  val steps={len(test_loader)}")

    model = Trans(vocab_size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # ignore_index=UNK_ID：<unk> 的位置不产生梯度，也不进 loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=UNK_ID)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ---------- 训练 ----------
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            # 准确率和 loss 都只统计真实字符，<unk> 位置跳过
            mask = y != UNK_ID
            n_valid = int(mask.sum().item())
            if n_valid == 0:
                continue
            loss = loss_fn(logits.transpose(1, 2), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 防止梯度爆炸
            opt.step()

            run_loss += loss.item() * n_valid
            correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
            total += n_valid
            if step % log_every == 0 or step == len(train_loader):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {run_loss / total:.4f} acc {correct / total:.4f}")

        train_loss, train_acc = run_loss / total, correct / total
        # ---------- 验证 ----------
        val_loss, val_acc = evaluate(model, test_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch:3d}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
              f"val loss {val_loss:.4f} acc {val_acc:.2%} | ppl {math.exp(val_loss):.1f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(MODEL_PATH, model, char_index, index_char)
            print(f"  ↑ 验证集 loss 创新低，已保存模型+词表到 {MODEL_PATH.name}")

    # ---------- 保存结果 ----------
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(epochs):
            writer.writerow([i + 1,
                             f"{history['train_loss'][i]:.6f}", f"{history['train_acc'][i]:.6f}",
                             f"{history['val_loss'][i]:.6f}", f"{history['val_acc'][i]:.6f}"])
    print(f"训练历史已保存: {HISTORY_PATH}")

    plot_curves(history, CURVE_PATH)

    # ---------- 校验 checkpoint 能把模型和词表一起还原 ----------
    _, restored_char_index, restored_index_char = load_checkpoint(MODEL_PATH)
    assert restored_char_index == char_index and restored_index_char == index_char
    print(f"checkpoint 校验通过: 词表 {len(restored_index_char)} 个字符，"
          f"<unk>={restored_index_char[UNK_ID]!r}，可直接用 load_checkpoint() 加载")


if __name__ == '__main__':
    main()

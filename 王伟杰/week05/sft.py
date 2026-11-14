import torch
import torch.nn as nn
import sys
import json
sys.path.insert(0, "../week04/循环神经网络语言模型")
from language_model import TransformerLM
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F



BASE_CKPT = "../week04/循环神经网络语言模型/tf_best_10ep.pt"   # 上周训练好的模型
SPECIAL_TOKENS = ["<SEP>", "<EOS>"]
MAX_LEN = 128      # SFT 数据最长 113，留一点余量
EPOCHS = 10


def resize_embedding(old, new_num):
    """把字向量表 old 扩成 new_num 行：旧的行原样保留，新的行用小随机数"""
    old_num, dim = old.weight.shape
    # 1. 建一张更大的新表
    new = nn.Embedding(new_num, dim)
    # 2. 新表先全部填上小随机数（和上周 tricks 里的小初始化一样）
    nn.init.normal_(new.weight, std=0.02)
    # 3. 把旧表的内容抄到新表的前 old_num 行
    with torch.no_grad():
        new.weight[:old_num] = old.weight
    return new

def load_base_model(path):
    """加载上周的预训练模型，加入特殊符号，并扩大字向量表"""
    ck = torch.load(path, map_location="cpu")
    cfg, char2idx, idx2char = ck["args"], ck["char2idx"], ck["idx2char"]
    model = TransformerLM(len(char2idx), cfg["embed_dim"], cfg["num_heads"], cfg["ff_dim"],
                          cfg["num_layers"], cfg["seq_len"], cfg["dropout"], tricks=cfg["tricks"])
    model.load_state_dict(ck["model_state"])

    # 1. 字表里加上特殊符号，编号接在最后面
    for tok in SPECIAL_TOKENS:
        idx = len(char2idx)
        char2idx[tok] = idx
        idx2char[idx] = tok
    # 2. 把字向量表扩大到新字表的大小
    model.embed = resize_embedding(model.embed , len(char2idx))
    # 3. 重新绑定权重共享：让 fc 也指向新表
    model.fc.weight = model.embed.weight
    # 4. 位置表从 64 行扩到 MAX_LEN 行，因为 SFT 数据最长有 113 个字
    model.pos_embed = resize_embedding(model.pos_embed, MAX_LEN)
    model.max_len = MAX_LEN

    return model, cfg, char2idx, idx2char

def encode(sample, char2idx):
    """把一条 {"instruction": ..., "output": ...} 变成一串编号：指令 <SEP> 回答 <EOS>"""
    prompt = [char2idx[c] for c in sample["instruction"] if c in char2idx]
    answer = [char2idx[c] for c in sample["output"] if c in char2idx]
    ids = prompt + [char2idx["<SEP>"]] + answer + [char2idx["<EOS>"]]
    return ids, len(prompt)      # 第 2 个返回值是"指令有多长"，下一步算 loss 时要用

def make_xy(ids, n_prompt):
    """把一串编号变成训练用的 x 和 y，并把指令部分的 y 设为 -100（不算 loss）"""
    ids = torch.tensor(ids, dtype=torch.long)
    x = ids[:-1]           # 去掉最后一个
    y = ids[1:].clone()    # 去掉第一个（clone 是复制一份，改 y 时不会影响 ids）
    y[:n_prompt] = -100         # 指令部分不算 loss
    return x, y

class SFTDataset(Dataset):
    """读一个 jsonl 文件，每条数据都处理成 (x, y)"""
    def __init__(self, path, char2idx):
        self.samples = []
        for line in open(path, encoding="utf-8"):
            ids, n_prompt = encode(json.loads(line), char2idx)
            ids = ids[:MAX_LEN + 1]        # 保险：x、y 最多各 MAX_LEN 个
            self.samples.append(make_xy(ids, n_prompt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

def collate(batch):
    """batch = [(x1, y1), (x2, y2), ...]，长度各不相同 → 在后面补齐成一样长"""
    L = max(len(x) for x, y in batch)                          # 这一批里最长的有多长
    xs = torch.full((len(batch), L), 0, dtype=torch.long)      # x 补 0
    ys = torch.full((len(batch), L), -100, dtype=torch.long)    # y 补什么？
    for i, (x, y) in enumerate(batch):                         # 第 i 条数据
        xs[i, :len(x)] = x                                     # 把 x 填进第 i 行的前面
        ys[i, :len(y)] = y                                        # 把 y 填进第 i 行的前面
    return xs, ys

@torch.no_grad()
def evaluate(model, loader, device):
    """在回答部分算平均 loss 和 PPL（y = -100 的位置不算）"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                                         # (B, L, 4002)
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   reduction="sum")               # 这一批的 loss 总和（-100 自动跳过）
        total_loss += loss_sum.item()
        total_tokens += (y != -100).sum().item()    # 这一批有几个位置算了 loss？
    avg = total_loss / total_tokens
    return avg, math.exp(avg)

def train_one_epoch(model, loader, optimizer, device):
    """训练一轮，返回回答部分的平均 loss"""
    model.train()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))   # 平均 loss，-100 自动跳过
        optimizer.zero_grad()
        loss.backward()         # 反向传播，算出梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # 梯度裁剪（上周学的）
        optimizer.step()        # 按梯度更新参数
        n = (y != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens

# if __name__ == "__main__":
#     old = nn.Embedding(4000, 192)
#     new = resize_embedding(old, 4002)
#     print("新表形状:", tuple(new.weight.shape), "  期望 (4002, 192)")
#     print("前 4000 行和旧表一样吗:", torch.equal(new.weight[:4000], old.weight), "  期望 True")
if __name__ == "__main__":
    model, cfg, char2idx, idx2char = load_base_model(BASE_CKPT)
    # print("特殊符号编号:", char2idx["<SEP>"], char2idx["<EOS>"], "  期望 4000 4001")
    # print("embed 形状:", tuple(model.embed.weight.shape), "  期望 (4002, 192)")
    # print("fc    形状:", tuple(model.fc.weight.shape), "  期望 (4002, 192)")
    # print("还是同一张表吗:", model.fc.weight is model.embed.weight, "  期望 True")
    # sample = json.loads(open("sft_train.jsonl", encoding="utf-8").readline())   # 读第一条数据
    # ids, n_prompt = encode(sample, char2idx)
    # print("编号:", ids)
    # print("还原:", "".join(idx2char[i] for i in ids))
    # print("总长度:", len(ids), "  指令长度:", n_prompt)
    # x, y = make_xy(ids, n_prompt)
    # print("\n位置  x(输入)  →  y(要预测的)")
    # for t in range(len(x)):
    #     target = "（不算 loss）" if y[t] == -100 else idx2char[y[t].item()]
    #     print(f"{t:>3}   {idx2char[x[t].item()]:<6}→  {target}")
    # print("位置表:", tuple(model.pos_embed.weight.shape), " max_len:", model.max_len, "  期望 (128, 192) 128")
    ds = SFTDataset("sft_train.jsonl", char2idx)
    xs, ys = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate)))
    print("\n数据条数:", len(ds), "   一个 batch 的形状 x:", tuple(xs.shape), " y:", tuple(ys.shape))
    print("y 第 1 行的末尾:", ys[0, -15:].tolist())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_loader = DataLoader(SFTDataset("sft_val.jsonl", char2idx), batch_size=64, collate_fn=collate)
    # loss, ppl = evaluate(model, val_loader, device)
    # print(f"\nSFT 前  验证集（只算回答部分）  loss {loss:.3f}   PPL {ppl:.1f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_loader = DataLoader(SFTDataset("sft_train.jsonl", char2idx), batch_size=32,
                              shuffle=True, collate_fn=collate)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_ppl = evaluate(model, val_loader, device)
        mark = ""
        if va_loss < best:                      # 验证 loss 创新低，就保存
            best, mark = va_loss, "  *"
            cfg["seq_len"] = MAX_LEN
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": cfg}, "sft_best.pt")
        print(f"epoch {epoch:>2}   train loss {tr_loss:.3f}   val loss {va_loss:.3f}   val PPL {va_ppl:.1f}{mark}")


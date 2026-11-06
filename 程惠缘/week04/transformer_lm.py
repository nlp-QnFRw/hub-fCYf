"""
transformer_lm.py —— 字符级语言模型训练脚本（Transformer 版）

和同目录的 language_model.py 是**同一份脚本的两个版本**：
数据加载、字表、切块、划分、PPL 口径、存档格式全部一模一样，只把循环网络换成 Transformer。
所以两者的 PPL 可以直接比。

用法:
    python transformer_lm.py --epochs 20
    python transformer_lm.py --epochs 5 --d_model 128 --layers 3   # 快速试
    python transformer_lm.py --stride 1                            # 慢 64 倍，别怪没提醒

和 language_model.py 的三处有意差异:
    1. --stride 默认 64（不重叠切块）。原脚本默认 1，样本会被放大 64 倍。
    2. --corpus / --save 默认按脚本所在目录解析，从任何位置运行都不会找不到文件。
    3. 存档多写两个字段：arch="transformer" 和结构超参（d_model 等），
       这样 use_tf.py 能仅凭 .pt 重建模型。

产出:
    best_model_tf.pt   最佳验证 PPL 的模型（含权重/字表/超参）
"""
import math
import argparse
import glob
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────── 数据 ───────────────────────────
# 以下三个函数与 language_model.py 逐字一致，保证语料划分完全相同

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """按出现频率取前 max_vocab 个字（None = 全部字符）。"""
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """x = 连续 seq_len 个字符，y = 往后错一位。"""

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class TransformerLM(nn.Module):
    """pre-LN 的因果 Transformer：Embedding + 位置编码 + N×EncoderLayer + Linear。

    · causal 掩码保证第 t 个位置只看得到 t 及以前 —— 漏了它模型会"偷看答案"，
      PPL 会假性变好（而且好得离谱），这是语言模型最容易写错的地方。
    · 位置编码是学出来的（nn.Embedding），上限 = seq_len。
    """

    def __init__(self, vocab_size, d_model=192, nhead=4, num_layers=4,
                 ffn=512, dropout=0.1, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.register_buffer(
            "causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool),
                                 diagonal=1),
            persistent=False)

    def forward(self, x):
        length = x.size(1)
        pos = torch.arange(length, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        h = self.encoder(h, mask=self.causal[:length, :length])
        return self.fc(self.norm(h))          # (B, T, V)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64,
                        help="切块间隔：默认 64=不重叠（等于 seq_len）；设 1 会慢 64 倍")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--d_model",    type=int,   default=192)
    parser.add_argument("--nhead",      type=int,   default=4)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--ffn",        type=int,   default=512)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default=os.path.join(HERE, "*.txt"),
                        help="语料 glob；默认按脚本所在目录找，所以从任何位置运行都行")
    parser.add_argument("--save",       default=os.path.join(HERE, "best_model_tf.pt"))
    parser.add_argument("--threads",    type=int,   default=0,
                        help="0=用 torch 默认线程数（GPU 训练时无所谓）")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ── 数据准备（与 language_model.py 完全同一步骤，划分因此完全一致）──
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"样本: 训练 {len(train_ds):,}（{len(train_loader)} batch/epoch）  "
          f"验证 {len(val_ds):,}")

    # ── 模型 ──
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ffn=args.ffn,
        dropout=args.dropout,
        max_len=args.seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer,
                                        device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            saved_args = dict(vars(args))
            # 补一个 model 字段，和 language_model.py 存档的约定保持一致
            # （那边是 "rnn"/"lstm"，这里是 "transformer"），方便统一的加载器分流
            saved_args["model"] = "transformer"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": saved_args,
                # 下面两个字段是 language_model.py 没有的：让它能独立重建模型
                "arch": "transformer",
                "model_cfg": {
                    "d_model": args.d_model, "nhead": args.nhead,
                    "num_layers": args.num_layers, "ffn": args.ffn,
                    "dropout": args.dropout, "max_len": args.seq_len,
                    "vocab_size": vocab_size,
                },
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()

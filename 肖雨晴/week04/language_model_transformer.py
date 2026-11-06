"""
基于 Transformer 的字符级自回归语言模型，含 PPL 计算。
使用位置嵌入和因果注意力掩码，根据已有字符预测下一个字符。
用法:
    python language_model_transformer.py --epochs 20
    python language_model_transformer.py --embed_dim 128 --num_heads 4 --num_layers 2
"""

import math
import argparse
import glob
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="./*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_heads=4, dropout=0.3, max_seq_len=512):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须为正数且能被正整数 num_heads 整除")
        if hidden_dim <= 0 or num_layers <= 0 or max_seq_len <= 0:
            raise ValueError("hidden_dim、num_layers 和 max_seq_len 必须为正数")

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # 不使用双向编码：掩码让每个位置只能关注自身及之前的字符。
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        if x.dim() != 2 or not 1 <= x.size(1) <= self.max_seq_len:
            raise ValueError(f"输入形状必须为 (B, T)，且 1 <= T <= {self.max_seq_len}")
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        e = self.embed(x) * math.sqrt(self.embed_dim) + self.position_embed(positions)
        out = self.transformer(self.drop(e), mask=self.causal_mask[:seq_len, :seq_len])
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        raise ValueError("没有可用于训练或评估的字符，请增加语料或减小 seq_len")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transformer 字符级自回归语言模型")
    parser.add_argument("--model",      default="transformer", choices=["transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256, help="Transformer 前馈网络的隐藏维度")
    parser.add_argument("--num_heads",  type=int,   default=4, help="注意力头数，必须整除 embed_dim")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default=str(Path(__file__).resolve().parent / "corpus.txt"))
    parser.add_argument("--save",       default="best_transformer_model.pt", help="每轮覆盖保存的检查点路径")
    args = parser.parse_args()
    if args.seq_len <= 0:
        parser.error("--seq_len 必须大于 0")
    if not 0 < args.val_ratio < 1:
        parser.error("--val_ratio 必须在 0 和 1 之间")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("训练集和验证集各自至少需要 seq_len + 1 个字符，请增加语料或减小 --seq_len")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_name(save_path.name + ".tmp")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl

        # 每轮覆盖同一个检查点；写入完成后再替换，保留上一份完整文件。
        try:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ppl": va_ppl,
                "best_val_ppl": best_val_ppl,
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, temp_path)
            temp_path.replace(save_path)
        finally:
            temp_path.unlink(missing_ok=True)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}"
              f"  已保存 → {save_path}", flush=True)

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  最新一轮模型已保存至 {save_path}")


if __name__ == "__main__":
    main()

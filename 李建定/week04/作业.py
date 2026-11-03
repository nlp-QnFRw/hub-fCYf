
"""
字符级语言模型训练脚本，支持 RNN / LSTM / Transformer 切换，含 PPL 计算与早停机制。
用法:
    python language_model.py --model transformer --epochs 20
    python language_model.py --model lstm        --epochs 20
    python language_model.py --model rnn         --epochs 20
"""

import math
import argparse
import glob
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# CPU 多线程优化
torch.set_num_threads(16)        # 根据你 CPU 的核心数调整（通常 4, 8, 12, 16）
torch.set_num_interop_threads(2) # 负责不同算子之间并行的线程数


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
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

class LM(nn.Module):
    """原 RNN / LSTM 语言模型。"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))
        return logits


class PositionalEncoding(nn.Module):
    """学习式位置编码。"""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.pos = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        T = x.size(1)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        return self.drop(x + self.pos(pos_ids))


class TransformerLM(nn.Module):
    """因果（自回归）Transformer 语言模型。"""
    def __init__(self, vocab_size, seq_len, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(seq_len, d_model, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        e = self.embed(x)
        e = self.pos(e)
        T = x.size(1)
        mask = self.causal_mask[:T, :T]
        out = self.encoder(e, mask=mask)
        out = self.norm(out)
        logits = self.fc(out)
        return logits


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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 50))
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer",
                        choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    
    # 早停与续训参数
    parser.add_argument("--patience",   type=int,   default=3, 
                        help="早停耐心值：连续多少轮验证集不下降就停止训练")
    parser.add_argument("--resume",     action="store_true", 
                        help="是否加载已有模型继续训练（断点续训）")

    # Transformer 专属
    parser.add_argument("--nhead",      type=int,   default=4)
    parser.add_argument("--dim_ff",     type=int,   default=512)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
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
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 模型
    if args.model == "transformer":
        assert args.embed_dim == args.hidden_dim, "Transformer 要求 embed_dim == hidden_dim"
        assert args.embed_dim % args.nhead == 0, "embed_dim 必须能被 nhead 整除"
        model = TransformerLM(
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            d_model=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_ff,
            dropout=args.dropout,
        ).to(device)
    else:
        model = LM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            model_type=args.model,
            dropout=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 断点续训：如果指定了 --resume 且文件存在，则加载权重 ──
    start_epoch = 1
    if args.resume and os.path.exists(args.save):
        print(f"\n[续训模式] 发现已保存的模型 {args.save}，正在加载权重...")
        checkpoint = torch.load(args.save, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        # 由于是续训，学习率通常建议调小，这里也可以直接使用 args.lr
        print("权重加载成功，准备继续训练。\n")

    best_val_ppl = float("inf") if not args.resume else checkpoint.get("best_val_ppl", float("inf"))
    patience_counter = 0  # 记录连续没进步的轮数

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            patience_counter = 0  # 重置早停计数器
            marker = "  *"
            
            # 保存模型时，把 best_val_ppl 一起存进去，方便续训时参考
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
                "best_val_ppl": best_val_ppl,
            }, args.save)
        else:
            patience_counter += 1
            marker = f"  (early-stop: {patience_counter}/{args.patience})"

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

        # ── 早停判断 ──
        if patience_counter >= args.patience:
            print(f"\n[早停触发] 验证集 PPL 连续 {args.patience} 轮未下降，模型已收敛，停止训练。")
            break

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()

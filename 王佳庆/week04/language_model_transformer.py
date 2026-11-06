"""
字符级语言模型训练脚本，直接使用完整 torch.nn.Transformer（encoder-decoder），含 PPL 计算。
用法:
    python language_model_v2.py --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

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

class PositionalEncoding(nn.Module):
    """正弦位置编码：Transformer 本身对顺序不敏感，需要显式注入位置信息。"""

    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LM(nn.Module):
    """
    直接使用完整 torch.nn.Transformer（encoder-decoder）的字符级语言模型。

    数据流（teacher forcing）:
        encoder 输入 src = x            —— 用因果掩码编码上文
        decoder 输入 tgt = y（=x 右移一位）—— 预测下一字符
    decoder 每个位置 t 综合「encoder 的上文记忆」和「tgt 中 <t 的已生成字符」，
    预测 y[t]（即 x[t+1]），这就是下一字符预测。

    三个掩码都必须是因果掩码，否则会泄露未来信息：
        src_mask    —— encoder 位置 t 只看 x[:t+1]
        tgt_mask    —— decoder 位置 t 只看 y[:t]
        memory_mask —— decoder 位置 t 只看 encoder 的前 t 个输出
                        （不加的话交叉注意力能看到 x[t+1:]，直接偷看答案）
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=4096)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # 输入输出为 (batch, seq, feature)
            norm_first=True,    # Pre-LN，训练更稳定
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        S, T = src.size(1), tgt.size(1)      # S = seq_len, T = seq_len - 1（teacher forcing 切掉一位）
        src_mask = nn.Transformer.generate_square_subsequent_mask(S, device=src.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)
        memory_mask = torch.triu(
            torch.ones(T, S, device=src.device, dtype=torch.bool), diagonal=1
        )
        e = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))   # (B, S, d_model)
        t = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))   # (B, T, d_model)
        out = self.transformer(
            e, t,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )                                                  # (B, T, d_model)
        logits = self.fc(out)                              # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, scheduler=None, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, x)   # src=x, tgt=y[:-1]（teacher forcing）
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # Transformer 训练标配：梯度裁剪，防止早期梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()   # 每个 step 调整学习率（warmup + cosine 退火）

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--d_model",    type=int,   default=128, help="Transformer 隐藏维度（需能被 nhead 整除）")
    parser.add_argument("--nhead",      type=int,   default=4,    help="多头注意力头数")
    parser.add_argument("--dim_ff",     type=int,   default=512,  help="前馈层维度")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--warmup",     type=int,   default=1000, help="学习率线性预热的步数")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_v2.pt")
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    if args.d_model % args.nhead != 0:
        raise ValueError(f"d_model ({args.d_model}) 必须能被 nhead ({args.nhead}) 整除")
    model = LM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度：前 warmup 步线性升温，之后按 cosine 退火（Transformer 常用配置）
    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, len(train_loader) * args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, scheduler, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()

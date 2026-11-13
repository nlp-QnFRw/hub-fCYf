"""
字符级语言模型训练脚本，支持 RNN / LSTM / Transformer 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
    python language_model.py --model transformer    # 默认超参和参考答案 transformer_lm.py 一致
"""

import math
import time
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

class LM(nn.Module):
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
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits

class TransformerLM(nn.Module):
    """
    tricks=False：最初的版本（旧的 checkpoint 都是这样训练出来的）
    tricks=True ：加上参考答案 transformer_lm.py 里的几个技巧
        - GELU 激活
        - Pre-LN（norm_first=True）+ 最后一层再接一个 LayerNorm
        - 权重共享：输出层 fc 和字向量 embed 共用同一张表
        - 小初始化 std=0.02：防止共享后 logits 太大、初始 loss 爆炸
    """
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout, tricks=False):
        super().__init__()
        self.max_len = max_len
        # 1. 字向量：和 LSTM 版一样
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 2. 位置向量：第 0, 1, 2, ..., max_len-1 个位置，每个位置学一个向量
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        # 3. 定义"一层" Transformer 长什么样
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,#惯例是取 embed_dim 的 4 倍
            dropout=dropout,
            activation="gelu" if tricks else "relu",
            norm_first=tricks,          # Pre-LN：LayerNorm 放在注意力/FFN 之前，训练更稳
            batch_first=True,
        )
        # 4. 把这一层叠 num_layers 次（Pre-LN 时最后还要再接一个 LayerNorm）
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim) if tricks else None,
            enable_nested_tensor=False,
        )
        self.drop = nn.Dropout(dropout)
        # 5. 输出层
        self.fc = nn.Linear(embed_dim, vocab_size, bias=not tricks)
        if tricks:
            # 6. 权重共享：出口直接用入口那张表（两个名字指向同一张表）
            self.fc.weight = self.embed.weight
            # 7. 小初始化：默认 ±1 太大，共享后 logits 会爆炸，改成 ±0.02（GPT-2 的做法）
            nn.init.normal_(self.embed.weight, std=0.02)
            nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x):
        # x: (B, T)
        x = x[:, -self.max_len:]   # 超过 max_len 就只保留最后 max_len 个字（输入法会用到，先照抄）
        B, T = x.shape
        # 1. 位置编号：0, 1, 2, ..., T-1
        positions = torch.arange(T, device=x.device)
        # 2. 字向量 + 位置向量
        h = self.drop(self.embed(x) + self.pos_embed(positions))
        # 3. 因果掩码：T×T 的上三角矩阵
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        # 4. 送进 Transformer
        out = self.transformer(h, mask=mask, is_causal=True)
        # 5. 输出层（和 LSTM 版一样）
        return self.fc(self.drop(out))



# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, clip=0.0):
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
            if clip > 0:
                # 梯度裁剪：梯度总长度超过 clip 就等比例缩小（只缩步长，不改方向）
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # 下面 default=None 的参数，默认值按模型类型决定，见 MODEL_DEFAULTS
    parser.add_argument("--model",      default="lstm", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=None, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=None)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=None)
    parser.add_argument("--num_heads",  type=int,   default=4,    help="Transformer 注意力头数，embed_dim 要能被它整除")
    parser.add_argument("--ff_dim",     type=int,   default=None, help="Transformer FFN 升维后的维度，默认 4×embed_dim")
    parser.add_argument("--dropout",    type=float, default=None)
    parser.add_argument("--clip",       type=float, default=None, help="梯度裁剪阈值，0=不裁剪")
    parser.add_argument("--tricks",     action=argparse.BooleanOptionalAction, default=None,
                        help="Transformer 是否启用 Pre-LN/GELU/权重共享/小初始化（--no-tricks 关闭）")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default=None)
    args = parser.parse_args()

    # 不同模型的默认超参：transformer 和参考答案 transformer_lm.py 一致，rnn/lstm 保持原来的默认值
    MODEL_DEFAULTS = {
        "transformer": dict(epochs=6,  stride=64, embed_dim=192, num_layers=4, dropout=0.1,
                            clip=1.0, tricks=True,  save="tf_best.pt"),
        "lstm":        dict(epochs=20, stride=1,  embed_dim=128, num_layers=2, dropout=0.3,
                            clip=0.0, tricks=False, save="best_model.pt"),
    }
    MODEL_DEFAULTS["rnn"] = MODEL_DEFAULTS["lstm"]
    for k, v in MODEL_DEFAULTS[args.model].items():
        if getattr(args, k) is None:        # 命令行没指定的，才用默认值
            setattr(args, k, v)
    if args.ff_dim is None:
        args.ff_dim = 4 * args.embed_dim

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
    if args.model == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_layers=args.num_layers,
            max_len=args.seq_len,      # 位置表的行数 = 训练时的句子长度
            dropout=args.dropout,
            tricks=args.tricks,
        ).to(device)
        print(f"Transformer: d={args.embed_dim} 层数={args.num_layers} 头数={args.num_heads} "
              f"FFN={args.ff_dim} dropout={args.dropout} tricks={args.tricks} clip={args.clip}")
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

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'Time(s)':>8}")
    print("-" * 66)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True, clip=args.clip)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        epoch_time = time.time() - t0   # 训练 + 验证一轮的耗时，用来对比速度

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}  {epoch_time:>8.1f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()

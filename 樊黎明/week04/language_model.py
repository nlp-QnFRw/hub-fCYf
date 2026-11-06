"""
字符级语言模型训练脚本，支持 RNN / LSTM / Transformer 切换，含 PPL 计算。
用法:
    python language_model.py --model transformer --epochs 20
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
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


# ─────────────────────────── Transformer 核心模块 ───────────────────────────

class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码：为序列注入绝对位置信息，解决Transformer无法感知语序的问题。
    每个位置生成唯一的编码向量，叠加到词嵌入上，让模型识别字符的先后顺序。
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: [seq_len, batch_size, embedding_dim]
        输出叠加位置编码后的向量，形状保持不变
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    """
    纯解码器风格的字符级Transformer语言模型，适配自回归生成任务。
    完全基于PyTorch官方nn.TransformerDecoder实现，无需手写注意力底层逻辑。
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, nhead, dropout, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        # 字符嵌入层：将Token ID映射为稠密向量
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 位置编码层：补充序列顺序信息
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)
        # Transformer解码器层：堆叠多层自注意力+前馈网络
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False  # 保持和位置编码的seq_len优先输入格式兼容
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 输出投射层：将隐层向量映射回词表维度，得到每个位置的字符预测得分
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        输入x形状: [batch_size, seq_len]
        输出logits形状: [batch_size, seq_len, vocab_size]
        """
        seq_len = x.size(1)
        # 生成自注意力掩码：防止每个位置看到未来的字符，保证自回归特性
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # 嵌入层 + 位置编码，调整维度适配Transformer输入格式
        e = self.embed(x).permute(1, 0, 2) * math.sqrt(self.embed_dim)
        e = self.pos_encoder(e)

        # 解码器前向传播，传入全0记忆向量适配纯语言模型场景
        memory = torch.zeros(1, x.size(0), self.embed_dim).to(x.device)
        out = self.transformer_decoder(e, memory, tgt_mask=mask)

        # 调整维度后投射到词表空间
        logits = self.fc(out.permute(1, 0, 2))
        return logits


# ─────────────────────────── 兼容原有RNN/LSTM的模型入口 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout, nhead=4):
        super().__init__()
        self.model_type = model_type
        if model_type == "transformer":
            # Transformer模式：自动适配头数，保证embed_dim可被nhead整除
            assert embed_dim % nhead == 0, "Transformer的embed_dim必须能被nhead整除"
            self.core = TransformerLM(vocab_size, embed_dim, hidden_dim, num_layers, nhead, dropout)
        else:
            # 保留原有RNN/LSTM逻辑，实现无缝切换
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
        if self.model_type == "transformer":
            return self.core(x)
        else:
            e = self.drop(self.embed(x))
            out, _ = self.rnn(e)
            logits = self.fc(self.drop(out))
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
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--nhead",      type=int,   default=4, help="Transformer注意力头数")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="week04/*.txt")
    parser.add_argument("--save",       default="week04/best_model.pt")
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

    # 模型初始化
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
        nhead=args.nhead
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
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
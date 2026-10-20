"""字符级、仅解码器结构的 Mini GPT 训练与文本生成脚本。

训练模型：
    python language_model.py --epochs 20

使用指定采样策略生成文本：
    python language_model.py --task generate --prompt "Once upon a time" \
        --sampling top_p --top_p 0.9
    python language_model.py --task generate --prompt "Once upon a time" \
        --sampling beam_search --beam_size 4
"""

import argparse
import glob
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


NUM_LAYERS = 6


# ─────────────────────────── 数据 ───────────────────────────

def detect_text_encoding(path):
    """根据文件开头的 BOM 判断常见 Unicode 编码，默认使用 UTF-8。"""
    with open(path, "rb") as f:
        bom = f.read(4)

    if bom.startswith((b"\xff\xfe", b"\xfe\xff")):
        # 使用 utf-16 而不是固定的 utf-16-le，让 Python 自动读取并移除 BOM。
        return "utf-16"
    if bom.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def load_corpus(pattern="./*.txt"):
    """自动识别 UTF-8/UTF-16 语料编码，并拼接匹配到的全部文本。"""
    texts = []
    for path in glob.glob(pattern):
        encoding = detect_text_encoding(path)
        with open(path, encoding=encoding) as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    """建立字符到编号、编号到字符的双向映射。"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """把连续文本切分为定长的输入序列和右移一位的预测目标。"""

    def __init__(self, text, char2idx, seq_len):
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # 输入为 text[i:i+T]，标签为 text[i+1:i+T+1]，即预测下一个字符。
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── Mini GPT ───────────────────────────

class CausalSelfAttention(nn.Module):
    """手动实现带下三角因果掩码的多头自注意力。"""

    def __init__(self, d_model, nhead, block_size, dropout):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"embed_dim ({d_model}) must be divisible by nhead ({nhead})")

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        # 合并 Q、K、V 的线性变换，减少三次独立矩阵乘法的开销。
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # 下三角区域为 True，保证当前位置只能关注自身及之前的字符。
        # 注册为 buffer 后会自动随模型迁移到 CPU/GPU，但不参与梯度更新。
        causal_mask = torch.tril(
            torch.ones(block_size, block_size, dtype=torch.bool)
        ).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x):
        batch_size, length, d_model = x.shape

        # (B, T, 3C) -> (3, B, H, T, D)，其中 C = H * D。
        qkv = self.qkv(x)
        qkv = qkv.view(
            batch_size, length, 3, self.nhead, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        # 缩放点积注意力：(B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)。
        attention_scores = (query @ key.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:, :, :length, :length]
        # 将未来位置设为负无穷，使其经过 softmax 后的权重为 0。
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # 对 Value 加权求和，再合并所有注意力头为 (B, T, C)。
        output = attention_weights @ value
        output = output.transpose(1, 2).contiguous().view(
            batch_size, length, d_model
        )
        output = self.output_projection(output)
        return self.output_dropout(output)


class FeedForward(nn.Module):
    """逐位置前馈网络：升维、GELU 激活，再投影回模型维度。"""

    def __init__(self, d_model, ff_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """使用 Pre-LayerNorm 和残差连接的 Transformer Block。"""

    def __init__(self, d_model, nhead, ff_dim, block_size, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model=d_model,
            nhead=nhead,
            block_size=block_size,
            dropout=dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x):
        # Pre-LN 结构通常比先做残差再归一化的 Post-LN 更容易稳定训练。
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class MiniGPT(nn.Module):
    """由 6 个自定义 Transformer Block 组成的单向语言模型。"""

    def __init__(self, vocab_size, d_model, nhead, ff_dim, block_size, dropout):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"embed_dim ({d_model}) must be divisible by nhead ({nhead})")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        # ModuleList 会正确注册每一层参数，便于优化器更新和 checkpoint 保存。
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, ff_dim, block_size, dropout)
            for _ in range(NUM_LAYERS)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 输入嵌入与输出分类头共享权重，这是 GPT 常用的权重绑定方式。
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        _, length = token_ids.shape
        if length > self.block_size:
            raise ValueError(
                f"input length {length} exceeds model block size {self.block_size}"
            )

        # Token Embedding 表示字符内容，Position Embedding 表示字符所在位置。
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)  # 输出形状：(批大小, 序列长度, 词表大小)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """运行一轮训练或验证，并返回平均交叉熵与困惑度 PPL。"""
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # 将所有时间步展平后统一计算下一个字符的交叉熵。
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            # 验证阶段只计算指标，不更新参数。
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        raise ValueError("dataset has no sequences; use more corpus text or a smaller --seq_len")
    avg_loss = total_loss / total_tokens
    # 限制指数输入，避免模型训练初期因损失过大造成数值溢出。
    ppl = math.exp(min(avg_loss, 80))
    return avg_loss, ppl


def save_checkpoint(path, model, char2idx, idx2char, args):
    """同时保存模型权重、词表和结构配置，供后续生成时完整恢复。"""
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "model_config": {
            "d_model": args.embed_dim,
            "nhead": args.nhead,
            "ff_dim": args.hidden_dim,
            "block_size": args.seq_len,
            "dropout": args.dropout,
            "num_layers": NUM_LAYERS,
        },
        "args": vars(args),
    }, path)


def train(args, device):
    """准备字符级数据，训练 Mini GPT，并保存验证集 PPL 最优的模型。"""
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"语料为空或未找到文件: {args.corpus}")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 按行切分训练集和验证集，避免同一行的重叠窗口同时出现在两者中。
    lines = text.splitlines() or [text]
    random.shuffle(lines)
    if len(lines) > 1:
        split = max(1, min(len(lines) - 1, int(len(lines) * (1 - args.val_ratio))))
        train_text = "\n".join(lines[:split])
        val_text = "\n".join(lines[split:])
    else:
        # 单行语料无法按行划分，因此复用为验证集，使小语料仍可训练。
        train_text = val_text = lines[0]

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "训练集或验证集太短，无法形成一个序列；请减小 --seq_len 或增加语料。"
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=args.embed_dim,
        nhead=args.nhead,
        ff_dim=args.hidden_dim,
        block_size=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}  model: MiniGPT ({NUM_LAYERS} layers)")
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        # 只保存验证集困惑度最低的 checkpoint。
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            save_checkpoint(args.checkpoint, model, char2idx, idx2char, args)

        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
            f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}"
        )

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.checkpoint}")


# ─────────────────────────── 文本生成 ───────────────────────────

def filter_logits(logits, sampling, top_k=50, top_p=0.9):
    """对一维 logits 执行 Top K 或 Top P（核采样）过滤。"""
    if sampling == "top_k":
        # 只保留概率最高的 K 个候选字符。
        k = min(top_k, logits.numel())
        threshold = torch.topk(logits, k).values[-1]
        return logits.masked_fill(logits < threshold, float("-inf"))

    if sampling == "top_p":
        # 按概率降序排列，保留累计概率刚好覆盖 top_p 的最小候选集合。
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_to_remove = cumulative_probs > top_p
        # 右移删除掩码，保留第一个使累计概率超过阈值的字符。
        sorted_to_remove[1:] = sorted_to_remove[:-1].clone()
        sorted_to_remove[0] = False
        remove = torch.zeros_like(sorted_to_remove).scatter(
            0, sorted_indices, sorted_to_remove
        )
        return logits.masked_fill(remove, float("-inf"))

    return logits


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, device, sampling="greedy",
             temperature=1.0, top_k=50, top_p=0.9, beam_size=4):
    """根据提示文本逐字符生成，支持 greedy、beam、Top P 和 Top K。"""
    model.eval()
    if not prompt_ids:
        raise ValueError("prompt 不能为空")

    if sampling in ("beam", "beam_search"):
        # 每轮扩展所有 beam，并仅保留累计对数概率最高的 beam_size 条路径。
        beams = [(list(prompt_ids), 0.0)]
        for _ in range(max_new_tokens):
            candidates = []
            for sequence, score in beams:
                # 模型位置嵌入长度有限，只输入最近 block_size 个字符。
                context = sequence[-model.block_size:]
                x = torch.tensor([context], dtype=torch.long, device=device)
                # 温度越小，概率分布越尖锐；温度越大，候选概率越均匀。
                next_logits = model(x)[0, -1] / temperature
                log_probs = torch.log_softmax(next_logits, dim=-1)
                values, indices = torch.topk(log_probs, min(beam_size, log_probs.numel()))
                for value, token_id in zip(values.tolist(), indices.tolist()):
                    candidates.append((sequence + [token_id], score + value))
            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
        return beams[0][0]

    # greedy、Top P、Top K 共用逐字符生成循环。
    sequence = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = sequence[-model.block_size:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / temperature

        if sampling == "greedy":
            # 贪心解码始终选择当前概率最高的字符。
            next_token = torch.argmax(logits).item()
        else:
            logits = filter_logits(logits, sampling, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            # 从过滤后的概率分布中随机抽取下一个字符。
            next_token = torch.multinomial(probs, num_samples=1).item()
        sequence.append(next_token)

    return sequence


def load_checkpoint(path, device):
    """从 checkpoint 恢复模型结构、权重和字符词表。"""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 兼容尚不支持 weights_only 参数的旧版 PyTorch
        checkpoint = torch.load(path, map_location=device)

    config = checkpoint["model_config"]
    if config.get("num_layers") != NUM_LAYERS:
        raise ValueError(f"checkpoint must contain a {NUM_LAYERS}-layer model")
    model = MiniGPT(
        vocab_size=len(checkpoint["char2idx"]),
        d_model=config["d_model"],
        nhead=config["nhead"],
        ff_dim=config["ff_dim"],
        block_size=config["block_size"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["char2idx"], checkpoint["idx2char"]


def generate_from_checkpoint(args, device):
    """校验提示文本、调用生成函数，并把字符编号还原为文本。"""
    model, char2idx, idx2char = load_checkpoint(args.checkpoint, device)
    if args.prompt is None:
        raise ValueError("生成时请通过 --prompt 提供起始文本")
    unknown = sorted(set(args.prompt) - set(char2idx))
    if unknown:
        raise ValueError(f"prompt 中包含词表外字符: {''.join(unknown)!r}")
    prompt_ids = [char2idx[c] for c in args.prompt]
    result = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        device=device,
        sampling=args.sampling,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        beam_size=args.beam_size,
    )
    print("".join(idx2char[token_id] for token_id in result))


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="训练或使用六层 Mini GPT 生成文本")
    parser.add_argument("--task", choices=["train", "generate"], default="train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256, help="前馈网络隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parent / "corpus.txt"),
        help="语料文件路径或 glob 匹配表达式",
    )
    parser.add_argument("--checkpoint", "--save", dest="checkpoint", default="best_model.pt")
    parser.add_argument("--prompt", default=None, help="文本生成的起始提示")
    parser.add_argument(
        "--sampling",
        choices=["greedy", "beam_search", "beam", "top_p", "top_k"],
        default="greedy",
    )
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度；值越低输出越稳定，值越高随机性越强",
    )
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--beam_size", type=int, default=4)
    args = parser.parse_args()

    if args.seq_len < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("--seq_len、--batch_size 和 --epochs 必须为正数")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val_ratio 必须在 [0, 1) 范围内")
    if args.max_new_tokens < 0 or args.beam_size < 1 or args.top_k < 1:
        parser.error("--max_new_tokens 不能为负数，--beam_size 和 --top_k 必须为正数")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p 必须在 (0, 1] 范围内")
    if args.temperature <= 0:
        parser.error("--temperature 必须大于 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "train":
        train(args, device)
    else:
        print(f"device: {device}  sampling: {args.sampling}")
        generate_from_checkpoint(args, device)


if __name__ == "__main__":
    main()

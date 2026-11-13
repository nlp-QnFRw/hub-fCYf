import argparse
import glob
import math
import os
import random
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# 1. 数据
# ═══════════════════════════════════════════════════════════════

def load_corpus(pattern):
    """读语料，支持通配符；errors='ignore' 容错坏字节，sorted 保证可复现。"""
    text = ""
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    if not text:
        raise FileNotFoundError(f"没读到语料: {pattern}")
    return text


def build_vocab(text, max_vocab):
    """字符 → id；max_vocab>0 时保留最高频的 N 个字符。"""
    chars = [c for c, _ in Counter(text).most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    return char2idx, {i: c for c, i in char2idx.items()}


class CharDataset(Dataset):
    """x = 连续 seq_len 个字符，y = 错一位；stride 控制切块间隔。"""

    def __init__(self, text, char2idx, seq_len, stride):
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = max(1, stride)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        return self.data[s:s + self.seq_len], self.data[s + 1:s + self.seq_len + 1]


# ═══════════════════════════════════════════════════════════════
# 2. 模型
# ═══════════════════════════════════════════════════════════════

class TransformerLM(nn.Module):
    """
    Token Emb + 可学习 Pos Emb + N×TransformerEncoderLayer(Pre-LN) + 输出头（权重共享）。
    causal=True 时传下三角掩码，等价 GPT 的因果注意力。
    """

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=512,
                 dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying

        nn.init.normal_(self.tok.weight, std=0.02)  # GPT-2 初始化，防初始 loss 飙到 ~19
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx)                            # (B, T, d)
        if self.use_pos:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                idx.size(1), device=idx.device)
        x = self.encoder(x, mask=mask)
        return self.head(x)                          # (B, T, V) logits


# ═══════════════════════════════════════════════════════════════
# 3. 评估 / 生成 / 因果性验证
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, last_only=False):
    """返回 (loss, ppl)；last_only=True 只看每段最后一个位置（抄不到答案）。"""
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if last_only:
            logits, y = logits[:, -1, :], y[:, -1]
        total += float(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
        tokens += y.numel()
    loss = total / tokens
    return loss, math.exp(loss)


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, n_chars, seq_len,
             temperature=0.8, top_k=40, device="cpu"):
    """逐 token 自回归续写；temperature<=0.01 走贪心。"""
    model.eval()
    ids = [char2idx[ch] for ch in prompt if ch in char2idx]
    if not ids:
        return "（开头里的字都不在字表里）"
    out = list(ids)
    for _ in range(n_chars):
        logits = model(torch.tensor([out[-seq_len:]], device=device))[0, -1]
        if temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        out.append(nxt)
    return "".join(idx2char[i] for i in out[len(ids):])


def verify_causal(args):
    """
    因果性验证：扰动最后一个 token，比较各位置输出变化。
    有掩码时前 T-1 个位置变化必须严格为 0；无掩码时全部跟着变。
    """
    device = torch.device("cpu")   # 小模型验证用 CPU 足够，结果确定
    T, V = args.seq_len, 100
    torch.manual_seed(0)

    for causal in [True, False]:
        m = TransformerLM(V, d_model=32, n_head=4, n_layer=2, max_len=T,
                          dropout=0.0, use_pos=True, causal=causal).to(device).eval()
        x = torch.randint(0, V, (1, T))
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % V          # 只改最后一个 token
        with torch.no_grad():
            y1, y2 = m(x), m(x2)
        diff = (y1 - y2).abs().amax(dim=-1)[0]   # 每个位置的最大绝对变化 (T,)
        tag = "有掩码" if causal else "无掩码"
        print(f"{tag}: 前 T-1 位置最大变化 = {diff[:-1].max().item():.3e}，"
              f"末位置变化 = {diff[-1].item():.3e}")
    print("\n结论：有掩码时前 T-1 位置变化严格为 0 → 未来信息不可见；"
          "无掩码时全部跟着变 → 泄漏被直接观测到。")


# ═══════════════════════════════════════════════════════════════
# 4. 训练
# ═══════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text = load_corpus(args.corpus)
    if args.max_chars and len(text) > args.max_chars:
        text = text[:args.max_chars]
    char2idx, idx2char = build_vocab(text, args.max_vocab)
    V = len(char2idx)
    print(f"device: {device}  语料 {len(text):,} 字  字表 {V}")

    # 按行打乱再切分：避免同一段文字同时落进训练段和验证段（验证集泄漏）
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    tr = CharDataset("\n".join(lines[:split]), char2idx, args.seq_len, args.stride)
    va = CharDataset("\n".join(lines[split:]), char2idx, args.seq_len, args.stride)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TransformerLM(V, args.d_model, args.heads, args.layers, args.seq_len,
                          args.dropout, use_pos=not args.no_pos,
                          causal=not args.no_mask).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: d={args.d_model} h={args.heads} L={args.layers} Pre-LN  "
          f"位置编码={'有' if not args.no_pos else '无'}  "
          f"因果掩码={'有' if not args.no_mask else '**无（泄漏实验）**'}  "
          f"参数量 {n_par/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL(全位置)':>15} {'ValPPL(末位)':>13} {'tok/s':>9}")
    print("-" * 60)

    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * y.numel()
            seen += y.numel()
        dt = time.time() - st

        _, v_ppl = evaluate(model, val_loader, device, last_only=False)
        _, v_ppl_last = evaluate(model, val_loader, device, last_only=True)
        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args)}, args.save)
        print(f"{ep:>6} {tot/seen:>10.4f} {v_ppl:>15.2f} {v_ppl_last:>13.2f} "
              f"{seen/dt:>9,.0f}{mark}")

    print(f"\n最佳验证 PPL（全位置）: {best:.2f}  已保存 {args.save}")

    # 训练结束后自动演示续写，方便对比三种配置
    print(f"\n续写演示（prompt=今天天气，40 字）:")
    print("  " + generate(model, char2idx, idx2char, "今天天气", 40,
                         args.seq_len, temperature=0.65, top_k=40, device=device))


# ═══════════════════════════════════════════════════════════════
# 5. 命令行入口
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    default_corpus = os.path.join(HERE, "corpus_sample.txt")
    ap.add_argument("--corpus", default=default_corpus)
    ap.add_argument("--max_chars", type=int, default=0)
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mask", dest="no_mask", action="store_true",
                    help="去掉因果掩码（泄漏实验）")
    ap.add_argument("--no-pos", dest="no_pos", action="store_true",
                    help="去掉位置编码")
    ap.add_argument("--save", default="tf_best.pt")
    ap.add_argument("--generate", default=None, help="给一段开头，续写文本")
    ap.add_argument("--ckpt", default="tf_best.pt")
    ap.add_argument("--n_chars", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--verify-causal", dest="verify_causal", action="store_true",
                    help="跑因果性验证实验（不训练）")
    args = ap.parse_args()

    if args.verify_causal:
        verify_causal(args)
        return

    if args.generate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.ckpt, map_location=device)
        c = ck["args"]
        char2idx, idx2char = ck["char2idx"], ck["idx2char"]
        model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"],
                              c["seq_len"], 0.0,
                              use_pos=not c.get("no_pos", False),
                              causal=not c.get("no_mask", False)).to(device)
        model.load_state_dict(ck["model_state"])
        print(f"输入: {args.generate}")
        print(f"续写: {generate(model, char2idx, idx2char, args.generate,
                                args.n_chars, c['seq_len'],
                                args.temperature, args.top_k, device)}")
        return

    train(args)


if __name__ == "__main__":
    main()

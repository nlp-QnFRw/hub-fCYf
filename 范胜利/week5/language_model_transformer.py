# -*- coding: utf-8 -*-
"""
language_model_transformer.py —— 字符级 Transformer：两阶段训练（预训练 → SFT）

【为什么要两个阶段】
    第 5 周 PPT 里的那句话：**知识来自预训练，形式来自 SFT**。
    · 只在 3000 条 SFT 数据上从零训练 → 只能学会「像回答的格式」，答不出内容
      （实测：第一个回答字的准确率只有 1.5%，生成时第一个字就错，后面全跑偏）
    · 先在 600 万字通用语料上做 next-token 预训练 → 模型学会语言统计与常识
      → 再用 SFT 调行为，同样规模下答对率会明显提升

【两个阶段共用一张字表】
    预训练和 SFT 必须用同一个 char2idx，否则 <|user|> 在两边含义不同，权重直接作废。
    所以：stage=pretrain 时构建字表（语料字符 ∪ SFT 数据字符）并存进 ckpt；
         stage=sft 时**优先从 --init 的 ckpt 里读字表**，而不是重新构建。

【两阶段做的事】
    ① 预训练（stage=pretrain）
       数据：corpus_general.txt（602 万字）切窗口
       目标：每个位置都预测下一个字（next-token）
       产出：pretrain_best.pt
    ② SFT（stage=sft，默认）
       数据：sft_train.jsonl（3000 条 instruction/output）
       目标：只在回答段算 loss（prompt 打 -100）
       产出：sft_best.pt

用法:
    # 阶段一：预训练（3 轮，MPS 上约 20 分钟；先 --max_chars 30 万试跑）
    python language_model_transformer.py --stage pretrain --epochs 3
    python language_model_transformer.py --stage pretrain --epochs 1 --max_chars 300000 --save pretrain_smoke.pt

    # 阶段二：SFT（默认会自动加载 pretrain_best.pt）
    python language_model_transformer.py --stage sft --epochs 8

    # 只做 SFT（没有预训练权重时，就是之前的从零 SFT）
    python language_model_transformer.py --stage sft --init ""

    # 测试 / 交互
    python language_model_transformer.py --generate "计算 12 和 35 的平均数"
    python language_model_transformer.py --stage sft --eval-first-token   # 看「第一个字」的准确率
"""

import argparse
import glob
import json
import math
import os
import random
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────── 特殊 token 与模板 ───────────────────────────

PAD, USER, ASSISTANT, END = 0, 1, 2, 3
SPECIALS = ["<pad>", "<|user|>", "<|assistant|>", "<|end|>"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(pattern=os.path.join(HERE, "corpus_general.txt")):
    """读预训练语料；相对路径先试当前目录，再试脚本目录。"""
    paths = sorted(glob.glob(pattern))
    if not paths and not os.path.isabs(pattern):
        paths = sorted(glob.glob(os.path.join(HERE, pattern)))
    text = ""
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    return text


def build_vocab(rows, corpus_text="", max_vocab=0):
    """
    字表 = 4 个特殊 token（固定 id 0~3）+ 语料与 SFT 数据里出现过的字符（按频率降序）。

    把语料也算进来，是为了让两阶段共用同一张表：预训练时见过的高频字，
    SFT 阶段不会因为「SFT 数据里没有」而被丢掉。
    """
    from collections import Counter
    cnt = Counter(corpus_text)
    for r in rows:
        cnt.update(r.get("instruction", ""))
        cnt.update(r.get("output", ""))
    chars = [c for c, _ in cnt.most_common(max_vocab)] if max_vocab else \
            [c for c, _ in cnt.most_common()]
    char2idx = {s: i for i, s in enumerate(SPECIALS)}
    for c in chars:
        char2idx[c] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def encode(text, char2idx):
    return [char2idx[c] for c in text if c in char2idx]


def decode(ids, idx2char):
    return "".join(idx2char.get(i, "") for i in ids if i >= len(SPECIALS))


# ─────────────────────────── 数据集：预训练 / SFT ───────────────────────────

class CharDataset(Dataset):
    """预训练用：连续 seq_len 个字预测下一个（stride=seq_len 不重叠）。"""

    def __init__(self, text, char2idx, seq_len, stride=None):
        self.seq_len = seq_len
        self.stride = max(1, stride or seq_len)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        return (self.data[s: s + self.seq_len],
                self.data[s + 1: s + self.seq_len + 1])


def build_example(row, char2idx):
    """
    SFT 样本：ids = [USER] + 指令 + [ASST] + 回答 + [END]

    标签必须错一位（第 i 个 logits 预测 ids[i+1]），并且 prompt 段打 -100：
        labels = [-100]*(prompt_len-1) + answer + [-100]
    唯一「在 prompt 里但参与 loss」的位置是最后一个 prompt token（<|assistant|>），
    它负责预测回答的第一个字 —— 这个位置学得好不好，直接决定生成能不能起步。
    """
    prompt = [USER] + encode(row.get("instruction", ""), char2idx) + [ASSISTANT]
    answer = encode(row.get("output", ""), char2idx) + [END]
    ids = prompt + answer
    labels = [-100] * (len(prompt) - 1) + answer + [-100]
    return ids, labels


class SFTDataset(Dataset):
    def __init__(self, rows, char2idx, seq_len):
        self.samples = []
        for r in rows:
            ids, labels = build_example(r, char2idx)
            if len(ids) > seq_len:            # 超长：保留结尾的 <|end|> 作为学习目标
                ids = ids[:seq_len]
                labels = labels[:seq_len]
                ids[-1] = END
                labels[-2] = END
                labels[-1] = -100
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_sft(batch):
    """补到本 batch 最长：输入补 PAD，标签补 -100（补位不参与 loss）。"""
    L = max(len(ids) for ids, _ in batch)
    x = torch.zeros(len(batch), L, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
    return x, y


# ─────────────────────────── 模型 ───────────────────────────

def causal_mask(T, device):
    """
    因果掩码（bool）：True = 不许看。

    用 bool 而非 float -inf：一是和 padding mask 类型一致（避免 PyTorch 告警），
    二是绕开 MPS 上 0 × (-inf) = nan 的坑。
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=256,
                 dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
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
        self.head.weight = self.tok.weight      # weight tying

        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx, pad_mask=None):
        x = self.tok(idx)
        if self.use_pos:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        mask = causal_mask(idx.size(1), idx.device) if self.causal else None
        x = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)
        return self.head(x)


def pick_device(choice="auto"):
    if choice and choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ─────────────────────────── 阶段一：预训练 ───────────────────────────

def run_pretrain_epoch(model, loader, optimizer, device, train=True, log_every=0):
    model.train(train)
    total, toks = 0.0, 0
    t0 = time.time()
    n = len(loader)
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss) * y.numel()
        toks += y.numel()
        if train and log_every and (i % log_every == 0 or i == n):
            avg = total / max(1, toks)
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"    batch {i:>5}/{n}  loss={avg:.4f}  ppl={math.exp(avg):>8.1f}  "
                  f"已用 {elapsed:>4.0f}s  剩余≈{eta/60:.1f}min", flush=True)
    if toks == 0:
        return float("inf"), float("inf")
    avg = total / toks
    return avg, math.exp(avg)


def train_pretrain(args, device, char2idx, idx2char):
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"没读到预训练语料：{args.corpus}")
    if args.max_chars:
        text = text[:args.max_chars]
    print(f"预训练语料 {len(text):,} 字")

    lines = text.splitlines()
    random.seed(args.seed)
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    tr_text = "\n".join(lines[:split])
    va_text = "\n".join(lines[split:])

    tr = CharDataset(tr_text, char2idx, args.seq_len)
    va = CharDataset(va_text, char2idx, args.seq_len)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"窗口数：训练 {len(tr):,} / 验证 {len(va):,}（seq_len={args.seq_len}，stride={args.seq_len}）")

    model = TransformerLM(len(char2idx), args.d_model, args.heads, args.layers,
                          args.seq_len, args.dropout).to(device)
    if args.init:
        model = _maybe_load(model, args.init)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_par:,}（d={args.d_model} 层={args.layers} 头={args.heads}）")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainPPL':>9}  {'ValLoss':>9}  {'ValPPL':>8}")
    print("-" * 58)
    for ep in range(1, args.epochs + 1):
        st = time.time()
        trl, trp = run_pretrain_epoch(model, tr_loader, opt, device, True, args.log_every)
        with torch.no_grad():
            val, vap = run_pretrain_epoch(model, va_loader, opt, device, False)
        mark = ""
        if vap < best:
            best = vap
            mark = "  *"
            torch.save({"stage": "pretrain", "model_state": model.state_dict(),
                        "char2idx": char2idx, "idx2char": idx2char,
                        "specials": SPECIALS, "args": vars(args)}, args.save)
        print(f"{ep:>6}  {trl:>10.4f}  {trp:>9.1f}  {val:>9.4f}  {vap:>8.1f}{mark}  "
              f"({time.time()-st:.0f}s)")
    print(f"\n预训练完成。最佳验证 PPL {best:.2f}，已保存 {args.save}")


# ─────────────────────────── 阶段二：SFT ───────────────────────────

def run_sft_epoch(model, loader, optimizer, device, train=True, log_every=0):
    """返回 (回答段 loss, ppl, token 准确率)。只统计 labels != -100 的位置。"""
    model.train(train)
    total, toks, hit = 0.0, 0, 0
    t0 = time.time()
    n = len(loader)
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        pad_mask = (x == PAD)
        logits = model(x, pad_mask=pad_mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        k = int((y != -100).sum())
        total += float(loss) * k
        toks += k
        with torch.no_grad():
            pred = logits.argmax(-1)
            hit += int(((pred == y) & (y != -100)).sum())
        if train and log_every and (i % log_every == 0 or i == n):
            avg = total / max(1, toks)
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"    batch {i:>4}/{n}  loss={avg:.4f}  ppl={math.exp(avg):>7.1f}  "
                  f"acc={hit/max(1,toks):.3f}  已用 {elapsed:>4.0f}s  剩余≈{eta/60:.1f}min",
                  flush=True)
    if toks == 0:
        return float("inf"), float("inf"), 0.0
    avg = total / toks
    return avg, math.exp(avg), hit / toks


def train_sft(args, device, char2idx, idx2char, from_pretrain=None):
    train_rows = load_jsonl(os.path.join(args.data_dir, args.train_file))
    val_rows = load_jsonl(os.path.join(args.data_dir, args.val_file))
    if args.limit:
        train_rows = train_rows[:args.limit]
    print(f"SFT 样本：训练 {len(train_rows)} / 验证 {len(val_rows)}   字表 {len(char2idx)}")

    tr = SFTDataset(train_rows, char2idx, args.seq_len)
    va = SFTDataset(val_rows, char2idx, args.seq_len)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_sft, drop_last=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_sft, drop_last=False)

    ids, labels = tr[0]
    n_mask = sum(1 for v in labels if v == -100)
    print(f"样例长度 {len(ids)}：prompt 段屏蔽 {n_mask} 个，回答段参与 loss {len(ids)-n_mask} 个")

    model = TransformerLM(len(char2idx), args.d_model, args.heads, args.layers,
                          args.seq_len, args.dropout).to(device)
    if from_pretrain:
        model = _maybe_load(model, from_pretrain)
        print(f"已加载预训练权重：{os.path.basename(from_pretrain)}")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_par:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainPPL':>9}  {'TrainAcc':>9}  "
          f"{'ValLoss':>9}  {'ValPPL':>8}  {'ValAcc':>7}")
    print("-" * 76)
    for ep in range(1, args.epochs + 1):
        st = time.time()
        trl, trp, tra = run_sft_epoch(model, tr_loader, opt, device, True, args.log_every)
        with torch.no_grad():
            val, vap, vac = run_sft_epoch(model, va_loader, opt, device, False)
        mark = ""
        if vap < best:
            best = vap
            mark = "  *"
            torch.save({"stage": "sft", "model_state": model.state_dict(),
                        "char2idx": char2idx, "idx2char": idx2char,
                        "specials": SPECIALS, "args": vars(args)}, args.save)
        print(f"{ep:>6}  {trl:>10.4f}  {trp:>9.1f}  {tra:>9.3f}  "
              f"{val:>9.4f}  {vap:>8.1f}  {vac:>7.3f}{mark}  ({time.time()-st:.0f}s)")

    print(f"\nSFT 完成。最佳验证 PPL {best:.2f}，已保存 {args.save}")
    first_token_report(model, val_rows, char2idx, idx2char, device)


# ─────────────────────────── 诊断 & 生成 ───────────────────────────

@torch.no_grad()
def first_token_report(model, rows, char2idx, idx2char, device, n=200):
    """
    「第一个回答字」的准确率 —— 生成能不能起步，全看这个位置。

    普通的 token 准确率是在「已知答案前缀」的前提下算的，会严重高估生成质量；
    这里单独看 prompt 最后一个位置的预测，才是 SFT 是否真的学会「听指令」的指标。
    """
    rows = rows[:n]
    hit = 0
    for r in rows:
        ids, labels = build_example(r, char2idx)
        p = next(i for i, v in enumerate(labels) if v != -100)
        lg = model(torch.tensor([ids[:p]], device=device))[0, -1]
        hit += int(int(lg.argmax()) == labels[p])
    print(f"\n第一个回答字准确率：{hit}/{len(rows)} = {hit/max(1,len(rows)):.1%}"
          f"（随机基线 {1/len(char2idx):.2%}）")
    print("  这个数低，说明模型还没学会「听指令」，生成会一开口就跑偏。")
    return hit / max(1, len(rows))


@torch.no_grad()
def generate(model, char2idx, idx2char, instruction, max_new=96,
             temperature=0.0, top_k=0, device=None, rep_penalty=1.3):
    """按 chat 模板生成回答，遇到 <|end|> 停止。"""
    device = device or next(model.parameters()).device
    prompt = [USER] + encode(instruction, char2idx) + [ASSISTANT]
    out = list(prompt)
    seq_len = model.pos.num_embeddings if model.pos is not None else 512

    for _ in range(max_new):
        logits = model(torch.tensor([out[-seq_len:]], device=device))[0, -1]
        logits = logits.float().cpu()
        logits[PAD] = float("-inf")
        if rep_penalty and rep_penalty > 1.0:
            for t in set(out[len(prompt):]):
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 \
                    else logits[t] * rep_penalty
        if temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        if nxt == END:
            break
        out.append(nxt)
    return decode(out[len(prompt):], idx2char)


# 兼容旧的演示层：它 import 的是 collate / run_epoch 这两个名字
collate = collate_sft
run_epoch = run_sft_epoch


def _maybe_load(model, path):
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(path):
        print(f"[提示] 没找到 {os.path.basename(path)}，本次从随机初始化开始")
        return model
    ck = torch_load(path, "cpu")
    try:
        model.load_state_dict(ck["model_state"])
        print(f"已加载权重：{os.path.basename(path)}")
    except RuntimeError as e:
        print(f"[跳过] 权重形状不匹配（字表或结构不同）：{str(e)[:80]}")
    return model


def load_chat_model(args, device):
    # generate 分支在 --save 被赋值之前就要用，所以要在这里兜底默认文件名
    path = args.ckpt or args.save or "sft_best.pt"
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(HERE, path)
    ck = torch_load(path, "cpu")
    a = ck["args"]
    model = TransformerLM(len(ck["char2idx"]), a["d_model"], a["heads"], a["layers"],
                          a["seq_len"], 0.0).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sft", choices=["pretrain", "sft"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="默认：预训练 3 轮 / SFT 8 轮")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "sft_train"))
    ap.add_argument("--train-file", default="sft_train.jsonl")
    ap.add_argument("--val-file", default="sft_val.jsonl")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus_general.txt"))
    ap.add_argument("--max_chars", type=int, default=0, help="预训练时只用前 N 个字符（试跑）")
    ap.add_argument("--limit", type=int, default=0, help="SFT 只用前 N 条（试跑）")
    ap.add_argument("--seq_len", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--max_vocab", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", default=None,
                    help="初始化权重：SFT 默认用 pretrain_best.pt；传空串则从零开始")
    ap.add_argument("--save", default=None,
                    help="默认：预训练 pretrain_best.pt / SFT sft_best.pt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--num_threads", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    # 推理
    ap.add_argument("--generate", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.3)
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---------- 推理 ----------
    if args.generate:
        device = pick_device(args.device)
        model, ck = load_chat_model(args, device)
        ans = generate(model, ck["char2idx"], ck["idx2char"], args.generate,
                       args.max_new, args.temperature, args.top_k, device,
                       args.rep_penalty)
        print(f"\n指令：{args.generate}\n回答：{ans or '（空）'}")
        return

    device = pick_device(args.device)
    stage = args.stage
    epochs = args.epochs if args.epochs is not None else (3 if stage == "pretrain" else 8)
    save = args.save or os.path.join(
        HERE, "pretrain_best.pt" if stage == "pretrain" else "sft_best.pt")

    # ---------- 字表：两阶段共用 ----------
    init_path = args.init
    if stage == "sft" and init_path is None:
        # 默认尝试加载同目录的预训练权重
        init_path = os.path.join(HERE, "pretrain_best.pt")
        if not os.path.exists(init_path):
            init_path = ""

    char2idx = idx2char = None
    if stage == "sft" and init_path:
        p = init_path if os.path.isabs(init_path) else os.path.join(HERE, init_path)
        if os.path.exists(p):
            ck = torch_load(p, "cpu")
            char2idx, idx2char = ck["char2idx"], ck["idx2char"]
            print(f"字表来自 {os.path.basename(p)}（{len(char2idx)} 个）")
    if char2idx is None:
        rows = load_jsonl(os.path.join(args.data_dir, args.train_file))
        corpus_text = ""
        if stage == "pretrain":
            corpus_text = load_corpus(args.corpus)[: args.max_chars or None]
        char2idx, idx2char = build_vocab(rows, corpus_text, args.max_vocab)
        print(f"字表已构建（{len(char2idx)} 个字符，含语料）")

    args.epochs, args.save = epochs, save
    print(f"device {device}  stage={stage}  epochs={epochs}  save={os.path.basename(save)}")

    if stage == "pretrain":
        train_pretrain(args, device, char2idx, idx2char)
    else:
        train_sft(args, device, char2idx, idx2char, from_pretrain=init_path)


if __name__ == "__main__":
    main()

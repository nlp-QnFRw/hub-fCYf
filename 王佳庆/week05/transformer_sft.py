# -*- coding: utf-8 -*-
"""
transformer_sft.py — 把 transformer_lm.py（字符级自回归 LM 预训练）改造成 SFT 指令微调

【和原版 transformer_lm.py 的区别，也是 SFT 的全部要点】
    原版是无监督语言模型：读一整篇语料 -> 定长滑窗 -> "每个位置的下一个字"都算 loss。
    本版是监督微调（Supervised Fine-Tuning）：
      1) 数据换成 jsonl 指令对 {"instruction": ..., "output": ...}，用统一模板拼成一条样本：
             ### 指令:
             {instruction}

             ### 回答:
             {output}<EOS>
      2) 只对「回答」部分算 loss：prompt 部分和 padding 位置的 label 置 -100（ignore_index），
         模型不会被要求去学习"复述用户指令"，优化目标只对准"按指令给出回答"。
      3) 样本长短不一 -> collate 动态 padding，并给注意力传 key_padding_mask，
         让 padding 位置既不参与 attention 计算，也不参与 loss。
      4) 生成时按同一个模板输入指令，逐字解码直到 <EOS> 停止（预训练版是无限续写 n 个字）。
      5) 评测口径改成「回答 token 上的 loss / PPL / 首字准确率」，用它选最优 ckpt。
    模型结构（Pre-LN Transformer + 可学习位置编码 + 因果掩码 + 输入输出 embedding 绑定）
    以及 --no-mask / --no-pos 两个消融开关都保持原样，方便和预训练版对照。

【数据】指令对仍来自同目录 sft_train_data/sft_train.jsonl（3000 条）和 sft_val.jsonl（200 条），
    但字表不再从 jsonl 现造：改用与预训练 transformer_lm.py 同一份 corpus_general.txt、
    同一套 build_vocab 口径（默认 max_vocab=4000、按字频降序）构建，
    前 base 个 id 与预训练权重逐行对齐；SFT 必需的 <PAD>/<EOS>/<UNK>、模板字符，
    以及语料里没出现过的字符（实测连空格和 '#' 都没有）只追加在字表末尾，不动前面的 id。
    这样编码口径和字符分布都与预训练一致，也不会让 "### 指令:" 被编成一片 <UNK>。
    指令对只决定「学什么」，语料文件和字表口径决定「怎么编码」，两者互不干扰。

【用法】
    # 训练（默认 GPU 可用则用 GPU）
    python transformer_sft.py --epochs 8
    # 快速跑通流程（只取 64 条、1 个 epoch）
    python transformer_sft.py --max_samples 64 --epochs 1
    # 换一份/多份预训练语料构建字表（口径仍与 transformer_lm.py 一致）
    python transformer_sft.py --corpus "corpus_*.txt" --max_vocab 4000
    # 消融实验：去掉因果掩码 / 位置编码
    python transformer_sft.py --epochs 8 --no-mask
    python transformer_sft.py --epochs 8 --no-pos
    # 用训练好的 ckpt 按指令生成回答
    python transformer_sft.py --generate "给出两个数字，求它们的平均值。 25, 41"
    python transformer_sft.py --generate "解释一个术语或缩略语。 请解释以下缩写的含义：NLP。" --temperature 0.8
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
IGNORE = -100                       # cross_entropy 的 ignore_index：不计 loss 的 label

# 字符级模型没有 tokenizer，用 Unicode 私有区的 3 个单字符当特殊符号，
# 保证不会和数据里的真实文字撞车，也仍然能「一个字符 = 一个 id」地编解码。
PAD, EOS, UNK = "\ue000", "\ue001", "\ue002"
SPECIALS = [PAD, EOS, UNK]
PROMPT_TMPL = "### 指令:\n{instruction}\n\n### 回答:\n"

try:
    # 统一按 UTF-8 输出，避免 Windows 控制台/管道默认 GBK 导致中文乱码或生僻字报错
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# ─────────────────────── 数据（字表沿用预训练语料） ───────────────────────

def load_corpus(pattern):
    """
    读预训练语料，口径与 transformer_lm.py 的 load_corpus 完全一致：
    glob 展开（支持 "corpus_*.txt"）、按文件名排序后拼接、utf-8 + errors="ignore"。
    读法一致，后面统计出的字频与字表顺序才能和预训练对得上。
    """
    text = ""
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    if not text:
        raise FileNotFoundError(f"没读到预训练语料: {pattern}")
    return text


def build_prompt(instruction):
    """训练和推理必须用同一个模板，否则模型认不出 '该回答了' 的边界。"""
    return PROMPT_TMPL.format(instruction=instruction.strip())


def load_sft_jsonl(path):
    """逐行读 jsonl，校验字段；空行直接跳过，字段缺失给明确报错而不是 KeyError。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "instruction" not in obj or "output" not in obj:
                raise ValueError(f"{path}:{lineno} 缺少 instruction/output 字段: {line[:80]}")
            instruction = str(obj["instruction"])
            extra_input = str(obj.get("input", "")).strip()
            if extra_input:          # 兼容带 input 字段的指令数据，统一并入 instruction
                instruction += "\n" + extra_input
            rows.append((instruction, str(obj["output"])))
    if not rows:
        raise ValueError(f"SFT 数据为空: {path}")
    return rows


def build_vocab(text, max_vocab=4000, must_keep=""):
    """
    字表分两段构建，目的是和预训练 transformer_lm.py 保持同一套编码口径：

      1) 基础段（id 0..base-1）：逐字复刻预训练的 build_vocab ——
         max_vocab > 0 时取 Counter(text).most_common(max_vocab)（按字频降序），
         max_vocab = 0 时取 sorted(set(text))（按 Unicode 码位）。
         读的是同一份 corpus_general.txt、用的是同一个公式，
         这 base 个 id 与预训练 ckpt 的 embedding 行严格一一对应，
         日后加载预训练权重时不会出现「id 相同、字不同」的错位。

      2) 追加段（id base..）：SFT 必须能编码、但预训练语料里没有的字符 ——
         <PAD>/<EOS>/<UNK>、prompt 模板、指令/回答文本里的字符。
         实测 corpus_general.txt 连空格和 '#' 都没出现过，只用预训练字表的话
         "### 指令:" 会被编成一片 <UNK>，模型根本看不到「该回答了」的边界；
         SFT 文本里另有大量 ASCII 符号/生僻字也会丢失。
         这些字符按 sorted 追加在末尾，顺序可复现，且绝不改动基础段的 id。

    must_keep 传「模板 + 全部 SFT 文本」，都是无监督的字符集合，不含标签信息，
    不构成标签泄漏。返回 char2idx / idx2char / 基础段大小（便于核对预训练字表规模）。
    """
    ordered = [c for c, _ in Counter(text).most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(ordered)}
    base = len(char2idx)
    clash = [c for c in SPECIALS if c in char2idx]
    if clash:
        raise ValueError(f"预训练语料里出现了 SFT 专用特殊符号 {clash!r}，请改 SPECIALS")
    for c in sorted(set(SPECIALS) | set(must_keep)):
        if c not in char2idx:
            char2idx[c] = len(char2idx)
    return char2idx, {i: c for c, i in char2idx.items()}, base


def encode(text, char2idx):
    """字表外的字映射到 <UNK>（原版是直接丢弃，指令微调里会破坏用户输入，故改成 UNK）。"""
    unk = char2idx[UNK]
    return [char2idx.get(c, unk) for c in text]


class SFTDataset(Dataset):
    """
    一条样本 = prompt + 回答 + <EOS>，再错一位构成自回归 (x, y)：
        ids    = [p0 p1 ... pP-1 a0 a1 ... aN-1 EOS]
        x      = ids[:-1]
        y      = ids[1:]
        labels = [-100 ... -100  a0 a1 ... aN-1 EOS]
                   └─ P-1 个 ─┘  └──── 只在这里算 loss ────┘
    注意 y[P-1] == ids[P] == a0（第一个回答字），y 的最后一项 == EOS，
    所以「回答 + 学会停下来」都被监督到，而指令部分一个 token 都不进 loss。
    """

    def __init__(self, path, char2idx, max_len, name="data"):
        self.samples = []
        truncated = 0
        # 超长指令只截「指令正文」，必须保留模板头尾，否则模型看不到 "### 回答:" 这个切换信号。
        max_instr = max(1, max_len // 2 - len(build_prompt("")))
        for instruction, output in load_sft_jsonl(path):
            if len(instruction) > max_instr:
                truncated += 1
            p_ids = encode(build_prompt(instruction[:max_instr]), char2idx)
            a_ids = encode(output.strip(), char2idx)
            eos = char2idx[EOS]
            # 回答尾部截断后仍强制补 EOS，让模型始终学得到「何时停」。
            room = max(1, max_len - len(p_ids) - 1)
            if len(a_ids) >= room:
                a_ids = a_ids[:room - 1]
                truncated += 1
            ids = p_ids + a_ids + [eos]
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            labels = torch.full_like(y, IGNORE)
            labels[len(p_ids) - 1:] = y[len(p_ids) - 1:]
            assert x.numel() == labels.numel() and (labels != IGNORE).sum() == len(a_ids) + 1
            self.samples.append((x, labels))
        if truncated:
            print(f"[{name}] 共 {truncated} 处截断（超过 max_len={max_len}，指令/回答各计一次）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_collate(pad_id):
    """变长样本补齐到 batch 内最长；pad 位置 label=-100，并产出 attention 的 key_padding_mask。"""
    def collate(batch):
        T = max(x.numel() for x, _ in batch)
        input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), T), IGNORE, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            input_ids[i, :x.numel()] = x
            labels[i, :y.numel()] = y
        return input_ids, labels, input_ids.eq(pad_id) # key_padding_mask，padding部分不参与attention计算
    return collate


# ─────────────────────── 模型（结构沿用原版，只加 padding mask） ───────────────────────

class TransformerLM(nn.Module):
    """
    Pre-LN Transformer + 可学习位置编码 + weight tying，和预训练版完全同构。
    唯一改动：forward 支持 src_key_padding_mask，并改用 bool 因果掩码
    （bool mask 与 bool padding mask 组合时 PyTorch 会正确合并，不会退化/报警告）。
    """

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=192,
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
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx, key_padding_mask=None):
        T = idx.size(1)
        x = self.tok(idx)
        if self.use_pos:
            # 训练时所有样本左对齐、位置都从 0 开始；生成时也这样喂，两者才对齐。
            x = x + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)
        mask = None
        if self.causal:
            # True = 屏蔽。上三角（不含对角线）屏蔽未来，等价于 GPT 的因果注意力。
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        x = self.encoder(x, mask=mask, src_key_padding_mask=key_padding_mask)
        return self.head(x)


# ─────────────────────── 训练 / 评估 ───────────────────────

def answer_loss(logits, y, vocab_size):
    """回答 token 上的平均交叉熵；reduce=sum 再除以有效 token 数，保证 batch 间口径一致。"""
    loss_sum = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                               ignore_index=IGNORE, reduction="sum")
    n_valid = (y != IGNORE).sum().clamp(min=1)
    return loss_sum / n_valid, n_valid


@torch.no_grad()
def evaluate(model, loader, device, vocab_size):
    """验证集：只在回答 token 上算 loss / PPL / argmax 准确率。"""
    model.eval()
    total, tokens, correct = 0.0, 0, 0
    for x, y, kpm in loader:
        x, y, kpm = x.to(device), y.to(device), kpm.to(device)
        logits = model(x, kpm)
        total += float(F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                       ignore_index=IGNORE, reduction="sum"))
        valid = y != IGNORE
        tokens += int(valid.sum())
        correct += int(((logits.argmax(-1) == y) & valid).sum())
    loss = total / max(tokens, 1)
    return loss, math.exp(min(loss, 20)), correct / max(tokens, 1)


def train(args):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. 数据 + 字表：字表来自预训练语料，口径与 transformer_lm.py 一致
    corpus = load_corpus(args.corpus)
    tr_rows = load_sft_jsonl(args.train_file)
    va_rows = load_sft_jsonl(args.val_file)
    template_chars = build_prompt("")
    # 这些字符都必须能编码：模板 + 训练/验证里全部指令与回答（无监督字符集合）
    sft_text = template_chars + "".join(i + o for i, o in tr_rows + va_rows)
    char2idx, idx2char, base_vocab = build_vocab(corpus, args.max_vocab, must_keep=sft_text)
    V = len(char2idx)
    oov = sum(1 for c in sft_text if c not in char2idx)
    assert oov == 0, "SFT 文本里仍有字符无法编码，模板/数据会被 <UNK> 破坏"
    print(f"device: {device}  预训练语料 {len(corpus):,} 字 / 基础字表 {base_vocab}  "
          f"+ SFT 追加 {V - base_vocab} → 合计 {V}  SFT 未覆盖字符 {oov}")
    print(f"训练 {len(tr_rows)} 条 / 验证 {len(va_rows)} 条  max_len {args.max_len}  "
          f"字表口径 max_vocab={args.max_vocab}（与预训练一致）")

    tr = SFTDataset(args.train_file, char2idx, args.max_len, "train")
    va = SFTDataset(args.val_file, char2idx, args.max_len, "val")
    if args.max_samples:
        tr = Subset(tr, range(min(args.max_samples, len(tr))))
        va = Subset(va, range(min(args.max_samples, len(va))))
    collate = make_collate(char2idx[PAD])
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate)
    # 验证集不 drop_last，200 条要全部参与评测
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collate)

    # 2. 自检：label mask 是否只放行回答、causal + padding 双 mask 是否会产生 NaN
    x0, y0 = tr[0]
    target = "".join(idx2char[i] for i in y0.tolist() if i != IGNORE)
    print(f"SFT 样例 输入模板: {build_prompt(tr_rows[0][0]).replace(chr(10), ' / ')}[回答...]")
    print(f"SFT 样例 监督目标: {target.replace(EOS, '<EOS>')}")
    assert target.endswith(EOS) and len(target) > 1, "回答区 label mask 未正确对齐"

    model = TransformerLM(V, args.d_model, args.heads, args.layers, args.max_len,
                          args.dropout, use_pos=not args.no_pos, causal=not args.no_mask).to(device)
    x, y, kpm = next(iter(train_loader))
    with torch.no_grad():
        out = model(x[:2].to(device), kpm[:2].to(device))
    assert torch.isfinite(out).all(), "因果掩码 + padding mask 组合产生 NaN"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: Transformer x{args.layers} (d={args.d_model} h={args.heads} Pre-LN)  "
          f"参数量 {n_par/1e6:.2f}M  因果掩码={'有' if not args.no_mask else '**无(泄漏实验)**'}  "
          f"位置编码={'有' if not args.no_pos else '无'}")
    print(f"batch {tuple(x.shape)}  监督 token {(y != IGNORE).sum().item()}/{y.numel()}  "
          f"padding {kpm.sum().item()}  mask 自检通过")

    # 3. 训练
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} {'ValPPL':>9} {'ValAcc':>8} {'tok/s':>8}")
    print("-" * 60)
    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y, kpm in train_loader:
            x, y, kpm = x.to(device), y.to(device), kpm.to(device)
            loss, n_valid = answer_loss(model(x, kpm), y, V)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * int(n_valid)
            seen += int(n_valid)
        dt = time.time() - st
        v_loss, v_ppl, v_acc = evaluate(model, val_loader, device, V)
        mark = ""
        if v_loss < best:
            best = v_loss
            mark = " *"
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args),
                        "prompt_template": PROMPT_TMPL,
                        "corpus": args.corpus, "base_vocab_size": base_vocab}, args.save)
        print(f"{ep:>6} {tot/max(seen,1):>10.4f} {v_loss:>9.4f} {v_ppl:>9.2f} "
              f"{v_acc:>7.1%} {seen/max(dt,1e-6):>8,.0f}{mark}")
        
    print(f"\n最佳验证 loss: {best:.4f}  已保存 {args.save}")


def ban_repeated_ngrams(logits, out, n):
    """解码侧小技巧：屏蔽会与历史构成重复 n-gram 的字，缓解贪心解码的「议议议议」复读。"""
    if n < 2 or len(out) < n - 1:
        return
    prefix = tuple(out[-(n - 1):])
    banned = [out[i + n - 1] for i in range(len(out) - n + 1)
              if tuple(out[i:i + n - 1]) == prefix]
    if banned:
        logits[torch.as_tensor(banned, device=logits.device)] = -1e9


@torch.no_grad()
def generate(args):
    """按训练时的模板输入一条指令，逐字生成回答，遇到 <EOS> 停止。"""
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"],
                          c["max_len"], 0.0,
                          use_pos=not c.get("no_pos", False),
                          causal=not c.get("no_mask", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    prompt = ck.get("prompt_template", PROMPT_TMPL).format(instruction=args.generate.strip())
    ids = encode(prompt, char2idx)
    eos = char2idx[EOS]
    out = []
    for _ in range(args.n_chars):
        # 位置编码最多 max_len 行；超长时只保留最近的上下文（训练时位置也是从 0 起算）。
        ctx = (ids + out)[-c["max_len"]:]
        logits = model(torch.tensor([ctx], device=device))[0, -1]
        # 不让模型输出 PAD / UNK 这类非文本符号
        logits[char2idx[PAD]] = -1e9
        logits[char2idx[UNK]] = -1e9
        ban_repeated_ngrams(logits, out, args.no_repeat_ngram)
        if args.temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
        if nxt == eos:
            break
        out.append(nxt)
    print(f"指令: {args.generate}")
    print(f"回答: {''.join(idx2char[i] for i in out)}")


def main():
    ap = argparse.ArgumentParser(description="字符级 Transformer 的 SFT 指令微调")
    data = os.path.join(HERE, "sft_train_data")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus_general.txt"),
                    help="预训练语料路径（与 transformer_lm.py 同一份，支持通配符），用于构建字表")
    ap.add_argument("--train_file", default=os.path.join(data, "sft_train.jsonl"))
    ap.add_argument("--val_file", default=os.path.join(data, "sft_val.jsonl"))
    ap.add_argument("--max_len", type=int, default=192, help="单条样本最大长度（也是位置编码长度）")
    ap.add_argument("--max_vocab", type=int, default=6000,
                    help="预训练字表上限，0=保留语料全部字符；默认 4000 与 transformer_lm.py 一致")
    ap.add_argument("--max_samples", type=int, default=0, help="只取前 N 条，用于快速跑通")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12, help="CPU 上建议配合 --max_samples 先调小")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # 保留原版的两个消融开关：SFT 下 --no-mask 同样会出现「看到答案 -> loss 假性趋零」
    ap.add_argument("--no-mask", dest="no_mask", action="store_true", help="去掉因果掩码（泄漏实验）")
    ap.add_argument("--no-pos", dest="no_pos", action="store_true", help="去掉位置编码")
    ap.add_argument("--save", default=os.path.join(HERE, "sft_best.pt"))
    ap.add_argument("--generate", help="给一条指令，用 ckpt 生成回答（不训练）")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "sft_best.pt"))
    ap.add_argument("--n_chars", type=int, default=80, help="最多生成多少个字")
    ap.add_argument("--temperature", type=float, default=0.0, help="0=贪心解码，>0 为采样温度")
    ap.add_argument("--no_repeat_ngram", type=int, default=3, help="禁止重复 n-gram（0=关闭）")
    args = ap.parse_args()

    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

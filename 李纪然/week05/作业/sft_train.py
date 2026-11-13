#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第五周作业：基于语言模型的 SFT 训练
=============================================================

在 `sft_train.jsonl` / `sft_val.jsonl`（3000 / 200 条 指令-回答 对）上，
从零手写一个 decoder-only 的字符级 Transformer，走完课件第五部分的完整链路：

    ① 预训练（next-token prediction，100% 位置有监督）
        —— 得到一台「概率续写器」：给它一个问题，它接着列更多问题。
    ② SFT（同一个结构 / 同一个 next-token 目标，只改数据 + Loss Mask）
        —— 用 <|user|> / <|assistant|> / <|end|> 的 chat template，
           把 prompt 段 labels 打成 -100，只学「该怎么答」，
           并学会在答完后生成 <|end|> 停下。

核心口径（与课件第 28 页一致）：
    input_ids = [<|user|> 问题 <|assistant|> 回答 <|end|>]
    labels    = [-100 ... -100,             回答 <|end|>]
    loss      = F.cross_entropy(logits, labels, ignore_index=-100)
即：SFT 的 loss 只落在「该说的那半段」上。

用法
----
    # 两阶段全流程（预训练 + 全参数 SFT），并出曲线图
    python3 sft_train.py --stage both --tag sft

    # 只跑 SFT（复用 artifacts/ 里已有的预训练权重）
    python3 sft_train.py --stage sft --tag sft --peft none

    # 参数高效对照：LoRA（只训 0.3%~2% 参数）
    python3 sft_train.py --stage sft --tag lora --peft lora --lora_r 8

    # 不训练，只用已有 json 重画曲线
    python3 sft_train.py --plot_only --tag sft

产物
----
    提交物：sft_train.py / sft_train.jsonl / sft_val.jsonl / sft-curves.png / 实验报告.md
    过程物（不用交）：artifacts/ 下的权重、日志、每轮指标 json、生成样例
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 0. 全局：特殊 token / 设备
# --------------------------------------------------------------------------
PAD, UNK = "<pad>", "<unk>"
U_TAG, A_TAG, E_TAG = "<|user|>", "<|assistant|>", "<|end|>"
SPECIALS = [PAD, UNK, U_TAG, A_TAG, E_TAG]
PAD_ID, UNK_ID, U_ID, A_ID, E_ID = range(5)
IGNORE = -100  # PyTorch 约定：labels 里这些位置跳过，不参与 loss / 梯度


def pick_device(want="auto"):
    if want != "auto":
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 1. 数据：读 jsonl → 字表 → 拼 chat template / 拼预训练原始文本流
# --------------------------------------------------------------------------
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_vocab(rows, max_vocab=4000, corpus_text=""):
    """按词频取前 max_vocab 个字符（含特殊 token 占位），其余映射成 <unk>。
    corpus_text 里的字也计入词频 —— 预训练语料更广，字表要覆盖到。"""
    freq = {}
    for r in rows:
        for ch in r["instruction"] + r["output"]:
            freq[ch] = freq.get(ch, 0) + 1
    for ch in corpus_text:
        freq[ch] = freq.get(ch, 0) + 1
    chars = [c for c, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    chars = chars[: max(1, max_vocab - len(SPECIALS))]
    itos = SPECIALS + chars
    return {s: i for i, s in enumerate(itos)}, itos


_SPECIAL_RE = None


def _special_pattern():
    """把特殊 token 当成「整体」匹配，避免 <|user|> 被按字符拆成 < | u s e r | >。"""
    global _SPECIAL_RE
    if _SPECIAL_RE is None:
        import re

        _SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")
    return _SPECIAL_RE


def encode(text, stoi):
    """字符级编码，但保留特殊 token 的完整性（chat template 靠它生效）。"""
    ids = []
    for part in _special_pattern().split(text):
        if not part:
            continue
        if part in (PAD, UNK, U_TAG, A_TAG, E_TAG):
            ids.append(stoi[part])
            continue
        ids.extend(stoi.get(ch, UNK_ID) for ch in part)
    return ids


def make_chat_example(row, stoi, mask="answer"):
    """一条 SFT 样本：prompt 段打 -100，只留回答段（含 <|end|>）的 loss。
    mask="full" 时不做 mask（对照组：让它连问题一起背），用于验证 Loss Mask 的必要性。"""
    prompt_text = f"{U_TAG}{row['instruction']}{A_TAG}"
    answer_text = f"{row['output']}{E_TAG}"
    prompt = encode(prompt_text, stoi)
    answer = encode(answer_text, stoi)
    ids = prompt + answer
    if mask == "full":
        return ids, list(ids)
    return ids, [IGNORE] * len(prompt) + answer


def make_pretrain_stream(rows, stoi, block_size, corpus_text=""):
    """预训练语料：一条没有角色标记的原始文本流，按 block_size 切开
    （标签就是输入本身，错一位在 loss 里做）。
    有 --pretrain_corpus 时用外部语料，否则把「问题 + 回答」拼起来用。"""
    if corpus_text:
        stream = corpus_text
    else:
        stream = "".join(r["instruction"] + " " + r["output"] for r in rows)
    ids = encode(stream, stoi)
    blocks = []
    for i in range(0, len(ids) - block_size, block_size):
        seg = ids[i : i + block_size]
        blocks.append((seg, list(seg)))
    return blocks


def collate(batch, device):
    """动态 padding：input 补 <pad>，labels 补 -100（不参与 loss）。"""
    max_len = max(len(x) for x, _ in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        attn[i, :n] = 1
    return (
        input_ids.to(device),
        labels.to(device),
        attn.to(device),
    )


def make_batches(examples, batch_size, rng, shuffle=True):
    """按长度分桶再打乱，减少 padding 浪费（小数据集上比随机截断更快）。"""
    items = list(examples)
    order = list(range(len(items)))
    rng.shuffle(order)
    items = [items[i] for i in order]
    bucket = batch_size * 50
    batches = []
    for s in range(0, len(items), bucket):
        chunk = sorted(items[s : s + bucket], key=lambda x: len(x[0]))
        for i in range(0, len(chunk), batch_size):
            batches.append(chunk[i : i + batch_size])
    if shuffle:
        rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------
# 2. 模型：手写 decoder-only Transformer（Pre-LN + 因果自注意力）
# --------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # 下三角「谁见谁」表：1 = 可见，0 = 屏蔽（课件第 10 页那张矩阵）
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x, attn=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal[:, :, :T, :T], float("-inf"))
        if attn is not None:  # 再把 padding 位置屏蔽掉
            scores = scores.masked_fill(attn[:, None, None, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.drop(w)
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn=None):
        x = x + self.drop(self.attn(self.ln1(x), attn))
        h = self.ln2(x)
        h = self.drop(self.proj(F.gelu(self.fc(h))))
        return x + h


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 192
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 8
    dropout: float = 0.1


class GPT(nn.Module):
    """最小可用的 decoder-only LM：层数 / 维度 / 注意力头都不变，
    从预训练到 SFT 只换数据与 loss mask（课件第 27、29 页）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_head, cfg.dropout, cfg.block_size) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 权重共享，省参数
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, attn=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过 block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x, attn)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.0, top_k=None,
                 stop_ids=(E_ID,), forbidden_ids=(PAD_ID, U_ID, A_ID)):
        """贪心（temperature=0）或采样解码；遇到 <|end|> 或长度上限就停。
        「什么时候停」是 SFT 教出来的行为 —— base 模型没有这个能力。"""
        self.eval()
        out = idx
        for _ in range(max_new_tokens):
            ctx = out[:, -self.cfg.block_size :]
            logits = self(ctx)[:, -1, :].float()
            logits[:, list(forbidden_ids)] = float("-inf")
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            if nxt.item() in stop_ids:
                break
        return out


# --------------------------------------------------------------------------
# 3. LoRA：只训 0.3%~2% 参数（课件第 31 页「参数高效」的做法）
# --------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.r, self.scaling = r, alpha / r
        self.drop = nn.Dropout(dropout)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B 保持 0，保证初始等价

    def forward(self, x):
        out = self.base(x)
        return out + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def apply_lora(model, r=8, alpha=16, dropout=0.05):
    """只把注意力里的 qkv / 输出投影换成 LoRA 版，其余参数全部冻结。
    4 层 × (qkv + proj) 的可训练参数约 5 万，占 4.2M 的 1.2%（课件第 31 页「0.3%~1%」量级）。"""
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha, dropout)
        blk.attn.proj = LoRALinear(blk.attn.proj, r, alpha, dropout)
    # 只留 LoRA 的 A/B 可训练，其余全部冻结（这才是「参数高效」）
    for name, p in model.named_parameters():
        p.requires_grad = name.endswith("lora_A") or name.endswith("lora_B")
    return model


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------
# 4. 训练 / 评估：一轮 = 前向算 logits → 错一位 → 只在 labels 有效处算 loss
# --------------------------------------------------------------------------
def loss_and_metrics(model, batch, device):
    input_ids, labels, attn = collate(batch, device)
    logits = model(input_ids, attn)[:, :-1, :]
    tgt = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=IGNORE)
    with torch.no_grad():
        valid = tgt.reshape(-1) != IGNORE
        pred = logits.reshape(-1, logits.size(-1)).argmax(-1)
        acc = (pred[valid] == tgt.reshape(-1)[valid]).float().mean().item()
    return loss, acc, int(valid.sum().item())


def evaluate(model, batches, device):
    model.eval()
    tot_loss, tot_tok, tot_correct = 0.0, 0, 0
    with torch.no_grad():
        for b in batches:
            loss, acc, n = loss_and_metrics(model, b, device)
            tot_loss += loss.item() * n
            tot_tok += n
            tot_correct += acc * n
    ce = tot_loss / max(1, tot_tok)
    return {"ce": ce, "ppl": math.exp(min(ce, 20)), "acc": tot_correct / max(1, tot_tok),
            "tokens": tot_tok}


def train_stage(model, tr_batches, va_batches, args, device, stage, history):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    epochs = args.pretrain_epochs if stage == "pretrain" else args.sft_epochs
    total_steps = epochs * max(1, len(tr_batches))
    warmup = max(1, int(total_steps * 0.06))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    best = {"ce": float("inf"), "epoch": -1}
    step = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run_loss, run_tok = 0.0, 0
        for b in tr_batches:
            loss, _, n = loss_and_metrics(model, b, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item() * n
            run_tok += n
        tr_ce = run_loss / max(1, run_tok)
        va = evaluate(model, va_batches, device)
        rec = {"epoch": ep, "lr": sched.get_last_lr()[0], "train_ce": tr_ce,
               "train_ppl": math.exp(min(tr_ce, 20)), "val_ce": va["ce"], "val_ppl": va["ppl"],
               "val_acc": va["acc"], "tokens": va["tokens"], "elapsed_s": round(time.time() - t0, 1)}
        history[stage].append(rec)
        flag = ""
        if va["ce"] < best["ce"]:
            best = {"ce": va["ce"], "epoch": ep}
            torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__,
                        "stoi": stoi_global, "itos": itos_global,
                        "stage": stage, "epoch": ep, "val_ce": va["ce"]},
                       os.path.join(args.artifacts, f"{args.tag}-{stage}-best.pt"))
            flag = " *best"
        print(f"[{stage}] ep{ep:03d}/{epochs} lr={rec['lr']:.2e} train_ce={tr_ce:.4f} "
              f"val_ce={va['ce']:.4f} val_ppl={va['ppl']:.2f} val_acc={va['acc']*100:.2f}%{flag}",
              flush=True)
    return best


# --------------------------------------------------------------------------
# 5. 生成评估：base（续写）vs SFT（问答）—— 作业要的那个「行为变化」
# --------------------------------------------------------------------------
def char_f1(pred, ref):
    from collections import Counter
    a, b = Counter(pred), Counter(ref)
    common = sum((a & b).values())
    if common == 0:
        return 0.0
    p, r = common / max(1, len(pred)), common / max(1, len(ref))
    return 2 * p * r / (p + r)


def run_generation(model, rows, args, device, mode):
    """mode = 'base'：问题直接喂进去（没有角色标记，模型只会接着写）；
       mode = 'sft' ：加 chat template，看它会不会答完就停。"""
    model.eval()
    outs = []
    for row in rows:
        if mode == "sft":
            prompt = f"{U_TAG}{row['instruction']}{A_TAG}"
        else:
            prompt = row["instruction"]
        ids = encode(prompt, stoi_global)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=0.0)
        gen_ids = y[0, len(ids):].tolist()
        text = "".join(itos_global[i] for i in gen_ids if i < len(itos_global))
        stopped = E_ID in gen_ids
        answer = text.replace(E_TAG, "")
        outs.append({
            "instruction": row["instruction"],
            "reference": row["output"],
            "generated": answer,
            "raw": text,
            "stopped": stopped,
            "n_tokens": len(gen_ids),
            "char_f1": round(char_f1(answer, row["output"]), 4),
            "exact_match": int(answer.strip() == row["output"].strip()),
            "echo_prompt": int(row["instruction"][:8] in answer) if len(row["instruction"]) >= 8 else 0,
        })
    return outs


def summarize_generation(outs):
    n = max(1, len(outs))
    return {
        "n": len(outs),
        "stop_rate": round(sum(o["stopped"] for o in outs) / n, 4),
        "avg_len_tokens": round(sum(o["n_tokens"] for o in outs) / n, 1),
        "char_f1": round(sum(o["char_f1"] for o in outs) / n, 4),
        "exact_match": round(sum(o["exact_match"] for o in outs) / n, 4),
        "repeat_prompt_rate": round(sum(o["echo_prompt"] for o in outs) / n, 4),
    }


# --------------------------------------------------------------------------
# 6. 曲线图：预训练 loss / SFT loss（单轴 CE，PPL 标在图上）
# --------------------------------------------------------------------------
def plot_curves(history, out_png, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    panels = [("pretrain", "① 预训练：next-token，100% 位置有监督"),
              ("sft", "② SFT：Loss Mask，只算回答段")]
    panels = [(k, t) for k, t in panels if history.get(k)]
    if not panels:
        print("[plot] history 为空，跳过")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.6), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        rows = history[key]
        xs = [r["epoch"] for r in rows]
        ax.plot(xs, [r["train_ce"] for r in rows], color="#2f6fed", lw=1.9, label="train CE")
        ax.plot(xs, [r["val_ce"] for r in rows], color="#e2554a", lw=1.9, label="val CE")
        best = min(rows, key=lambda r: r["val_ce"])
        ax.scatter([best["epoch"]], [best["val_ce"]], color="#e2554a", s=55, zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"best: ep{best['epoch']}\nval CE={best['val_ce']:.3f}\nPPL={best['val_ppl']:.1f}",
                    xy=(best["epoch"], best["val_ce"]),
                    xytext=(8, 26), textcoords="offset points", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f4", ec="#e2554a", lw=0.8),
                    arrowprops=dict(arrowstyle="->", color="#e2554a", lw=0.8))
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy (nats, per char)")
        ax.grid(alpha=0.25, ls=":")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title(title, fontsize=11)
        lo = min(min(r["train_ce"] for r in rows), min(r["val_ce"] for r in rows))
        hi = max(max(r["train_ce"] for r in rows), max(r["val_ce"] for r in rows))
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.18 * (hi - lo))

    n_pre = len(history.get("pretrain") or [])
    n_sft = len(history.get("sft") or [])
    fig.suptitle(f"字符级 decoder-only Transformer 两阶段训练（tag={tag}｜预训练 {n_pre} 轮 + SFT {n_sft} 轮）",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


# --------------------------------------------------------------------------
# 7. 主流程
# --------------------------------------------------------------------------
stoi_global, itos_global = {}, []


def build_model(cfg, device, args, with_lora=False):
    model = GPT(cfg).to(device)
    if with_lora and args.peft == "lora":
        apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    total, trainable = param_counts(model)
    print(f"[model] 总参数 {total/1e6:.2f}M ｜ 可训练 {trainable/1e6:.2f}M "
          f"({trainable/total*100:.2f}%) ｜ peft={args.peft if with_lora else 'none'}", flush=True)
    return model


def main():
    global stoi_global, itos_global
    ap = argparse.ArgumentParser(description="Week5 SFT: decoder-only LM, pretrain → SFT + loss mask")
    ap.add_argument("--stage", default="both", choices=["both", "pretrain", "sft"])
    ap.add_argument("--tag", default="sft", help="产物前缀，例如 sft / lora")
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--train_file", default="sft_train.jsonl")
    ap.add_argument("--val_file", default="sft_val.jsonl")
    ap.add_argument("--pretrain_corpus", default="",
                    help="预训练用纯文本语料（默认空=把 SFT 训练集的问答拼成文本流）")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--block_size", type=int, default=192)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--sft_epochs", type=int, default=30)
    ap.add_argument("--peft", default="none", choices=["none", "lora"])
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--sft_mask", default="answer", choices=["answer", "full"],
                    help="answer=只在回答段算 loss（课件做法）；full=对照组，不 mask")
    ap.add_argument("--init_ckpt", default="", help="SFT 阶段的初始化权重路径，默认 <tag>-pretrain-best.pt")
    ap.add_argument("--gen_n", type=int, default=20, help="生成评估用多少条验证集样本")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--plot_png", default="sft-curves.png")
    args = ap.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)
    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    hist_path = os.path.join(args.artifacts, f"{args.tag}-history.json")

    # ---- 只重画曲线 ----
    if args.plot_only:
        if not os.path.exists(hist_path):
            sys.exit(f"找不到 {hist_path}，先训练一次")
        with open(hist_path, encoding="utf-8") as f:
            history = json.load(f)
        plot_curves(history, args.plot_png, args.tag)
        return

    device = pick_device(args.device)
    print(f"[env] device={device} torch={torch.__version__}", flush=True)

    train_rows = read_jsonl(os.path.join(args.data_dir, args.train_file))
    val_rows = read_jsonl(os.path.join(args.data_dir, args.val_file))
    corpus_text = ""
    if args.pretrain_corpus:
        p = os.path.join(args.data_dir, args.pretrain_corpus)
        with open(p, encoding="utf-8", errors="ignore") as f:
            corpus_text = f.read()
        print(f"[data] 预训练语料 {p}: {len(corpus_text)} 字", flush=True)
    stoi_global, itos_global = build_vocab(train_rows, args.max_vocab, corpus_text)
    print(f"[data] train={len(train_rows)} val={len(val_rows)} vocab={len(itos_global)}", flush=True)

    cfg = GPTConfig(vocab_size=len(itos_global), block_size=args.block_size,
                    d_model=args.d_model, n_layer=args.n_layer,
                    n_head=args.n_head, dropout=args.dropout)

    history = {"config": vars(args), "pretrain": [], "sft": []}
    if os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as f:
            old = json.load(f)
        history["pretrain"] = old.get("pretrain", []) if args.stage != "pretrain" else []
        history["sft"] = old.get("sft", []) if args.stage != "sft" else []

    # ---------------- 阶段一：预训练（base model = 概率续写器） ----------------
    model = None
    if args.stage in ("both", "pretrain"):
        model = build_model(cfg, device, args, with_lora=False)
        blocks = make_pretrain_stream(train_rows, stoi_global, args.block_size, corpus_text)
        rng.shuffle(blocks)
        split = max(1, int(len(blocks) * 0.05))
        pre_va, pre_tr = blocks[:split], blocks[split:]
        print(f"[pretrain] blocks={len(blocks)} (train={len(pre_tr)} val={len(pre_va)}) "
              f"tokens={sum(len(b[0]) for b in blocks)}", flush=True)
        tr_b = make_batches(pre_tr, args.batch_size, rng, shuffle=True)
        va_b = make_batches(pre_va, args.batch_size, rng, shuffle=False)
        history["pretrain"] = []
        best = train_stage(model, tr_b, va_b, args, device, "pretrain", history)
        print(f"[pretrain] best val_ce={best['ce']:.4f} @ ep{best['epoch']}", flush=True)
        ckpt = os.path.join(args.artifacts, f"{args.tag}-pretrain-best.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])

    # ---------------- 阶段二：SFT（chat template + Loss Mask） ----------------
    if args.stage in ("both", "sft"):
        if args.stage == "sft":
            ckpt = args.init_ckpt or os.path.join(args.artifacts, f"{args.tag}-pretrain-best.pt")
            if not os.path.exists(ckpt):
                sys.exit(f"找不到预训练权重 {ckpt}；先跑 --stage pretrain 或 --stage both")
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            if "itos" in ck:  # 词表跟着权重走，保证 id 对齐
                stoi_global, itos_global = ck["stoi"], ck["itos"]
                print(f"[sft] 从 ckpt 恢复词表 vocab={len(itos_global)}", flush=True)
            cfg = GPTConfig(**ck["cfg"])
            model = GPT(cfg).to(device)  # 先按基座结构载入，再按需套 LoRA
            model.load_state_dict(ck["model"])
            print(f"[sft] 已载入预训练权重 {ckpt}", flush=True)

        if args.peft == "lora":
            apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
            total, trainable = param_counts(model)
            print(f"[model] LoRA 已挂载 ｜ 总参数 {total/1e6:.2f}M ｜ 可训练 {trainable/1e6:.2f}M "
                  f"({trainable/total*100:.2f}%)", flush=True)

        tr_ex = [make_chat_example(r, stoi_global, args.sft_mask) for r in train_rows]
        va_ex = [make_chat_example(r, stoi_global, args.sft_mask) for r in val_rows]
        masked = sum(1 for ids, lab in tr_ex for x in lab if x == IGNORE)
        total_tok = sum(len(ids) for ids, _ in tr_ex)
        print(f"[sft] mask={args.sft_mask} 样本={len(tr_ex)} 总 token={total_tok} 被 mask={masked} "
              f"({masked/total_tok*100:.1f}% 位置不参与 loss)", flush=True)
        tr_b = make_batches(tr_ex, args.batch_size, rng, shuffle=True)
        va_b = make_batches(va_ex, args.batch_size, rng, shuffle=False)
        history["sft"] = []
        best = train_stage(model, tr_b, va_b, args, device, "sft", history)
        print(f"[sft] best val_ce={best['ce']:.4f} @ ep{best['epoch']}", flush=True)

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    plot_curves(history, args.plot_png, args.tag)

    # ---------------- 生成评估：base 续写 vs SFT 问答 ----------------
    gen_rows = val_rows[: args.gen_n]
    model.load_state_dict(torch.load(os.path.join(args.artifacts, f"{args.tag}-sft-best.pt"),
                                     map_location=device)["model"])
    sft_outs = run_generation(model, gen_rows, args, device, mode="sft")
    sft_sum = summarize_generation(sft_outs)

    base_path = os.path.join(args.artifacts, f"{args.tag}-pretrain-best.pt")
    base_sum, base_outs = None, []
    if os.path.exists(base_path):
        base_model = GPT(cfg).to(device)
        base_model.load_state_dict(torch.load(base_path, map_location=device)["model"])
        base_outs = run_generation(base_model, gen_rows, args, device, mode="base")
        base_sum = summarize_generation(base_outs)

    report = {"sft": sft_sum, "base": base_sum,
              "sft_samples": sft_outs[: args.gen_n], "base_samples": base_outs[: args.gen_n]}
    with open(os.path.join(args.artifacts, f"{args.tag}-generation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== 生成评估（贪心解码，验证集前 %d 条）=====" % args.gen_n)
    for name, s in (("base (续写)", base_sum), ("SFT  (问答)", sft_sum)):
        if s:
            print(f"{name}: 生成 <|end|> 停止率={s['stop_rate']*100:.0f}% "
                  f"平均长度={s['avg_len_tokens']} tok 字符F1={s['char_f1']:.3f} "
                  f"完全匹配={s['exact_match']*100:.1f}%")
    print("\n----- 样例 -----")
    for i in range(min(3, len(sft_outs))):
        o = sft_outs[i]
        print(f"[{i}] Q: {o['instruction'][:46]}")
        print(f"    ref: {o['reference'][:60]}")
        if base_outs:
            print(f"    base: {base_outs[i]['generated'][:60]}")
        print(f"    sft : {o['generated'][:60]}")
    print(f"\n[done] 曲线 {args.plot_png} ｜ 指标 {args.artifacts}/{args.tag}-*.json")


if __name__ == "__main__":
    main()

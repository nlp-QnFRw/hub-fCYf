"""
model.py -- 从零实现的 decoder-only Transformer 语言模型（GPT 风格）

结构（pre-LN，与 GPT-2 一致）：

    idx ─┬─> Token Embedding ─┐
         └─> Pos Embedding ───┴─> Dropout
                                     │
              ┌──────────────────────┘
              │  × n_layer
              │   x = x + Attention(LN(x))     ← 因果掩码：只能看左边
              │   x = x + MLP(LN(x))
              ▼
            LN ─> Linear(vocab) ─> logits        （Linear 与 Token Embedding 共享权重）

与课件里 RNN/LSTM 版本的关键区别：
    1. 没有循环，序列内所有位置一次并行算完 —— 训练快得多；
    2. 靠「因果掩码」而不是「时间步递推」保证只看上文；
    3. 位置信息由可学习的位置编码提供，不是隐状态天然携带的。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """模型超参。`block_size` 同时是训练上下文长度和位置编码的最大长度。"""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size 必须 >= 1，当前 {self.vocab_size}")
        if self.block_size < 1:
            raise ValueError(f"block_size 必须 >= 1，当前 {self.block_size}")
        if self.n_head < 1 or self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd({self.n_embd}) 必须能被 n_head({self.n_head}) 整除"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


class CausalSelfAttention(nn.Module):
    """多头自注意力 + 因果掩码。

    q/k/v 用一个 Linear 一次算出（3*n_embd 列再切开），比三个独立 Linear 少两次 kernel 启动。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # 下三角为 True：位置 i 可以看 j <= i。persistent=False 让 block_size
        # 不同的 checkpoint 也能互相加载（掩码是纯计算产物，不必存进 state_dict）。
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        if T > self.mask.size(-1):
            raise ValueError(
                f"序列长度 {T} 超过 block_size {self.mask.size(-1)}，请缩短输入或增大 block_size"
            )

        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # 被遮住的位置塞一个极小值，softmax 之后就是 0
        att = att.masked_fill(~self.mask[:, :, :T, :T], torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        weights = att
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return (y, weights) if return_attn else (y, None)


class MLP(nn.Module):
    """位置无关的前馈层：中间维度 4×n_embd，GELU 激活。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """一个 Transformer 层：pre-LN + 残差。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        a, w = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, w


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # 权重绑定：输出层与输入 embedding 共享同一份参数（省 vocab*n_embd 个参数，
        # 同时让「输入的字」和「预测的字」处在同一个向量空间）。
        self.head.weight = self.tok_emb.weight
        # 残差分支的输出投影缩小初始化，避免层数一多激活值爆炸
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attn: bool = False):
        """idx: (B, T) int64。targets 给出时返回 (logits, loss)。"""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"输入长度 {T} 超过 block_size {self.cfg.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        attns = []
        for blk in self.blocks:
            x, w = blk(x, return_attn=return_attn)
            if return_attn:
                attns.append(w)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        if return_attn:
            return logits, loss, attns
        return logits, loss

    # ─────────────────────── 参数与优化器 ───────────────────────

    @property
    def num_params(self) -> int:
        """去重后的参数量（绑定的权重只算一次）。"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_params_non_embedding(self) -> int:
        return self.num_params - self.tok_emb.weight.numel() - self.pos_emb.weight.numel()

    def configure_optimizers(self, lr: float, weight_decay: float,
                             betas=(0.9, 0.95)) -> torch.optim.Optimizer:
        """二维以上参数（Linear/Embedding）才做 weight decay；
        bias 和 LayerNorm 的缩放/偏移不衰减 —— 这是训练 Transformer 的通行做法。
        """
        decay, no_decay = [], []
        seen = set()
        for _, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    # ─────────────────────── 生成 ───────────────────────

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """自回归采样。idx: (B, T) 的提示词，返回 (B, T + max_new_tokens)。"""
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0")
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                # 累计概率超过 top_p 的整段丢掉，但要保留第一个（保证至少有一个可选项）
                removed = cumulative - probs > top_p
                sorted_logits = sorted_logits.masked_fill(removed, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

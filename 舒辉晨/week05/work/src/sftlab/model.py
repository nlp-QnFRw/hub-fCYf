"""小型 decoder-only Transformer（约 4M 参数）。

它不是什么新架构，就是 PPT 第三部分讲的那个 Decoder-only：
    Token Embedding + Position Embedding
    × N 个 Block（LayerNorm → 因果自注意力 → 残差 → LayerNorm → FFN → 残差）
    LayerNorm → Linear(词表) → 每个位置的 next-token logits

唯一需要特别注意的是「因果」：第 t 个位置只能看见 ≤ t 的位置，
实现方式就是在 softmax 之前把「未来」位置的分数填成 -inf —— 也就是 PPT 第 9 页那张 mask。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True

    @classmethod
    def from_dict(cls, d: dict, vocab_size: int) -> "GPTConfig":
        allowed = {f for f in cls.__dataclass_fields__} - {"vocab_size"}
        return cls(vocab_size=vocab_size, **{k: v for k, v in d.items() if k in allowed})


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # 下三角矩阵就是「谁见谁」表：mask[t, s] = 1 表示位置 t 可以看位置 s。
        # 注册成 buffer 会跟着模型一起搬到 GPU，但不参与训练。
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # (B, T, C) → (B, n_head, T, head_dim)：把通道维拆成「多头 × 每头维度」
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 缩放点积注意力：除以 sqrt(d) 是为了让 T 变大时 scores 的方差不至于爆炸
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 未来位置填 -inf：softmax 之后这些位置的概率就是 0，等于完全看不见
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """前馈网络：先升维到 4 倍再压回去，中间加非线性。"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN 结构：归一化放在子层之前，训练更稳，是小模型的标准做法。"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # 输入 embedding 和输出投影共享权重：小词表小模型能省近一半参数，
            # 而且「同一个字进出用同一个向量」本身也是合理的归纳偏置。
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # 残差分支的输出投影缩小初始化：层数越多，残差累加越容易数值发散
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长 {T} 超过 block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # 关键的一步：把序列向左错一位再算交叉熵。
            # logits[:, :-1] 是「看到前 t 个 token 时的预测」，labels[:, 1:] 是对应的「下一个 token」。
            # labels 里的 -100（prompt 段 + padding）会被 ignore_index 直接跳过。
            # logits 用 float32 算交叉熵：bf16 下 log_softmax 容易掉精度，
            # 损失值本身也小（2~4 左右），用低精度会抖。
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ── 训练用 ────────────────────────────────────────────────────────────────

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.Optimizer:
        """权重衰减只加在矩阵上；LayerNorm 和 bias 不衰减（衰减它们只会拖慢收敛）。"""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

"""
bert_model.py -- 从零实现的 BERT（encoder-only），可直接加载 bert-base-chinese 的权重

为什么不用 transformers：
    这台机器的 transformers 4.55 与已装的 tokenizers / huggingface-hub 版本元数据冲突，
    import 就会抛 ImportError。与其去改别人的环境，不如把结构写出来——
    反正 BERT 的结构比 GPT 还简单，而且这样能一眼看清每个权重用在哪。

和 model.py 里的 GPT 正好互为镜像：
    GPT : 因果掩码，每个位置只看左边        → 能生成
    BERT: 无掩码，每个位置看左右两边        → 只能填空（MLM）

权重兼容性：
    bert-base-chinese 是老版 checkpoint，LayerNorm 的参数名是 gamma/beta 而不是 weight/bias，
    所以下面自己写了 LayerNorm 而不是用 nn.LayerNorm。
    加载时用 strict=True —— 只要有一个键名或形状对不上就会报错，这本身就是一次结构校验。
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# checkpoint 里注意力掩码用的是 -10000 而不是 -inf（老实现习惯，保持数值一致）
MASK_VALUE = -10000.0


def resolve_bert_dir(explicit: str | None = None) -> str:
    """按优先级找 bert-base-chinese 目录，找不到就给出可操作的报错。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        explicit,
        os.environ.get("BERT_DIR"),
        os.path.join(here, "..", "bert-base-chinese"),
        r"D:\Baidu\BaiduNetdiskDownload\bert-base-chinese",
        os.path.join(here, "..", "..", "pretrain_models", "bert-base-chinese"),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "config.json")) \
                and os.path.isfile(os.path.join(c, "pytorch_model.bin")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "找不到 bert-base-chinese（需要 config.json + pytorch_model.bin）。\n"
        "用 --bert-dir 指定，或设置环境变量 BERT_DIR。尝试过：\n  "
        + "\n  ".join(str(c) for c in candidates if c)
    )


class BertConfig:
    def __init__(self, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, vocab_size=21128, max_position_embeddings=512,
                 type_vocab_size=2, layer_norm_eps=1e-12, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, pad_token_id=0):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.pad_token_id = pad_token_id
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str) -> "BertConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                "vocab_size", "max_position_embeddings", "type_vocab_size", "layer_norm_eps",
                "hidden_dropout_prob", "attention_probs_dropout_prob", "pad_token_id"]
        return cls(**{k: d[k] for k in keys if k in d})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class LayerNorm(nn.Module):
    """老版 checkpoint 用 gamma/beta 命名，这里保持一致以便 strict 加载。"""

    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)


class BertEmbeddings(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.LayerNorm = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        T = input_ids.size(1)
        if T > self.position_embeddings.num_embeddings:
            raise ValueError(f"序列长度 {T} 超过 max_position_embeddings "
                             f"{self.position_embeddings.num_embeddings}")
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = (self.word_embeddings(input_ids)
             + self.position_embeddings(pos)
             + self.token_type_embeddings(token_type_ids))
        return self.dropout(self.LayerNorm(x))


class BertSelfAttention(nn.Module):
    """和 GPT 的注意力唯一的区别：没有下三角掩码，所以能看到右边。"""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

    def forward(self, x, attention_mask=None, return_attn=False):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask          # (B,1,1,T)
        weights = F.softmax(scores, dim=-1)
        ctx = torch.matmul(self.dropout(weights), v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return (ctx, weights) if return_attn else (ctx, None)


class BertSelfOutput(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.LayerNorm = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, residual):
        return self.LayerNorm(self.dropout(self.dense(x)) + residual)


class BertAttention(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(cfg)
        self.output = BertSelfOutput(cfg)

    def forward(self, x, attention_mask=None, return_attn=False):
        ctx, w = self.self(x, attention_mask, return_attn=return_attn)
        return self.output(ctx, x), w


class BertIntermediate(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x):
        return F.gelu(self.dense(x))      # config 里 hidden_act=gelu，即精确 erf 版本


class BertOutput(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.LayerNorm = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, residual):
        return self.LayerNorm(self.dropout(self.dense(x)) + residual)


class BertLayer(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.attention = BertAttention(cfg)
        self.intermediate = BertIntermediate(cfg)
        self.output = BertOutput(cfg)

    def forward(self, x, attention_mask=None, return_attn=False):
        x, w = self.attention(x, attention_mask, return_attn=return_attn)
        x = self.output(self.intermediate(x), x)
        return x, w


class BertEncoder(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x, attention_mask=None, return_attn=False):
        attns = []
        for layer in self.layer:
            x, w = layer(x, attention_mask, return_attn=return_attn)
            if return_attn:
                attns.append(w)
        return x, (attns if return_attn else None)


class BertPooler(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x):
        return torch.tanh(self.dense(x[:, 0]))


class BertModel(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
        self.pooler = BertPooler(cfg)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                return_attn=False):
        x = self.embeddings(input_ids, token_type_ids)
        ext = None
        if attention_mask is not None:
            ext = (1.0 - attention_mask[:, None, None, :].float()) * MASK_VALUE
        x, attns = self.encoder(x, ext, return_attn=return_attn)
        return x, self.pooler(x), attns


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.LayerNorm = LayerNorm(cfg.hidden_size, cfg.layer_norm_eps)

    def forward(self, x):
        return self.LayerNorm(F.gelu(self.dense(x)))


class BertLMPredictionHead(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.transform = BertPredictionHeadTransform(cfg)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(self, x):
        return self.decoder(self.transform(x)) + self.bias


class BertOnlyMLMHead(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.predictions = BertLMPredictionHead(cfg)


class BertForMaskedLM(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg
        self.bert = BertModel(cfg)
        self.cls = BertOnlyMLMHead(cfg)
        # checkpoint 里还有 cls.seq_relationship.*（NSP 头），推理用不到，
        # 但留着才能让 strict 加载通过 —— 也顺便证明了结构完全对上。
        self.cls.seq_relationship = nn.Linear(cfg.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, return_attn=False):
        # 注意：默认不收集注意力权重，否则训练时会把每层 (B,H,T,T) 都留在显存里
        seq_out, _, attns = self.bert(input_ids, attention_mask, token_type_ids,
                                      return_attn=return_attn)
        logits = self.cls.predictions(seq_out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        if return_attn:
            return logits, loss, attns
        return logits, loss

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, lr: float, weight_decay: float, betas=(0.9, 0.999)):
        """BERT 微调的通行做法：bias、LayerNorm 的 gamma/beta 不做 weight decay。"""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or name.endswith(".bias") or name.endswith(".gamma") \
                    or name.endswith(".beta"):
                no_decay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas)


# ─────────────────────────── 加载预训练权重 ───────────────────────────

def load_pretrained(bert_dir: str, device="cpu", dtype=torch.float32):
    cfg = BertConfig.from_json(os.path.join(bert_dir, "config.json"))
    model = BertForMaskedLM(cfg)
    state = torch.load(os.path.join(bert_dir, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"权重与结构对不上 —— 缺少 {len(missing)} 项 {missing[:5]}，"
            f"多余 {len(unexpected)} 项 {unexpected[:5]}"
        )
    return model.to(device=device, dtype=dtype), cfg


@torch.no_grad()
def pseudo_perplexity(model: BertForMaskedLM, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor, mask_id: int) -> tuple[float, int]:
    """伪困惑度（Salazar et al., 2020）。

    做法：对序列里每个位置逐个遮住，让模型在「看得见左右两边」的条件下预测它，
    再对所有位置的对数概率求平均、取指数。
    这样得到的数和 GPT 的 PPL 同量纲（都是 exp(平均每 token 的负对数似然)），
    唯一区别是 BERT 偷看了右边 —— 这也正是它数值更低的原因。
    """
    model.eval()
    total_logp, count = 0.0, 0
    for b in range(input_ids.size(0)):
        length = int(attention_mask[b].sum().item())
        positions = list(range(1, max(1, length - 1)))     # 跳过 [CLS] 和 [SEP]
        if not positions:
            continue
        variants = input_ids[b: b + 1].repeat(len(positions), 1).clone()
        for j, p in enumerate(positions):
            variants[j, p] = mask_id
        vmask = attention_mask[b: b + 1].repeat(len(positions), 1)
        logits, _ = model(variants, vmask)
        rows = torch.arange(len(positions), device=logits.device)
        pos = torch.tensor(positions, device=logits.device)
        logp = F.log_softmax(logits[rows, pos].float(), dim=-1)
        total_logp += float(logp[rows, input_ids[b, pos]].sum().item())
        count += len(positions)
    if count == 0:
        return float("nan"), 0
    return math.exp(-total_logp / count), count

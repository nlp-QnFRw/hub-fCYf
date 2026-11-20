#region 0. 模块说明
"""
BERT 判别式分类的**模型与推理核心**（train_bert.py 与 predict_bert.py 共用）。

为什么不用 `AutoModelForSequenceClassification`：
  1. 它的池化固定为 CLS，而"pool=cls vs mean"是必做的消融项；
  2. 它的 classifier_dropout 在不同版本里取值规则不同（transformers 5.x 有变化），
     显式控制更稳；
  3. 自己拼 `backbone + pooling + Linear` 只有 20 行，换来的是**池化与损失的完全可控**
     （class_weight 需要自己算 loss，不能交给 model(labels=...)）。

显存与速度（本机实测，见 docs/PROBE.md）：
  bs=32 / L=256 → 121 samples/s，峰值 4.3GB。**权重保持 fp32、只用 autocast**；
  一旦把模型 .half() 再配 GradScaler，会报 "Attempting to unscale FP16 gradients"。

使用方式：
  from bert.model_bert import build_model, get_tokenized, run_eval
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import BertDataset, build_dataloaders, load_split, tokenize_records
from common.labels import NUM_LABELS
from common.paths import CACHE, model_path, split_file

#endregion

#region 2. 模型
class BertClassifier(nn.Module):
    """BERT 主干 + 池化 + 线性头。

    参数：
      backbone      : 已加载的 transformers 主干（AutoModel，输出 last_hidden_state）
      num_labels    : 类别数
      pool          : "cls" | "mean"；mean 必须用 attention_mask 掩掉 PAD，
                      否则 PAD 位置的 embedding 会稀释句向量（隐形 bug 常客）
      dropout       : 分类头 dropout 概率
      freeze_layers : 冻结底部 N 层（0 表示全参微调）
    """

    def __init__(self, backbone, num_labels: int = NUM_LABELS, pool: str = "cls",
                 dropout: float = 0.1, freeze_layers: int = 0) -> None:
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool 只能是 cls/mean，收到 {pool}")
        self.backbone = backbone
        self.pool = pool
        self.hidden = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden, num_labels)

        if freeze_layers > 0:
            # 冻结 embedding 与底部 N 层：省显存、加速，用于回答"冻结一半能省多少且掉多少"
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
            for layer in self.backbone.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor | None = None,
                class_weight: torch.Tensor | None = None) -> dict:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, L, H)
        if self.pool == "cls":
            pooled = h[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)  # (B, L, 1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            # 自己算 loss 而不是传给 backbone：这样才能注入 class_weight
            loss = F.cross_entropy(logits, labels, weight=class_weight)
        return {"loss": loss, "logits": logits, "pooled": pooled}


def build_model(alias: str = "bert-base", pool: str = "cls", dropout: float = 0.1,
                freeze_layers: int = 0, num_labels: int = NUM_LABELS) -> BertClassifier:
    """按别名构建模型，并打印参数量（总量 / 可训练 / 占比）。

    参数：
      alias : common/paths.py 里的短名（bert-base / roberta-wwm / macbert）
    返回：
      BertClassifier（**未搬到 GPU**，由调用方 .to(device)）
    """
    from transformers import AutoModel

    path = model_path(alias)
    # sdpa 显式指定：sm_120 没有 flash-attn，不指定会静默回落到 eager（慢 30% 且不报错）
    backbone = AutoModel.from_pretrained(path, attn_implementation="sdpa")
    model = BertClassifier(backbone, num_labels=num_labels, pool=pool,
                           dropout=dropout, freeze_layers=freeze_layers)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] {alias} 参数 总 {total / 1e6:.1f}M / 可训练 {trainable / 1e6:.1f}M "
          f"({trainable / total * 100:.1f}%) | pool={pool} dropout={dropout} freeze={freeze_layers}")
    return model


def make_param_groups(model: BertClassifier, lr: float, head_lr_mult: float = 3.0,
                      weight_decay: float = 0.01) -> list[dict]:
    """分层学习率 + 权重衰减分组。

    规则（业界惯例，也是本项目的固定约定）：
      - 主干用 lr，线性头用 lr * head_lr_mult（头是随机初始化的，学得更快才合理）
      - bias 与 LayerNorm 不做 weight decay（对它们做衰减会稳定地掉点）
    """
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    head_ids = {id(p) for p in head_params}

    decay, no_decay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim <= 1 or name.endswith(".bias"):  # LayerNorm.weight / bias
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": head, "lr": lr * head_lr_mult, "weight_decay": weight_decay},
    ]
#endregion

#region 3. tokenize 缓存（同一配置只算一次）
def _cache_key(split: str, max_len: int, mode: str, alias: str,
               limit: int | None) -> Path:
    """缓存键必须包含**数据指纹**（切分文件大小 + mtime）。

    踩过的坑：键里只写 split 名，重新生成切分后仍然命中旧缓存，
    于是"100k 训练"实际只训了冒烟阶段的 1200 条，val 卡在 0.44 而没人发现。
    把文件大小与修改时间纳入键，切分一变键就变，自然失效。
    """
    p = split_file(split)
    try:
        st = p.stat()
        fp = f"{st.st_size}_{int(st.st_mtime)}"
    except (FileNotFoundError, OSError):
        fp = "nodata"
    raw = f"{split}|{limit or 'all'}|{max_len}|{mode}|{alias}|{fp}".encode()
    h = hashlib.md5(raw).hexdigest()[:10]
    return CACHE / f"tok_{split}_{limit or 'all'}_{max_len}_{mode}_{h}.pt"


def get_tokenized(split: str, max_len: int, mode: str, alias: str = "bert-base",
                  limit: int | None = None, use_cache: bool = True) -> tuple[BertDataset, list[dict]]:
    """读取切分并 tokenize（带磁盘缓存）。

    为什么缓存：tokenize 100k 条约 2.5 分钟，20 个 run 重复 tokenize 会白烧 40 分钟。
    缓存键见 `_cache_key`（含数据指纹），切分重新生成后自动失效，不会串味。

    返回：
      (BertDataset, records)
    """
    from transformers import AutoTokenizer

    recs = load_split(split, limit=limit)
    path = _cache_key(split, max_len, mode, alias, limit)
    if use_cache and path.exists():
        obj = torch.load(path, weights_only=False)
        n_cached = obj["input_ids"].shape[0]
        if n_cached == len(recs):  # 条数对不上说明缓存与当前切分不一致，重算
            return BertDataset(obj["input_ids"], obj["attention_mask"], obj["labels"]), recs
        print(f"[缓存失效] {path.name} 条数 {n_cached} != 当前 {len(recs)}，重新 tokenize")

    tok = AutoTokenizer.from_pretrained(model_path(alias))
    ds = tokenize_records(recs, tok, max_len=max_len, mode=mode, with_labels=True)
    if use_cache:
        torch.save({"input_ids": ds.input_ids, "attention_mask": ds.attention_mask,
                    "labels": ds.labels}, path)
    return ds, recs
#endregion

#region 4. 评估与推理
@torch.no_grad()
def run_eval(model: BertClassifier, loader, device: torch.device,
             class_weight: torch.Tensor | None = None) -> dict:
    """在一个 DataLoader 上推理，返回 {preds, golds, probs, loss, ms_per_sample}。

    与训练循环解耦：训练每 epoch 调它算 val macro-F1，predict 脚本也调它出最终预测，
    保证"选模型的指标"与"汇报的指标"是同一份代码算的。
    """
    model.eval()
    all_pred, all_gold, all_prob = [], [], []
    total_loss, n = 0.0, 0
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    import time

    t0 = time.time()
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        am = batch["attention_mask"].to(device, non_blocking=True)
        y = batch.get("labels")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(ids, am, labels=y.to(device) if y is not None else None,
                        class_weight=class_weight)
        if out["loss"] is not None:
            total_loss += out["loss"].item() * ids.size(0)
        all_prob.append(torch.softmax(out["logits"].float(), dim=-1).cpu().numpy())
        all_pred.append(out["logits"].argmax(dim=-1).cpu().numpy())
        if y is not None:
            all_gold.append(y.numpy())
        n += ids.size(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {
        "preds": np.concatenate(all_pred) if all_pred else np.array([], dtype=int),
        "golds": np.concatenate(all_gold) if all_gold else np.array([], dtype=int),
        "probs": np.vstack(all_prob) if all_prob else np.zeros((0, NUM_LABELS)),
        "loss": total_loss / max(1, n),
        "ms_per_sample": elapsed / max(1, n) * 1000,
    }
#endregion

#region 5. 自检
if __name__ == "__main__":
    from common.seed import set_seed

    set_seed(42)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    m = build_model("bert-base", pool="cls")
    ids = torch.randint(0, 21000, (2, 32))
    am = torch.ones_like(ids)
    out = m(ids, am, labels=torch.tensor([0, 13]))
    print("logits", tuple(out["logits"].shape), "loss", round(out["loss"].item(), 4))
    m2 = build_model("bert-base", pool="mean", freeze_layers=6)
    out2 = m2(ids, am)
    print("mean-pool 前向 ok", tuple(out2["logits"].shape))
    groups = make_param_groups(m2, lr=2e-5)
    print("参数组：", [len(g["params"]) for g in groups])
#endregion

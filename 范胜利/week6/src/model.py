# -*- coding: utf-8 -*-
"""
model.py —— 基座无关的分类模型（两个基座共用同一份结构代码）

【只改两处：池化方式、输出头】
    BERT/RoBERTa 本体 12 层完全不动，后面接：
        池化（cls / mean / max） → Dropout → Linear(768, 15)
    「换基座」= 换 AutoModel.from_pretrained 的目录，结构代码一行不改。
    这是本作业能保证「同配置」的前提：如果两个基座各写一个模型类，
    迟早会出现 dropout、池化、初始化不一样的情况。

【池化为什么要 mask 掉 [PAD]】
    短文本 padding 到 64，有效 token 可能只有 20 个。
    mean 池化若不乘 attention_mask，44 个 PAD 位置的向量会把句向量稀释掉；
    max 池化若不把 PAD 压到 -inf，最大值可能来自 PAD 位置的偏置项。
    这两个 bug 都不报错，只是指标莫名低几个点——最费时间的一类问题。
"""

import torch
import torch.nn as nn

from labels import NUM_LABELS


class BackboneClassifier(nn.Module):
    def __init__(self, model_dir: str, num_labels: int = NUM_LABELS, pool: str = "cls", dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        # 分类头显式初始化：默认 nn.Linear 的初始化在 15 类上会让初期 loss 偏大，
        # 显式用正态 + 小方差，配合 warmup 更稳（两个基座同样处理）
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                      # [B, L, H]

        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()    # [B, L, 1]
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            elif self.pool == "max":
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
            else:
                raise ValueError(f"未知池化方式 {self.pool}")
        return self.classifier(self.dropout(vec))

    def backbone_params(self):
        return list(self.backbone.parameters())


def count_params(model: nn.Module) -> dict:
    """参数量（百万）与可训练参数量，用来解释「同样的结构、不同的预训练」这个前提"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_m": round(total / 1e6, 2), "trainable_m": round(trainable / 1e6, 2)}

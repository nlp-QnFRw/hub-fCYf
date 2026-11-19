"""BERT 分类器定义（训练与推理共用，避免模型结构两处不一致）"""

import torch
import torch.nn as nn


def load_pretrained(path):
    """
    兼容 transformers 4.x 与 5.x：4.x 用 torch_dtype，5.x 用 dtype。
    写死任一个都会在另一个版本上炸，这是切换环境时最常见的一类报错。
    """
    from transformers import AutoModel
    try:
        return AutoModel.from_pretrained(str(path), dtype=torch.float32)
    except TypeError:
        return AutoModel.from_pretrained(str(path), torch_dtype=torch.float32)


class BertClassifier(nn.Module):
    """
    预训练编码器 + 池化 + 线性头。

    pool：
      cls  —— 取 [CLS]（预训练就为它设计）
      mean —— 有效 token 均值，长文本上通常更稳
      max  —— 逐维最大，容易放大噪声
    三种都要 mask 掉 [PAD]，否则补位向量会污染统计量。
    """

    def __init__(self, model_dir, num_labels=10, pool="cls", dropout=0.1):
        super().__init__()
        self.bert = load_pretrained(model_dir)
        self.pool = pool
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))

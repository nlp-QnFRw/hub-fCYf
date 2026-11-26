"""模型层：中文 BERT 编码，Linear 输出标签分数，可选 CRF 联合解码。"""
import torch
from torch import nn
from transformers import AutoModel

class CRF(nn.Module):
    """线性链 CRF：学习相邻标签的转移分数，并用维特比算法找最优序列。"""

    def __init__(self, size):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(size))
        self.end = nn.Parameter(torch.zeros(size))
        self.transitions = nn.Parameter(torch.zeros(size, size))

    def loss(self, emissions, tags, mask):
        """负对数似然 = 所有路径的 log-sum-exp 分数 - 正确路径分数。"""
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        # alpha 通过前向动态规划累计所有可能路径的分数。
        alpha = self.start + emissions[:, 0]
        for i in range(1, emissions.size(1)):
            step = self.transitions[tags[:, i - 1], tags[:, i]] + emissions[:, i].gather(1, tags[:, i:i + 1]).squeeze(1)
            score = score + step * mask[:, i]
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, i:i + 1], nxt, alpha)
        last = tags.gather(1, (mask.long().sum(1) - 1).unsqueeze(1)).squeeze(1)
        return (torch.logsumexp(alpha + self.end, dim=1) - score - self.end[last]).mean()

    def decode(self, emissions, mask):
        """维特比解码：逐位置记录最佳前驱，最后倒序回溯。"""
        results = []
        for e, m in zip(emissions, mask):
            e = e[:int(m.sum())]
            score = self.start + e[0]
            history = []
            for step in e[1:]:
                best, indices = (score.unsqueeze(1) + self.transitions).max(dim=0)
                score = best + step
                history.append(indices)
            tag = int((score + self.end).argmax())
            path = [tag]
            for indices in reversed(history):
                tag = int(indices[tag])
                path.append(tag)
            results.append(list(reversed(path)))
        return results

class EventModel(nn.Module):
    """同一个 BERT 编码器，可选择逐 token 分类或 CRF 序列解码。"""

    def __init__(self, model_path, num_labels, decoder='linear'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels) if decoder == 'crf' else None

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.head(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        if labels is None:
            return logits
        # Linear 忽略标签为 -100 的位置；CRF 由 mask 排除 padding。
        if self.crf:
            return self.crf.loss(logits, labels, attention_mask.bool())
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    def predict(self, ids, mask):
        logits = self(ids, mask)
        if self.crf:
            return self.crf.decode(logits, mask.bool())
        return [row[:int(m.sum())].tolist() for row, m in zip(logits.argmax(-1), mask)]

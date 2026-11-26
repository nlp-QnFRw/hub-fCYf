"""训练数据准备：将事件的字符位置转换为 BIO 标签，再对齐到 BERT token。"""
import torch
from torch.utils.data import Dataset

class EventDataset(Dataset):
    """每条数据包含一句原文和一个事件，__getitem__ 生成模型输入。"""

    def __init__(self, rows, tokenizer, labels, max_length, crf=False):
        self.rows, self.tokenizer, self.ids, self.max_length, self.crf = (rows, tokenizer, {x: i for i, x in enumerate(labels)}, max_length, crf)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        text = row['text']
        event = row['event']
        # 先给每个字符标 O；B 是片段开头，I 是同一片段的后续字符。
        tags = ['O'] * len(text)
        # 先写论元、再写触发词：两者重叠时触发词覆盖论元。
        # 这是单层 BIO 的局限，不能同时表达重叠标签。
        spans = [(a, 'ARG:' + a['role']) for a in event.get('args', [])] + [(event['trigger'], 'TRG:' + event['type'])]
        for span, name in spans:
            s, e = (span['start'], span['end'])
            if 0 <= s < e <= len(text) and text[s:e] == span['text']:
                tags[s] = 'B-' + name
                for i in range(s + 1, e):
                    tags[i] = 'I-' + name
        # tokenizer 的 offset_mapping 告诉我们每个 token 对应原文哪段字符。
        # 超过 max_length 的内容会被截断，评估时应关注这一损失。
        enc = self.tokenizer(text, max_length=self.max_length, truncation=True, return_offsets_mapping=True)
        labels = []
        for s, e in enc.pop('offset_mapping'):
            # 特殊 token 没有字符位置；Linear 忽略它，CRF 把它当 O。
            if s == e:
                labels.append(0 if self.crf else -100)
            else:
                labels.append(self.ids.get(tags[s], 0))
        enc['labels'] = labels
        return enc

def collate(batch):
    """把不同长度的样本补齐为同一个 batch，attention_mask 标记有效位置。"""
    n = max((len(x['input_ids']) for x in batch))
    result = {}
    for key in ('input_ids', 'attention_mask', 'labels'):
        pad = -100 if key == 'labels' and any((-100 in x[key] for x in batch)) else 0
        result[key] = torch.tensor([x[key] + [pad] * (n - len(x[key])) for x in batch], dtype=torch.long)
    return result

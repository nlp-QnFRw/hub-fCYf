"""
bert_data.py -- BERT 的输入管线：WordPiece 分词 → 打包成块 → MLM 动态遮字

和字符级 GPT 的数据管线有三处根本区别：
    1. 分词器不同：BERT 用 WordPiece（中文基本还是一字一 token，但标点和英文会被合并）；
    2. 输入不是「一个长流」而是「[CLS] ... [SEP]」的句子块，要 pad 和 attention_mask；
    3. 训练目标是填空（MLM）而不是预测下一个字，所以需要把 15% 的位置遮掉并生成 labels。

分词用 tokenizers 库直接读 tokenizer.json —— 它和 transformers 的 FastTokenizer 是同一份实现，
所以结果和官方完全一致，但不用 import 那个在这台机器上装坏了的 transformers。
"""

from __future__ import annotations

import os

import torch
from tokenizers import Tokenizer


class BertTok:
    """tokenizer.json 的薄封装，只暴露训练需要的几个接口。"""

    def __init__(self, bert_dir: str):
        path = os.path.join(bert_dir, "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到分词器 {path}")
        self.tok = Tokenizer.from_file(path)
        self.vocab_size = self.tok.get_vocab_size()
        self.mask_id = self._id("[MASK]")
        self.cls_id = self._id("[CLS]")
        self.sep_id = self._id("[SEP]")
        self.pad_id = self._id("[PAD]")

    def _id(self, token: str) -> int:
        i = self.tok.token_to_id(token)
        if i is None:
            raise ValueError(f"词表里没有 {token}")
        return i

    def encode(self, text: str) -> list[int]:
        """不带 [CLS]/[SEP]，返回纯内容 token。"""
        return self.tok.encode(text, add_special_tokens=False).ids

    def tokens(self, text: str) -> list[str]:
        return self.tok.encode(text, add_special_tokens=False).tokens

    def decode(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=True)

    def token(self, i: int) -> str:
        return self.tok.id_to_token(int(i))


def build_blocks(lines: list[str], tok: BertTok, max_len: int) -> list[list[int]]:
    """把句子拼成固定长度的块，块首尾加 [CLS]/[SEP]。

    平均行长只有十几个 token，如果一句一 padding，超过一半的计算都花在 pad 上，
    所以这里按顺序把句子装进块里，装满就换下一块。
    """
    if max_len < 8:
        raise ValueError("max_len 太小，至少要能放下 [CLS] + 若干内容 + [SEP]")
    limit = max_len - 2
    blocks: list[list[int]] = []
    cur: list[int] = []
    for line in lines:
        ids = tok.encode(line)
        if not ids:
            continue
        if len(cur) + len(ids) > limit:          # 装不下就先收口
            blocks.append(cur)
            cur = []
        cur.extend(ids)
        cur.append(tok.sep_id)
        if len(cur) >= limit:
            blocks.append(cur[:limit])
            cur = []
    if cur:
        blocks.append(cur)
    return [[tok.cls_id] + b + [tok.sep_id] for b in blocks]


def pad_blocks(blocks: list[list[int]], max_len: int, pad_id: int):
    """右 padding，返回 (input_ids, attention_mask)，都是 int64 的 (N, max_len)。"""
    ids = torch.full((len(blocks), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(blocks), max_len), dtype=torch.long)
    for i, b in enumerate(blocks):
        b = b[:max_len]
        ids[i, :len(b)] = torch.tensor(b, dtype=torch.long)
        mask[i, :len(b)] = 1
    return ids, mask


def mask_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, tok: BertTok,
               prob: float, generator: torch.Generator):
    """MLM 遮字，返回 (被破坏的输入, labels)。

    labels 只在被选中的位置上等于原字，其余是 -100（cross_entropy 会忽略）。
    选中之后按 BERT 论文的 80/10/10：
        80% 换成 [MASK]，10% 换成随机字，10% 保持不变。
    后两者是为了让模型不能简单地「看到 [MASK] 就猜」，必须依赖上下文。
    """
    vocab_size = tok.vocab_size
    labels = input_ids.clone()

    # 特殊符号和 pad 不遮
    special = (input_ids == tok.cls_id) | (input_ids == tok.sep_id) | (attention_mask == 0)
    can_mask = ~special

    prob_matrix = torch.full(input_ids.shape, prob, dtype=torch.float)
    prob_matrix[~can_mask] = 0.0
    selected = torch.bernoulli(prob_matrix, generator=generator).bool()

    # 兜底：整条都被跳过的话随机补一个，否则这条样本对 loss 没有任何贡献
    empty_rows = (selected & can_mask).sum(1) == 0
    if empty_rows.any():
        for r in torch.nonzero(empty_rows).flatten().tolist():
            valid = torch.nonzero(can_mask[r]).flatten()
            if len(valid):
                selected[r, valid[torch.randint(len(valid), (1,), generator=generator)]] = True

    labels[~selected] = -100

    inputs = input_ids.clone()
    r = torch.rand(input_ids.shape, generator=generator)
    to_mask = selected & (r < 0.8)
    to_random = selected & (r >= 0.8) & (r < 0.9)
    inputs[to_mask] = tok.mask_id
    random_ids = torch.randint(0, vocab_size, input_ids.shape, generator=generator)
    inputs[to_random] = random_ids[to_random]
    return inputs, labels


class BlockBank:
    """块集合 + 按 batch 取样的接口，训练和评估共用。"""

    def __init__(self, blocks, tok: BertTok, max_len: int):
        self.ids, self.mask = pad_blocks(blocks, max_len, tok.pad_id)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return self.ids.size(0)

    def sample(self, batch_size: int, generator: torch.Generator):
        idx = torch.randint(len(self), (batch_size,), generator=generator)
        return self.ids[idx], self.mask[idx]

    def fixed_batches(self, batch_size: int, limit: int | None = None):
        """评估用：固定顺序、不重叠，保证每个阶段算的是同一批数据。"""
        n = len(self) if limit is None else min(limit, len(self))
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            yield self.ids[i:j], self.mask[i:j]

"""数据层：JSONL 读写、BIO 互转、词表、两个训练框架各自的数据集与 collate。

约定
----
- 金标/预测统一用字符 offset：{"start": int, "end": int, "label": str}
- 模型内部统一用 BIO 标签串：["O", "B-x", "I-x", ...]
- 子词对齐只在一处实现（tokenize_for_transformer），训练和推理共用，避免两套逻辑算出两种 offset
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .common import DATA_PROCESSED, read_json, read_jsonl


# ---------------------------------------------------------------- 数据装载
def dataset_dir(name: str) -> Path:
    return DATA_PROCESSED / name


def load_splits(name: str) -> dict[str, list[dict]]:
    d = dataset_dir(name)
    return {s: read_jsonl(d / f"{s}.jsonl") for s in ("train", "dev", "test")}


def load_labels(name: str) -> list[str]:
    return read_json(dataset_dir(name) / "labels.json")["labels"]


def build_label_maps(labels: list[str]) -> tuple[dict[str, int], list[str]]:
    """BIO 标签表：O + 每个类别的 B-/I-。id 顺序固定，训练与推理共用。"""
    tags = ["O"]
    for lab in labels:
        tags += [f"B-{lab}", f"I-{lab}"]
    return {t: i for i, t in enumerate(tags)}, tags


# ---------------------------------------------------------------- BIO
def entities_to_bio(text: str, entities: list[dict]) -> list[str]:
    tags = ["O"] * len(text)
    for e in entities:
        s, t, lab = int(e["start"]), int(e["end"]), e["label"]
        for i in range(s, t):
            tags[i] = f"B-{lab}" if i == s else f"I-{lab}"
    return tags


def bio_to_entities(tags: list[str]) -> list[dict]:
    """BIO → 实体区间；遇到非法 I- 开头就当作 B-（不静默丢实体）。"""
    ents: list[dict] = []
    cur_lab, cur_start = None, None
    for i, tag in enumerate(tags):
        if tag == "O" or not tag:
            if cur_lab is not None:
                ents.append({"start": cur_start, "end": i, "label": cur_lab})
                cur_lab, cur_start = None, None
            continue
        prefix, _, lab = tag.partition("-")
        if prefix == "B" or cur_lab != lab:
            if cur_lab is not None:
                ents.append({"start": cur_start, "end": i, "label": cur_lab})
            cur_lab, cur_start = lab, i
    if cur_lab is not None:
        ents.append({"start": cur_start, "end": len(tags), "label": cur_lab})
    return ents


# ---------------------------------------------------------------- BiLSTM 用
class CharDataset(Dataset):
    """把每条样本变成 (字 id 序列, 标签 id 序列)，按长度截断。"""

    def __init__(self, rows: list[dict], char2id: dict[str, int], tag2id: dict[str, int],
                 max_len: int = 512):
        self.items = []
        for r in rows:
            text = r["text"][:max_len]
            tags = entities_to_bio(r["text"], r["entities"])[:max_len]
            ids = [char2id.get(c, char2id["<unk>"]) for c in text]
            self.items.append((torch.tensor(ids, dtype=torch.long),
                               torch.tensor([tag2id[t] for t in tags], dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_vocab(rows: list[dict], min_freq: int = 1) -> dict[str, int]:
    c = Counter(ch for r in rows for ch in r["text"])
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in sorted(ch for ch, n in c.items() if n >= min_freq):
        vocab[ch] = len(vocab)
    return vocab


def collate_chars(batch, pad_id: int = 0, label_pad: int = -100):
    """动态 padding：本 batch 最长为准，不按全局 max_len 浪费显存。"""
    xs, ys = zip(*batch)
    L = max(len(x) for x in xs)
    X = torch.full((len(xs), L), pad_id, dtype=torch.long)
    Y = torch.full((len(ys), L), label_pad, dtype=torch.long)
    mask = torch.zeros((len(xs), L), dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)] = x
        Y[i, :len(y)] = y
        mask[i, :len(x)] = 1
    return {"input_ids": X, "labels": Y, "attention_mask": mask}


# ---------------------------------------------------------------- Transformer 用
def tokenize_for_transformer(text: str, entities: list[dict], tokenizer, tag2id: dict[str, int],
                             max_len: int = 512) -> dict:
    """用 offset_mapping + word_ids 对齐字符标签与子词。

    坑（写在这里，省得下次再踩）：
    - 中文基本一字一 token，但英文/邮箱/数字会被 WordPiece 拆开；
    - 同一 word 的首个子词用 B-，后续子词用 I-，否则「一个实体被拆成多个」；
    - [CLS]/[SEP]/padding 一律 -100，不参与 loss。
    """
    char_tags = entities_to_bio(text, entities)
    # @doc *** 子词对齐必须 word_ids + offset_mapping 一起用，只看一个都会错
    # 只用其中任何一个都会错：只按 word_ids 拿不到 offset，只按 offset 分不清 B/I 的续接关系
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True,
                    return_attention_mask=True)
    offsets = enc["offset_mapping"]
    word_ids = enc.word_ids()
    labels = [-100] * len(offsets)
    prev_wid, prev_tag = None, None
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid == prev_wid and prev_tag not in (None, "O"):
            _, _, lab = prev_tag.partition("-")
            labels[i] = tag2id[f"I-{lab}"]
        else:
            s = offsets[i][0]
            labels[i] = tag2id[char_tags[s] if s < len(char_tags) else "O"]
        prev_wid = wid
        prev_tag = next(t for t, j in tag2id.items() if j == labels[i])
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": labels, "offset_mapping": offsets}


class HFDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, tag2id: dict[str, int], max_len: int = 512):
        self.rows = rows
        self.feats = [tokenize_for_transformer(r["text"], r["entities"], tokenizer, tag2id, max_len)
                      for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        f = self.feats[i]
        return {k: f[k] for k in ("input_ids", "attention_mask", "labels")}


class PadCollator:
    """只 pad input_ids / attention_mask / labels；offset 只在推理时需要，不进 batch。"""

    def __init__(self, pad_token_id: int, label_pad: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad = label_pad

    def __call__(self, feats: list[dict]) -> dict:
        L = max(len(f["input_ids"]) for f in feats)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in feats:
            n = L - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad_token_id] * n)
            out["attention_mask"].append(f["attention_mask"] + [0] * n)
            out["labels"].append(f["labels"] + [self.label_pad] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}

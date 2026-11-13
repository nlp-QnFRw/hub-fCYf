"""数据管线：jsonl → 样本 → batch。

两种数据集对应两个训练阶段，这也是理解 SFT 的关键对比：

    PretrainDataset  整段语料拼成一条长流，每个位置都算 loss  → 学会「续写」
    SFTDataset       一条样本一个样本，只有回答部分算 loss     → 学会「应答」
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .template import IGNORE_INDEX, Example, encode_example
from .tokenizer import CharTokenizer


# ── 读数据 ────────────────────────────────────────────────────────────────────

def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    """读 instruction/output 格式的 jsonl，顺手丢掉空样本与坏行。"""
    rows: list[dict[str, str]] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            ins = str(obj.get("instruction", "")).strip()
            out = str(obj.get("output", "")).strip()
            if not ins or not out:
                bad += 1
                continue
            rows.append({"instruction": ins, "output": out})
    if bad:
        print(f"  [warn] {Path(path).name}: 跳过 {bad} 条无效样本")
    return rows


# ── SFT 数据集 ────────────────────────────────────────────────────────────────

@dataclass
class SFTStats:
    total: int = 0
    kept: int = 0
    truncated: int = 0
    total_tokens: int = 0
    supervised_tokens: int = 0

    @property
    def supervised_ratio(self) -> float:
        return self.supervised_tokens / max(1, self.total_tokens)


class SFTDataset(Dataset):
    """把 (问题, 回答) 编成带 loss mask 的定长以内序列。"""

    def __init__(
        self,
        records: Iterable[dict[str, str]],
        tok: CharTokenizer,
        max_len: int = 256,
    ) -> None:
        self.tok = tok
        self.max_len = max_len
        self.examples: list[Example] = []
        self.stats = SFTStats()

        for rec in records:
            self.stats.total += 1
            ex = encode_example(tok, rec["instruction"], rec["output"])
            if len(ex.ids) > max_len:
                # 截断时 ids 与 labels 必须同步切，否则 mask 会整体错位一格。
                ex.ids = ex.ids[:max_len]
                ex.labels = ex.labels[:max_len]
                # 截断后属于回答区的长度要重算，不能沿用截断前的值。
                ex.answer_len = sum(1 for v in ex.labels if v != IGNORE_INDEX)
                self.stats.truncated += 1
            self.examples.append(ex)

            self.stats.total_tokens += len(ex.ids)
            self.stats.supervised_tokens += ex.answer_len

        self.stats.kept = len(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def collate_sft(batch: Sequence[Example], pad_id: int) -> dict[str, torch.Tensor]:
    """把变长样本 pad 成矩形。label 的补齐位置填 -100，不能填 pad_id。"""
    max_len = max(len(ex.ids) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, ex in enumerate(batch):
        n = len(ex.ids)
        input_ids[i, :n] = torch.tensor(ex.ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(ex.labels, dtype=torch.long)
        attn[i, :n] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


# ── 预训练数据集 ──────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """模拟「只有语料、没有标注」的预训练：把文本拼成长流后切块，全位置算 loss。"""

    def __init__(
        self,
        records: Iterable[dict[str, str]],
        tok: CharTokenizer,
        block_size: int = 256,
    ) -> None:
        stream: list[int] = []
        for rec in records:
            # 用空格拼接问题与回答，不加任何模板标记 —— 这正是 Base Model 看到的样子。
            text = f"{rec['instruction']} {rec['output']}\n"
            stream.extend(tok.encode(text))

        self.block_size = block_size
        self.chunks: list[list[int]] = []
        for start in range(0, len(stream) - block_size, block_size):
            self.chunks.append(stream[start : start + block_size])
        self.total_tokens = len(stream)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Example:
        ids = self.chunks[idx]
        # 预训练没有 mask：每个 token 既是输入也是标签。
        return Example(ids=list(ids), labels=list(ids), prompt_len=0, answer_len=len(ids))


def collate_pretrain(batch: Sequence[Example], pad_id: int) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([ex.ids for ex in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": None}


# ── DataLoader ────────────────────────────────────────────────────────────────

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    pad_id: int,
    stage: str = "sft",
    num_workers: int = 0,
) -> DataLoader:
    # 用 partial 把 pad_id 提前绑进去，得到一个只接收 batch 的 collate_fn。
    collate = (
        partial(collate_sft, pad_id=pad_id)
        if stage == "sft"
        else partial(collate_pretrain, pad_id=pad_id)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def summarize(records: Iterable[dict[str, str]]) -> dict[str, Any]:
    """给 prepare 脚本用的小统计。"""
    rows = list(records)
    lens = [len(r["instruction"]) + len(r["output"]) for r in rows]
    return {
        "count": len(rows),
        "avg_chars": round(sum(lens) / max(1, len(lens)), 1),
        "max_chars": max(lens) if lens else 0,
    }

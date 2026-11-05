"""
data.py -- 词表、二进制语料、批采样

设计要点（对应旧版本的三个坑）：
    1. 字表外的字不再「直接丢弃」制造伪相邻，而是统一映射到 [UNK]（索引 0）；
    2. 语料编码成 uint16 的 .bin 一次落盘，训练时直接读进内存，不再每轮重新遍历文本；
    3. 评估用固定起点、不重叠的窗口 + 固定顺序，和训练时的随机裁剪区分开，
       保证「不同模型、不同次运行」报出的 PPL 是同一个口径。
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

UNK = "[UNK]"


# ─────────────────────────── 词表 ───────────────────────────

def load_vocab(data_dir: str) -> tuple[dict[str, int], list[str]]:
    with open(os.path.join(data_dir, "vocab.json"), encoding="utf-8") as f:
        itos = json.load(f)
    if itos[0] != UNK:
        raise ValueError(f"词表第 0 项必须是 {UNK}，实际是 {itos[0]!r}")
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    """逐字编码，未登录字 -> [UNK] 的索引 0。"""
    unk_id = stoi[UNK]
    return np.fromiter((stoi.get(c, unk_id) for c in text), dtype=np.uint16, count=len(text))


def decode(ids, itos: list[str]) -> str:
    return "".join(itos[i] for i in ids)


# ─────────────────────────── 二进制语料 ───────────────────────────

def load_tokens(path: str) -> torch.Tensor:
    """读 uint16 的 .bin，转成 int64 的 tensor（Embedding 只接受 long/int）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 prepare_data.py")
    arr = np.fromfile(path, dtype=np.uint16)
    if arr.size == 0:
        raise ValueError(f"{path} 是空的")
    return torch.from_numpy(arr.astype(np.int64))


def save_tokens(ids: np.ndarray, path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ids.astype(np.uint16).tofile(path)
    return int(ids.size)


def load_meta(data_dir: str) -> dict:
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 prepare_data.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────── 批采样 ───────────────────────────

def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """随机裁剪：x 是连续 block_size 个 token，y 是整体右移一位的同一段。

    语言模型的监督信号就是「用前 n 个字预测第 n+1 个字」，
    所以 y 不需要额外标注，同一段文本错一位即可。
    """
    max_start = len(tokens) - block_size - 1
    if max_start < 1:
        raise ValueError(
            f"语料只有 {len(tokens)} 个 token，不够一个 block_size={block_size} 的样本"
        )
    ix = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([tokens[i: i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1: i + block_size + 1] for i in ix])
    return x, y


def eval_windows(tokens: torch.Tensor, block_size: int, batch_size: int):
    """按固定起点、不重叠地切窗口，逐个 batch 产出。

    和 get_batch 的随机裁剪刻意区分开：评估必须可复现，
    而且窗口之间不重叠，避免同一段文本被反复计入。
    """
    n_windows = (len(tokens) - 1) // block_size
    if n_windows < 1:
        raise ValueError(
            f"验证集只有 {len(tokens)} 个 token，不足以切出 block_size={block_size} 的窗口"
        )
    usable = n_windows * block_size
    buf = tokens[: usable + 1]
    for i in range(0, n_windows, batch_size):
        j = min(i + batch_size, n_windows)
        starts = torch.arange(i, j) * block_size
        x = torch.stack([buf[s: s + block_size] for s in starts])
        y = torch.stack([buf[s + 1: s + block_size + 1] for s in starts])
        yield x, y

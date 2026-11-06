"""MyDataset：自回归语言模型的数据集。

把 data/train.txt 或 data/test.txt 中所有行拼在一起（每行以 <eos> 结尾），
形成一条长 token 序列，再按 cfg.hparams.max_len 切成固定长度的块。
每个样本返回 (x, y)，其中 y 是 x 右移一位（next-token prediction）。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

# dataset.py 位于 week04/code/util/ 下，向上两级到 week04/
ROOT = Path(__file__).resolve().parent.parent.parent


class MyDataset(Dataset):
    """字符级自回归语言模型的数据集。

    所有文本拼接成一条长序列，再按 max_len 不重叠切块。
    最后不足 max_len+1 的尾部被丢弃（无法构成一个完整的 (x, y) 对）。
    """

    def __init__(self, cfg, tok, split: str):
        if split not in ("train", "test"):
            raise ValueError(f"split 必须是 'train' 或 'test'，收到 {split!r}")

        self.cfg = cfg
        self.tok = tok
        self.max_len: int = cfg.hparams.max_len

        # 确定数据文件路径
        filename = cfg.hparams.train_filename if split == "train" else cfg.hparams.test_filename
        data_path = ROOT / "data" / filename
        if not data_path.exists():
            raise FileNotFoundError(f"找不到数据文件：{data_path}")

        # 把所有行编码后拼成长序列
        # 每行本身已以 <eos> 结尾，行与行之间不需要额外分隔
        all_ids: list[int] = []
        with data_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                all_ids.extend(tok.encode(line))

        # 不重叠切块：每块长度 = max_len + 1（前 max_len 个作 x，后 max_len 个作 y）
        # 最后不足 max_len+1 的尾部直接丢弃
        stride = self.max_len + 1
        num_blocks = len(all_ids) // stride
        self.blocks: list[list[int]] = [
            all_ids[i * stride : (i + 1) * stride] for i in range(num_blocks)
        ]

        if num_blocks == 0:
            raise RuntimeError(
                f"数据太短（共 {len(all_ids)} 个 token），max_len={self.max_len} 下无法切出任何样本"
            )

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        ids = self.blocks[idx]
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        x = ids_tensor[:-1]   # 长度 = max_len
        y = ids_tensor[1:]    # 长度 = max_len，右移一位
        return x, y

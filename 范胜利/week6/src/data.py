# -*- coding: utf-8 -*-
"""
data.py —— 数据读取、分层抽样、去重、动态 padding 分批

【四个容易踩的坑，这里都处理了】

1) 训练集/验证集有 1,099 条完全重复（11%）
   与训练集整句相同的验证样本会让指标虚高。本脚本不删训练数据（那是数据团队的事），
   但会**额外**报一份「去重验证集」上的指标，确认结论不是靠背答案撑起来的。

2) 定长 padding 白烧算力
   TNEWS 标题平均 20~30 字，最长 145。全部 padding 到 64，等于每步多算一倍多的
   PAD token——而 PAD 位置不参与有效语义，纯属浪费。
   改成「按 batch 内最长序列 padding」：实测每步从 12.4s 降到 5.4s（同一台 CPU）。
   注意：这只改变计算效率，不改变模型看到的内容，两个基座同样处理。

3) 每个 epoch 重新分词
   53k 条 × epoch 数 × 2 个基座，分词开销与一次前向同量级。
   改成训练前一次性分词成 id 列表（不 padding），之后每个 epoch 只取索引。

4) 两个基座必须看到同一批数据
   train_size 抽样用固定 seed + 分层比例，保证 15 类占比与原分布一致，
   且两个基座拿到的是完全相同的子集——否则比的是「数据差异」不是「基座差异」。
"""

import json
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR


def load_splits():
    """读 train/val，返回 list[dict(idx, sentence, label)]"""
    out = []
    for name in ("train.json", "val.json"):
        p = DATA_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"缺少数据文件 {p}，请把 train.json / val.json 放到 data/ 下")
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out[0], out[1]


def dedup_val(train, val):
    """去掉与训练集整句相同的验证样本，返回 (clean_val, n_dup)"""
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    return clean, len(val) - len(clean)


def stratified_subsample(items, n: int, seed: int = 42):
    """
    按类别比例分层抽样 n 条（n<=0 或 n>=总数时返回全部）。

    为什么不用 random.sample：15 类里最小类只有 257 条，均匀抽样会让小类只剩几条，
    甚至一条都抽不到（Macro-F1 直接 NaN）。分层保证每类至少 1 条、且比例与原分布一致。
    """
    if n <= 0 or n >= len(items):
        return list(items)
    by_class = defaultdict(list)
    for d in items:
        by_class[d["label"]].append(d)

    rng = random.Random(seed)
    total = len(items)
    picked, shortfall = [], 0
    for lab in sorted(by_class):
        group = by_class[lab]
        k = max(1, round(n * len(group) / total))
        if k > len(group):
            k = len(group)
        picked.extend(rng.sample(group, k))
        shortfall += max(0, round(n * len(group) / total) - k)

    # 补齐/裁掉差额，避免因为四舍五入少几条。
    # 用 idx 集合做排除：直接 `d not in picked` 是对 53k 条 × 2k 条列表的成员测试，
    # 复杂度 O(n·k)，在 5 万条数据上要跑几十秒，肉眼看着像「卡住了」。
    chosen_idx = {d["idx"] for d in picked}
    pool = [d for d in items if d["idx"] not in chosen_idx]
    rng.shuffle(pool)
    if shortfall > 0:
        picked.extend(pool[:shortfall])
    picked = picked[:n] if len(picked) > n else picked
    rng.shuffle(picked)
    return picked


class TnewsDataset(Dataset):
    """每个样本返回变长 ids（不 padding），padding 交给 collate 按 batch 做"""

    def __init__(self, token_ids, labels):
        self.token_ids, self.labels = token_ids, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"ids": torch.tensor(self.token_ids[i], dtype=torch.long),
                "y": torch.tensor(self.labels[i], dtype=torch.long)}


def make_collate(pad_id: int = 0):
    """
    把 batch 内变长序列 pad 到该 batch 的最长值。
    attention_mask 同步生成（1=有效，0=PAD）——池化和 attention 都靠它屏蔽 PAD。
    """

    def collate(batch):
        lens = [len(b["ids"]) for b in batch]
        L = max(lens)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["ids"])
            ids[i, :n] = b["ids"]
            mask[i, :n] = 1
        return ids, mask, torch.stack([b["y"] for b in batch])

    return collate


def tokenize(texts, tok, max_length: int):
    """只截断不 padding：返回 list[list[int]]，省内存也省算力"""
    enc = tok(list(texts), truncation=True, max_length=max_length, padding=False)
    return enc["input_ids"]


def build_loaders(train, val, tok, max_length: int, batch_size: int, seed: int = 42, num_workers: int = 0):
    """
    返回 (train_loader, val_loader)。
    训练集 shuffle 用独立 generator + 固定 seed，保证两个基座的 batch 顺序一致。
    """
    tr_ids = tokenize([d["sentence"] for d in train], tok, max_length)
    tr_y = [d["label"] for d in train]
    va_ids = tokenize([d["sentence"] for d in val], tok, max_length)
    va_y = [d["label"] for d in val]

    collate = make_collate(pad_id=tok.pad_token_id or 0)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(TnewsDataset(tr_ids, tr_y), batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(TnewsDataset(va_ids, va_y), batch_size=max(batch_size, 128),
                            shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, val_loader


def class_counts(items) -> dict:
    """类别样本量，用于报告里解释「为什么不平衡要看 Macro-F1」"""
    return dict(sorted(Counter(d["label"] for d in items).items()))


def build_eval_loader(items, tok, max_length: int, batch_size: int = 128, num_workers: int = 0):
    """单独给「去重验证集」这类临时评测集建 loader（不 shuffle，顺序稳定）"""
    ids = tokenize([d["sentence"] for d in items], tok, max_length)
    ys = [d["label"] for d in items]
    return DataLoader(TnewsDataset(ids, ys), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=make_collate(pad_id=tok.pad_token_id or 0))

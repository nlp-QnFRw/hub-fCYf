#region 0. 模块说明
"""
数据读取、分层抽样、Dataset 与 collate 的**唯一实现**。

设计重点（每一条都是踩过坑换来的）：
  1. **全局唯一 id**：每条样本带 all.jsonl 里的行号 id。所有方法的预测都按 id 对齐，
     而不是按"子集内下标"。前一个项目就是因为缺这个，三个模型的预测交集算成了 0。
  2. **嵌套子集**：5k ⊂ 20k ⊂ 50k ⊂ 100k，保证数据量曲线单调可比（各自独立抽样会出现
     "5k 比 20k 还好"的噪声）。
  3. **离线预 tokenize**：Windows 无 fork，num_workers>0 走 spawn 会重新 import torch
     并复制 tokenizer；长文本在线 tokenize 在首个 epoch 能卡几分钟。预 tokenize 后
     100k×256 的 int32 张量只有 ~100MB，num_workers=0 足够（瓶颈在 GPU，不在 IO）。
  4. **collate 独立成函数**（不做闭包），由外层 lambda 固定 pad_id 后交给 DataLoader。

使用方式：
  from common.data import load_split, stratified_sample, BertDataset, build_dataloaders
  recs = load_split("train_100k")
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from common.labels import LABEL2ID, LABELS
from common.paths import ALL_JSONL, split_file
from common.text import build_text

#endregion

#region 2. 读取 jsonl
def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """逐行读 jsonl（生成器，避免一次性把 74 万条读进内存）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_split(name: str, limit: int | None = None) -> list[dict]:
    """读取一个切分文件，返回 list[dict]。

    参数：
      name  : 切分名（train_100k / val_10k / test_nat_10k / test_bal / sub_5k …）
      limit : 只取前 N 条（冒烟用），None 表示全量
    返回：
      [{"id": int, "title": str, "content": str, "label": str, "label_id": int}, ...]
    """
    p = split_file(name)
    if not p.exists():
        raise FileNotFoundError(f"切分文件不存在：{p}\n请先运行 python scripts/02_splits.py")
    recs: list[dict] = []
    for i, r in enumerate(iter_jsonl(p)):
        if limit is not None and i >= limit:
            break
        r["label_id"] = LABEL2ID[r["label"]]  # 统一在此补 label_id，下游不再各自映射
        recs.append(r)
    return recs


def label_counts(recs: Sequence[dict]) -> dict[str, int]:
    """统计各类样本数（按 LABELS 顺序返回，缺失补 0）。"""
    c = Counter(r["label"] for r in recs)
    return {name: c.get(name, 0) for name in LABELS}
#endregion

#region 3. 分层抽样（切分脚本与子集生成共用同一份实现）
def stratified_sample(recs: Sequence[dict], n: int, seed: int = 42,
                      min_per_class: int = 0) -> list[dict]:
    """按类别分层抽 n 条；每类至少 min_per_class 条（不足则全取）。

    参数：
      recs          : 候选池
      n             : 目标总数
      seed          : 固定随机种子（保证每次抽到同一批）
      min_per_class : 每类保底数（test_bal 用它保证长尾类的 F1 有统计意义）
    返回：
      抽样后的 list，顺序已按 (label, id) 排序以便复现
    """
    rng = random.Random(seed)
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_cls[r["label"]].append(r)
    for v in by_cls.values():
        rng.shuffle(v)

    total = sum(len(v) for v in by_cls.values())
    if total == 0:
        return []

    take: dict[str, int] = {}
    if min_per_class > 0:
        # 先满足保底，剩余名额按原分布比例分配
        base = {k: min(min_per_class, len(v)) for k, v in by_cls.items()}
        left = max(0, n - sum(base.values()))
        for k, v in by_cls.items():
            take[k] = base[k] + int(round(left * len(v) / total))
    else:
        for k, v in by_cls.items():
            take[k] = int(round(n * len(v) / total))

    out: list[dict] = []
    for k, v in by_cls.items():
        out.extend(v[: take[k]])
    # 名额取整可能多出/少几条，用整体 shuffle 后截断到 n 兜底
    rng.shuffle(out)
    out = out[:n]
    out.sort(key=lambda r: (r["label"], r["id"]))
    return out


def nested_subset(recs: Sequence[dict], n: int, seed: int = 42) -> list[dict]:
    """从已抽好的大集合里再抽 n 条，得到严格嵌套的子集（5k ⊂ 20k ⊂ 50k ⊂ 100k）。

    实现上就是"对同一个大集合再做一次分层抽样"，因为分层抽样本身是确定性的，
    对 100k 抽 5k 与对 100k 抽 20k 的结果天然满足 5k ⊆ 20k 吗？——不保证。
    所以这里改成：先对大集合按 id 排序，再用固定 seed 抽，且小集合的抽样池是大集合
    本身（不是原始池）。为保证严格嵌套，我们对每个类按 id 排序后取前 k 条：
    这样任何更小的 n 都是更大 n 的前缀，天然嵌套。
    """
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_cls[r["label"]].append(r)
    for v in by_cls.values():
        v.sort(key=lambda r: r["id"])

    total = sum(len(v) for v in by_cls.values())
    out: list[dict] = []
    for k, v in by_cls.items():
        k_n = int(round(n * len(v) / total))
        out.extend(v[: max(1, k_n)])
    out.sort(key=lambda r: (r["label"], r["id"]))
    return out[:n]
#endregion

#region 4. BERT 数据集（预 tokenize）
class BertDataset(Dataset):
    """把已 tokenize 的 tensor 包成 Dataset。

    为什么先在外部 tokenize 再传进来：
      Dataset 里只做 `tensor[i]` 切片，`__getitem__` 是纯内存操作，
      因此 num_workers=0 也不会成为瓶颈（实测 BERT 训练 121 samples/s，
      瓶颈在 GPU；tokenize 单条 <2ms）。
    """

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 labels: torch.Tensor | None = None) -> None:
        assert input_ids.shape == attention_mask.shape
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
        }
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def collate_fn(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """把变长样本 pad 成等长 batch。

    三条 pad 规则（任何任务都一样，写死在这里避免各脚本写错）：
      input_ids      → pad_id
      attention_mask → 0
      labels         → -100（CrossEntropy 的默认 ignore_index，不参与 loss）
    本函数是独立函数而非闭包，外层用 `lambda b: collate_fn(b, tok.pad_token_id)`。
    """
    max_len = max(x["input_ids"].size(0) for x in batch)
    n = len(batch)
    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    am = torch.zeros((n, max_len), dtype=torch.long)
    out: dict[str, torch.Tensor] = {}
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        ids[i, :L] = x["input_ids"]
        am[i, :L] = x["attention_mask"]
    out["input_ids"] = ids
    out["attention_mask"] = am
    if "labels" in batch[0]:
        lab = torch.full((n,), -100, dtype=torch.long)
        for i, x in enumerate(batch):
            lab[i] = x["labels"]
        out["labels"] = lab
    return out


def tokenize_records(recs: Sequence[dict], tokenizer, max_len: int,
                     mode: str = "head_tail", with_labels: bool = True,
                     show_progress: bool = False) -> BertDataset:
    """批量 tokenize（这一步在 CPU 上一次性做完，之后训练不再碰 tokenizer）。

    参数：
      recs        : load_split 返回的记录
      tokenizer   : transformers tokenizer
      max_len     : 最大 token 数
      mode        : 截断策略（common.text.TRUNCATIONS）
      with_labels : 测试集无标签时传 False
    返回：
      BertDataset
    """
    texts = [build_text(r, mode=mode, max_len=max_len) for r in recs]
    it = range(0, len(texts), 512)
    if show_progress:
        try:
            from tqdm import tqdm

            it = tqdm(it, desc="tokenize")
        except ImportError:
            pass

    ids_buf: list[torch.Tensor] = []
    am_buf: list[torch.Tensor] = []
    for i in it:
        enc = tokenizer(texts[i: i + 512], max_length=max_len, truncation=True,
                        padding=False, return_attention_mask=True)
        ids_buf.extend(torch.tensor(x, dtype=torch.long) for x in enc["input_ids"])
        am_buf.extend(torch.tensor(x, dtype=torch.long) for x in enc["attention_mask"])

    input_ids = torch.nn.utils.rnn.pad_sequence(ids_buf, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(am_buf, batch_first=True, padding_value=0)
    labels = None
    if with_labels:
        labels = torch.tensor([r["label_id"] for r in recs], dtype=torch.long)
    return BertDataset(input_ids, attention_mask, labels)


def build_dataloaders(ds: BertDataset, pad_id: int, batch_size: int,
                      shuffle: bool, seed: int = 42) -> DataLoader:
    """统一构造 DataLoader（num_workers=0，见模块说明第 3 条）。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id),
        generator=g if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )
#endregion

#region 5. 自检
if __name__ == "__main__":
    if not ALL_JSONL.exists():
        print(f"尚无 {ALL_JSONL}，跳过自检（先跑 scripts/01_to_jsonl.py）")
    else:
        recs = list(iter_jsonl(ALL_JSONL))
        print(f"all.jsonl 共 {len(recs)} 条；类别分布前 5：")
        for k, v in sorted(label_counts(recs).items(), key=lambda x: -x[1])[:5]:
            print(f"  {k}: {v}")
        sub = stratified_sample(recs[:20000], 1000, min_per_class=50)
        print(f"分层抽样 1000 条，覆盖 {len(set(r['label'] for r in sub))} 类")
#endregion

"""
数据读取、去重与分层切分（THUCNews）

与原项目（TNEWS）的差别：
  1. 原始文件是「label<TAB>正文」的纯文本，不是 jsonl；解析要自己写。
  2. 正文比标题长得多，所以此处引入 **截断策略**：默认保留前 max_len 字
     （新闻的标题与导语都在开头，这是长文本分类最常用的低成本做法）。
  3. 依然保留「先查泄漏再评测」的习惯：同一篇文章在语料里会被多家媒体转载，
     不去重的话验证集里混进训练样本，指标会虚高。

用法：
  from common.data import build_splits, load_splits
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from labels import CLASSES, CLASS2ID

WORK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORK_ROOT / "data"
RAW_FILE = DATA_DIR / "cnews.train.txt"


def _clean(text: str) -> str:
    """把正文压成单行：去掉全角空格、制表符、连续空白"""
    text = text.replace("\u3000", " ").replace("\t", " ")
    return " ".join(text.split())


def _truncate(text: str, max_len: int, mode: str = "head") -> str:
    """
    长文本截断。两种口径：
      head      —— 只留开头（标题 + 导语），最常用
      head_tail —— 前后各留一半，适合「结论在文末」的任务（如财报、判决书）
    """
    if len(text) <= max_len:
        return text
    if mode == "head":
        return text[:max_len]
    half = max_len // 2
    return text[:half] + " " + text[-half:]


def load_raw(min_chars: int = 10, max_chars: int = 20000):
    """
    解析 cnews.train.txt → [(label_id, text), ...]

    过滤太短（<10 字，几乎必然是脏数据）和超长（>2 万字，评测时也用不到）的样本。
    """
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"缺少语料 {RAW_FILE}，先跑 scripts/fetch_data.py")

    rows = []
    bad_label = Counter()
    with open(RAW_FILE, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\t" not in line:
                continue
            label, content = line.split("\t", 1)
            label, content = label.strip(), _clean(content)
            if label not in CLASS2ID:
                bad_label[label] += 1
                continue
            if not (min_chars <= len(content) <= max_chars):
                continue
            rows.append((CLASS2ID[label], content))
    if bad_label:
        print(f"[warn] 忽略了 {sum(bad_label.values())} 条未知标签：{dict(bad_label)}")
    return rows


def build_splits(n_train: int = 40000, n_val: int = 8000, seed: int = 42,
                 max_len: int = 256, trunc: str = "head", use_cache: bool = True):
    """
    分层切分：每类等量抽训练集，验证集从**剩下的**样本里抽，两者天然不重叠。

    为什么先全局去重再切分：
      语料里同一篇文章被多家媒体转载的情况很多。如果先切分再去重，
      验证集会被「剩下」的近似重复样本占满；先全局去重，才能保证两边都干净。
    """
    cache = DATA_DIR / f"splits_{n_train}_{n_val}_{max_len}_{trunc}_{seed}.json"
    if use_cache and cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["train"], d["val"], d["meta"]

    raw = load_raw()
    n_raw = len(raw)

    # 全局去重：按清洗后的整篇文本判重（最朴素的泄漏形态）
    seen, uniq = set(), []
    for y, t in raw:
        if t in seen:
            continue
        seen.add(t)
        uniq.append((y, t))
    n_dup = n_raw - len(uniq)

    by_cls = defaultdict(list)
    for y, t in uniq:
        by_cls[y].append(t)

    n_cls = len(CLASSES)
    per_train = max(1, n_train // n_cls)
    per_val = max(1, n_val // n_cls)
    if per_train + per_val > min(len(v) for v in by_cls.values()):
        raise ValueError(
            f"每类样本不足：需要 {per_train + per_val} 条/类，"
            f"最少的一类只有 {min(len(v) for v in by_cls.values())} 条"
        )

    rng = random.Random(seed)
    train, val = [], []
    for cid in range(n_cls):
        pool = by_cls[cid][:]
        rng.shuffle(pool)
        for t in pool[:per_train]:
            train.append({"text": _truncate(t, max_len, trunc), "label": cid,
                          "label_name": CLASSES[cid]})
        for t in pool[per_train:per_train + per_val]:
            val.append({"text": _truncate(t, max_len, trunc), "label": cid,
                        "label_name": CLASSES[cid]})
    rng.shuffle(train)
    rng.shuffle(val)

    meta = {
        "dataset": "THUCNews (cnews 10 类)",
        "n_raw": n_raw,
        "n_unique": len(uniq),
        "n_dup_global": n_dup,
        "dup_rate": round(n_dup / n_raw, 4) if n_raw else 0.0,
        "n_train": len(train),
        "n_val": len(val),
        "per_class_train": per_train,
        "per_class_val": per_val,
        "max_len": max_len,
        "trunc": trunc,
        "seed": seed,
        "class_counts_raw": {CLASSES[c]: len(v) for c, v in sorted(by_cls.items())},
    }
    d = {"train": train, "val": val, "meta": meta}
    if use_cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return train, val, meta


def load_splits(n_train: int = 40000, n_val: int = 8000, **kw):
    """最常用入口：返回 (train, val, meta)"""
    return build_splits(n_train=n_train, n_val=n_val, **kw)


def subset(items, n, seed=42):
    """固定 seed 抽样：所有方法必须评测同一批样本"""
    if n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)

# -*- coding: utf-8 -*-
"""
构造「类型留出」数据集变体：训练集删掉 3 个事件类型，评测集只保留这 3 个类型。

为什么这么设计：
  本周作业要求「换不同数据集」。除了换数据规模（全量 5747 vs 小样本 800），
  最能说明问题的是**换数据分布**：把 3 个类型（质押 / 高管变动 / 被约谈）从训练集里
  整类删掉，再在这 3 类上评测。
    - 判别式模型（BERT/BiLSTM）在训练里没见过这些标签 → 预期接近 0
    - LLM zero-shot 靠 prompt 里的 schema 描述，理应仍有产出 → 对比最直观

实现要点：
  1. schema 仍保留全部 13 类（标签空间不缩小），否则「没学会」会被"标签根本不存在"掩盖；
  2. 训练集只删**这 3 类**的样本（5008 句），评测集只留这 3 类（116 句）；
  3. 输出目录结构与 data/processed 完全一致，可直接被 --data_dir / --data 复用。

用法：
  python 代码/build_holdout.py [--project <课程项目 code 目录>] [--out <输出目录>]
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

HOLDOUT_TYPES = ["质押", "高管变动", "被约谈"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump_jsonl(rows: list[dict], p: Path) -> None:
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(here.parent.parent / "week7-信息抽取" / "code"))
    ap.add_argument("--out", default=str(here / "data_holdout"))
    args = ap.parse_args()

    src = Path(args.project) / "data" / "processed"
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(src / "train_extract.jsonl")
    dev = load_jsonl(src / "dev_extract.jsonl")

    train_keep = [r for r in train if r["event"]["type"] not in HOLDOUT_TYPES]
    dev_hold = [r for r in dev if r["event"]["type"] in HOLDOUT_TYPES]

    dump_jsonl(train_keep, dst / "train_extract.jsonl")
    dump_jsonl(dev_hold, dst / "dev_extract.jsonl")
    shutil.copy(src / "event_schema.json", dst / "event_schema.json")

    stats = {
        "holdout_types": HOLDOUT_TYPES,
        "train_rows": len(train_keep),
        "dev_rows": len(dev_hold),
        "dev_type_counts": dict(Counter(r["event"]["type"] for r in dev_hold)),
        "note": "训练集整类删掉 holdout_types；评测集只保留这些类型；schema 仍为全部 13 类",
    }
    (dst / "holdout_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

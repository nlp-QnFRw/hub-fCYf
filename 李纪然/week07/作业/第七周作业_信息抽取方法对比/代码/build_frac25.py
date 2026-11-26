# -*- coding: utf-8 -*-
"""
构造「25% 训练数据」数据集变体：训练集随机抽 1/4（seed=42），评测集保持全量 dev 930。

用途：在**同一评测集**上比较"同样的配方、更少的数据"，回答"数据量对分层方法的影响"。
与 data/sample（学生包 800/300）不同，这里的评测集仍是完整 dev 930，指标可以和全量实验直接比。

用法：
  python 代码/build_frac25.py [--project <课程项目 code 目录>] [--out <输出目录>] [--frac 0.25]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(here.parent.parent / "week7-信息抽取" / "code"))
    ap.add_argument("--out", default=str(here / "data_frac25"))
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.project) / "data" / "processed"
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in (src / "train_extract.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(args.seed).shuffle(rows)
    keep = rows[: int(len(rows) * args.frac)]

    with open(dst / "train_extract.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("dev_extract.jsonl", "event_schema.json"):
        shutil.copy(src / name, dst / name)

    stats = {
        "frac": args.frac,
        "seed": args.seed,
        "train_rows": len(keep),
        "dev_rows": sum(1 for _ in open(dst / "dev_extract.jsonl", encoding="utf-8")),
        "note": "训练集按 seed=42 随机抽 25%；评测集仍是完整 dev 930，便于和全量实验直接比",
    }
    (dst / "frac_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

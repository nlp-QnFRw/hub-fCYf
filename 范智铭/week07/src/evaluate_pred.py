"""对任意 pred.jsonl 计算统一指标（所有方法同一口径）。

    python -m src.evaluate_pred --dataset cluener --pred outputs/cluener/bert/pred.jsonl \
        --out outputs/cluener/bert/metrics.json

--ids 可以指定只评某个 id 子集（LLM 抽样后，让本地模型在同一子集上另算一份，保证可比）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import get_logger, read_json, read_jsonl, write_json
from .data import load_labels, load_splits
from .metrics import evaluate, format_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--ids", default=None, help="JSON 文件，内容为 id 列表")
    args = ap.parse_args()

    splits = load_splits(args.dataset)
    labels = load_labels(args.dataset)
    pred = read_jsonl(args.pred)
    ids = set(read_json(args.ids)) if args.ids else None

    result = evaluate(splits["test"], pred, labels, subset_ids=ids)
    hard = [r for r in splits["test"] if r.get("meta", {}).get("hard")]
    if hard:
        hard_ids = {r["id"] for r in hard} & (ids if ids is not None else {r["id"] for r in splits["test"]})
        if hard_ids:
            result["hard_subset"] = evaluate(splits["test"], pred, labels, subset_ids=hard_ids)

    out = Path(args.out)
    metrics = {"dataset": args.dataset, "method": args.method or out.parent.name,
               "pred": str(args.pred), "subset": sorted(ids) if ids else None, "test": result}
    write_json(out, metrics)
    log = get_logger("eval")
    log.info("%s / %s\n%s", args.dataset, metrics["method"], format_table(result))


if __name__ == "__main__":
    main()

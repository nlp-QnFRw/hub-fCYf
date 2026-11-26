"""篇章级的便宜闸门实验：句级模型会「见句就抽」，用论元数量阈值压掉低质量事件。

这不是新模型，只是事后分析：把 macbert_crf_doc 的预测按 #args 阈值过滤，
看篇章事件集合 F1 怎么变。阈值若来自测试集调参，只能当诊断，不能当上线指标——README 里如实标注。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_ROOT, METRIC_DIR, PRED_DIR  # noqa: E402
from common.bridge import load_jsonl, save_jsonl  # noqa: E402
from common.eval_local import doc_event_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="macbert_crf_doc")
    ap.add_argument("--thresholds", default="1,2,3")
    args = ap.parse_args()
    rows = load_jsonl(str(PRED_DIR / f"pred_{args.src}.jsonl"))
    index = DATA_ROOT / "doc" / "doc_index.json"
    base = json.loads((METRIC_DIR / f"{args.src}.json").read_text(encoding="utf-8"))
    print(f"基线（无闸门）doc F1={base['doc_level']['f1']:.4f}")
    out = []
    for th in [int(x) for x in args.thresholds.split(",")]:
        filtered = []
        for r in rows:
            evs = [e for e in (r.get("events") or []) if len(e.get("args") or []) >= th]
            filtered.append({"id": r["id"], "sent_id": r.get("sent_id", r["id"]),
                             "doc_id": r.get("doc_id"), "events": evs})
        name = f"{args.src}_gate{th}"
        save_jsonl(filtered, PRED_DIR / f"pred_{name}.jsonl")
        d = doc_event_f1(PRED_DIR / f"pred_{name}.jsonl", index)
        out.append({"min_args": th, **d})
        print(f"min_args>={th}: doc P={d['precision']:.4f} R={d['recall']:.4f} "
              f"F1={d['f1']:.4f} tp={d['tp']} fp={d['fp']}")
    with open(METRIC_DIR / f"{args.src}_gate_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": base["doc_level"], "thresholds": out,
                   "note": "阈值在测试集上扫过，属事后诊断，不能当无偏上线指标"},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

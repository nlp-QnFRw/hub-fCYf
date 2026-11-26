"""work 项目的落盘评测：沿用 week7 的唯一评测函数，输出全部写到 work/outputs。

额外提供 doc-level 事件集合 F1——week7 的评测只到句子级，篇章变体必须补这一层，
否则「切句→抽取→篇章合并」的收益根本看不出来。
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from config import METRIC_DIR, PRED_DIR, SEED
from common.bridge import evaluate, load_jsonl, save_jsonl


def sample_rows(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """所有生成式方法共用同一批抽样 id，保证横向可比。"""
    rows = sorted(rows, key=lambda r: str(r.get("id", "")))
    if n >= len(rows):
        return rows
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def event_key(ev: dict) -> tuple:
    trigger = ev.get("trigger") or {}
    trig = trigger.get("text", "") if isinstance(trigger, dict) else str(trigger)
    args = tuple(sorted((a.get("role", ""), a.get("text", "")) for a in (ev.get("args") or [])))
    return (ev.get("type", ""), trig, args)


def doc_event_f1(pred_path, index_path) -> dict:
    """整篇公告的事件集合 F1（类型 + 触发词文本 + 论元集合全等才算对）。"""
    index = load_jsonl(str(index_path)) if str(index_path).endswith(".jsonl") else json.loads(
        Path(index_path).read_text(encoding="utf-8"))
    preds = defaultdict(list)
    for r in load_jsonl(str(pred_path)):
        doc_id = r.get("doc_id") or str(r.get("id", "")).split("__s")[0]
        for ev in r.get("events") or []:
            preds[doc_id].append(event_key(ev))

    tp = fp = fn = 0
    n_docs_with_gold = 0
    for d in index:
        gold = {event_key(e) for e in d["gold_events"]}
        if not gold:
            continue
        n_docs_with_gold += 1
        p = set(preds.get(d["doc_id"], []))
        tp += len(gold & p)
        fp += len(p - gold)
        fn += len(gold - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "n_docs_with_gold": n_docs_with_gold}


def run_eval(name: str, pred_rows: list[dict], gold_path, meta: dict | None = None,
             doc_index=None, verbose: bool = True) -> dict:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PRED_DIR / f"pred_{name}.jsonl"
    save_jsonl(pred_rows, pred_path)
    res = evaluate(load_jsonl(str(gold_path)), pred_rows, verbose=verbose)
    res["method"] = name
    res["gold"] = str(gold_path)
    res["pred"] = str(pred_path)
    if meta:
        res["meta"] = meta
    if doc_index is not None:
        res["doc_level"] = doc_event_f1(pred_path, doc_index)
        if verbose:
            d = res["doc_level"]
            print(f"  doc_level    P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f} "
                  f"(tp={d['tp']} fp={d['fp']} fn={d['fn']})")
    with open(METRIC_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def record_skip(name: str, reason: str) -> None:
    """没跑就明确写「未跑」，绝不用估算值占位（沿用 week7 的诚实口径）。"""
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRIC_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"method": name, "status": "NOT_RUN", "reason": reason},
                  f, ensure_ascii=False, indent=2)
    print(f"[skip] {name}: {reason}")

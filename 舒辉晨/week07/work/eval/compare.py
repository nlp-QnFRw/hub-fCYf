"""把各路结果投影到同 60 条，再汇总成一张可复现的对比表。

判别式跑全量 dev（930 / 75），生成式只跑 60 条——直接把两组数字并排是错的。
这里把判别式预测按 eval_ids 投影到同一批 60 条，用同一个 evaluate() 重算。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import METRIC_DIR, OUT, PRED_DIR  # noqa: E402
from common.bridge import evaluate, load_jsonl  # noqa: E402

VARIANTS = ["sent", "heldout"]
FULL_METHODS = {
    "sent": ["rule_sent", "rbt3_crf_sent", "macbert_crf_sent"],
    "heldout": ["rule_heldout", "rbt3_crf_heldout", "macbert_crf_heldout"],
}


def project(variant: str, method: str) -> dict | None:
    gold_path = OUT / f"gold_{variant}_n60.jsonl"
    pred_path = PRED_DIR / f"pred_{method}.jsonl"
    if not (gold_path.exists() and pred_path.exists()):
        return None
    # 必须用 (id, sent_id) 二元组：一篇公告可能有多句，id 会重复（week7 的 row_key 口径）
    key = lambda r: (str(r.get("id", "")), str(r.get("sent_id", "")))
    ids = {key(r) for r in load_jsonl(str(gold_path))}
    preds = [r for r in load_jsonl(str(pred_path)) if key(r) in ids]
    if len(preds) != len(ids):
        print(f"[warn] {method} 投影只命中 {len(preds)}/{len(ids)}，跳过")
        return None
    res = evaluate(load_jsonl(str(gold_path)), preds, verbose=False)
    res["method"] = f"{method}_n60"
    res["meta"] = {"projection_from": str(pred_path), "gold_subset": str(gold_path),
                   "n": len(preds)}
    with open(METRIC_DIR / f"{method}_n60.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def f1(res: dict, key: str) -> float:
    return res.get(key, {}).get("f1", float("nan"))


def main():
    for variant, methods in FULL_METHODS.items():
        for m in methods:
            r = project(variant, m)
            if r:
                print(f"{m}_n60: argument F1={r['argument']['f1']:.4f} "
                      f"event F1={r['event_exact']['f1']:.4f}")

    rows = []
    for path in sorted(METRIC_DIR.glob("*.json")):
        if path.stem.endswith("_analysis") or path.stem == "summary":
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("status") == "NOT_RUN":
            rows.append({"name": path.stem, "status": "NOT_RUN", "reason": d["reason"]})
            continue
        meta = d.get("meta") or {}
        rows.append({
            "name": path.stem,
            "status": "OK",
            "n": d.get("num_samples"),
            "trigger": f1(d, "trigger"), "argument": f1(d, "argument"),
            "event": f1(d, "event_exact"), "type_macro": d.get("type_macro_f1"),
            "doc_f1": (d.get("doc_level") or {}).get("f1"),
            "train_s": meta.get("train_s"), "ms": meta.get("ms_per_sentence"),
            "params_m": meta.get("total_params_m"),
            "gate": (d.get("parse_stats") or {}).get("json_ok"),
        })

    lines = [
        "# work 实验汇总（自动生成，细节与结论见 README.md）", "",
        "| 方法 | 样本数 | trigger F1 | argument F1 | event F1 | type macro F1 | 篇章事件 F1 | 训练(s) | 延迟(ms/句) | 参数(M) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["status"] != "OK":
            lines.append(f"| {r['name']} | — | — | — | — | — | — | — | — | NOT_RUN |")
            continue
        fmt = lambda x, p=4: "—" if x is None else f"{x:.{p}f}"
        lines.append(
            f"| {r['name']} | {r['n']} | {fmt(r['trigger'])} | {fmt(r['argument'])} | "
            f"{fmt(r['event'])} | {fmt(r['type_macro'])} | {fmt(r['doc_f1'])} | "
            f"{fmt(r['train_s'], 0)} | {fmt(r['ms'], 2)} | {fmt(r['params_m'], 1)} |")
    lines += ["", "> 投影组（`_n60`）与生成式跑在同一批 60 条；其余为各自完整 dev。",
              "> 跨变体的 `sent` / `heldout` / `doc` 不是同一个测试集，不能直接比数值高低。", ""]
    (METRIC_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("written:", METRIC_DIR / "summary.md")


if __name__ == "__main__":
    main()

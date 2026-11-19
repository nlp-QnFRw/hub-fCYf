"""
统一汇总：把各路结果拉到一张表，并做口径检查

口径检查不是形式主义：不同脚本如果用了不同 n_eval 或不同 seed，
「差 2 个点」可能只是抽样波动。这里先比 meta，不一致直接报警。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

SOURCES = [
    ("规则·类名匹配", "rule_pred.json", ("modes", "naive")),
    ("规则·关键词表", "rule_pred.json", ("modes", "keywords")),
    ("TF-IDF + SVM", "tfidf_pred.json", None),
    ("TextCNN（从零）", "textcnn_pred.json", None),
    ("rbt3 微调", "rbt3_pred.json", None),
    ("MacBERT 微调", "macbert_pred.json", None),
    ("Qwen3-0.6B 零样本", "zero_shot_pred.json", None),
]


def pick(d, key):
    if key is None:
        return d
    for k in key:
        d = d[k]
    return d


def main():
    rows, metas = [], []
    for name, fname, key in SOURCES:
        p = OUT / fname
        if not p.exists():
            rows.append({"method": name, "acc": "-", "f1": "-", "n": "-", "note": "未运行"})
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        node = pick(data, key)
        r, m = node.get("report", node), data.get("meta", {})
        metas.append((name, m))
        rows.append({
            "method": name,
            "acc": r.get("acc_incl_unparsed", r.get("acc")),
            "f1": r.get("macro_f1"),
            "n": m.get("n_eval", r.get("n")),
            "unparsed": r.get("unparsed", 0),
            "params_m": r.get("params_m") or r.get("params"),
            "cost_s": r.get("train_seconds") or r.get("elapsed_s") or r.get("fit_seconds"),
            "note": "",
        })

    sizes = {m.get("n_eval") for _, m in metas if m.get("n_eval")}
    if len(sizes) > 1:
        print(f"[warn] 各路评测样本量不一致：{sizes}，数字不可直接比较")

    head = "| 方法 | 准确率 | Macro-F1 | 评测条数 | 参数量 | 训练耗时 | 解析失败 |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        lines.append(f"| {r['method']} | {r['acc']} | {r['f1']} | {r['n']} | "
                     f"{r.get('params_m') or '-'} | {r.get('cost_s') or '-'} | {r.get('unparsed') or 0} |")
    table = "\n".join(lines)

    print("\n" + table + "\n")
    (OUT / "summary.md").write_text(table + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT/'summary.md'}")


if __name__ == "__main__":
    main()

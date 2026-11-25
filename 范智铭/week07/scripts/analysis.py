"""报告用的补充分析：截断率、标注口径差异（书名号）、hard 子集、跨数据集排位。

    python scripts/analysis.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import OUTPUTS, REPORTS, read_jsonl  # noqa: E402
from src.data import load_labels, load_splits  # noqa: E402
from src.metrics import evaluate  # noqa: E402

BRACKETS = "《》〈〉「」『』\"'“”‘’（）()"


def strip_brackets(rows: list[dict], text_key: str = "text") -> list[dict]:
    """把实体首尾的书名号/引号去掉，模拟「标注口径对齐后」的评测。"""
    out = []
    for r in rows:
        ents = []
        for e in r.get("entities", []):
            s, t = e["start"], e["end"]
            text = r[text_key]
            while s < t and text[s] in BRACKETS:
                s += 1
            while t > s and text[t - 1] in BRACKETS:
                t -= 1
            ents.append({"start": s, "end": t, "label": e["label"]})
        out.append({"id": r["id"], "text": r[text_key], "entities": ents})
    return out


def truncation_stats() -> list[str]:
    lines = ["## 截断率（字符长度超过训练 max_len 的比例）", "",
             "| 数据集 | 训练 max_len | test 超长句占比 | test 落在截断后的实体占比 |",
             "| --- | ---: | ---: | ---: |"]
    cfg = {"resume": 512, "cluener": 128, "msra": 256}
    for ds, max_len in cfg.items():
        rows = load_splits(ds)["test"]
        over = sum(1 for r in rows if len(r["text"]) > max_len) / len(rows)
        ent = [e for r in rows for e in r["entities"]]
        lost = sum(1 for e in ent if e["end"] > max_len) / max(1, len(ent))
        lines.append(f"| {ds} | {max_len} | {over*100:.1f}% | {lost*100:.2f}% |")
    return lines


def bracket_share(ds: str) -> str:
    rows = load_splits(ds)["test"]
    ent = [r["text"][e["start"]:e["end"]] for r in rows for e in r["entities"]]
    with_b = sum(1 for s in ent if s and (s[0] in BRACKETS or s[-1] in BRACKETS))
    return f"{ds}: 带书名号/引号的实体 {with_b}/{len(ent)} = {with_b/max(1,len(ent))*100:.1f}%"


def normalized_f1(ds: str, tag: str, method: str) -> tuple[float, float]:
    gold = load_splits(ds)["test"]
    if method == "llm":
        pred = read_jsonl(OUTPUTS / ds / "llm" / f"pred_{tag}.jsonl")
    else:
        pred = read_jsonl(OUTPUTS / ds / method / "pred.jsonl")
    ids = {r["id"] for r in pred}
    labels = load_labels(ds)
    strict = evaluate(gold, pred, labels, subset_ids=ids)["overall"]["f1"]
    norm = evaluate(strip_brackets(gold), strip_brackets(pred), labels, subset_ids=ids)["overall"]["f1"]
    return strict, norm


def ranking() -> list[str]:
    """从 compare 产出的 RESULTS.csv 读 F1，给出每个数据集的排位。"""
    import csv

    p = REPORTS / "RESULTS.csv"
    if not p.exists():
        return ["（RESULTS.csv 还没有，先跑 src.compare）"]
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    best: dict[str, list] = defaultdict(list)
    for r in rows:
        best[r["dataset"]].append((float(r["F1_same_batch"]), r["method"]))
    lines = ["## 各数据集排位", "", "| 数据集 | 第 1 | 第 2 | 第 3 | 第 4 | 第 5 |", "| --- | --- | --- | --- | --- | --- |"]
    for ds, lst in best.items():
        lst.sort(reverse=True)
        cells = [f"{m} {f:.3f}" for f, m in lst[:5]]
        cells += ["-"] * (5 - len(cells))
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    lines = ["# 补充分析（自动生成）", ""]
    lines += truncation_stats()
    lines += ["", "## 书名号/引号占比（CLUENER 的标注口径）", ""]
    for ds in ("cluener", "msra", "resume"):
        lines.append("- " + bracket_share(ds))
    lines += ["", "## 把书名号从两边去掉后的 F1（只作分析，不进正式表）", "",
              "| 数据集 | 方法 | strict F1 | 去书名号后 F1 | 差值 |", "| --- | --- | ---: | ---: | ---: |"]
    for ds in ("cluener", "msra", "resume"):
        for method, tag in (("bert", ""), ("llm", "few_n300")):
            if method == "llm":
                p = OUTPUTS / ds / "llm" / f"pred_{tag}.jsonl"
                if not p.exists():
                    continue
            try:
                s, n = normalized_f1(ds, tag, method)
            except Exception:
                continue
            lines.append(f"| {ds} | {method} | {s:.4f} | {n:.4f} | {n-s:+.4f} |")
    lines += [""] + ranking()
    out = REPORTS / "ANALYSIS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

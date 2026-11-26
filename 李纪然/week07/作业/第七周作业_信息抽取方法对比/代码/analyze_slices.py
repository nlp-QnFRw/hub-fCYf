# -*- coding: utf-8 -*-
"""
细粒度分析：把同一批预测按「事件类型」「论元角色」拆开比，回答"谁在哪类上赢"。

产出：
  结果/分类型对比.md      各方法在每个事件类型上的 F1（全量 dev 930 口径）
  结果/分角色对比.md      论元角色维度的 F1 Top/Bottom
  结果/错误类型分布.md    规则链路 + 生成式链路的错误分类统计

注意口径：
  - 判别式方法（规则/BiLSTM/BERT）用全量 dev 930；
  - LLM 链路只跑了 n=200 子集，单独给出、不与 930 混排。

用法：
  python 代码/analyze_slices.py --project <课程项目 code 目录>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(here.parent.parent / "week7-信息抽取" / "code"))
    ap.add_argument("--out_dir", default=str(here.parent / "结果"))
    args = ap.parse_args()
    project = Path(args.project)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project / "common"))
    from evaluate_events import evaluate, load_jsonl  # noqa

    dev = load_jsonl(str(project / "data" / "processed" / "dev_extract.jsonl"))
    gold_n200 = load_jsonl(str(project / "outputs" / "llm" / "gold_n200.jsonl"))
    k200 = {(r["id"], r.get("sent_id")) for r in gold_n200}

    preds = {
        "规则": project / "outputs" / "disc" / "pred_hw_rule_full.jsonl",
        "BiLSTM-CRF": project / "outputs" / "disc" / "pred_hw_bilstm_crf.jsonl",
        "BERT+Linear": project / "outputs" / "disc" / "pred_hw_bert_linear.jsonl",
    }
    llm_preds = {
        "Flash zero": project / "outputs" / "llm" / "pred_hw_api_flash_zero_n200.jsonl",
        "Flash few": project / "outputs" / "llm" / "pred_hw_api_flash_few_n200.jsonl",
        "V4-Pro zero": project / "outputs" / "llm" / "pred_hw_api_pro_zero_n200.jsonl",
    }

    # ── 1) 分事件类型 F1（全量 dev 930）──────────────────────────────────
    lines = ["# 分事件类型 F1（全量 dev 930 句）", ""]
    per_type = {}
    for name, p in preds.items():
        if not p.exists():
            continue
        res = evaluate(dev, load_jsonl(str(p)))
        per_type[name] = res["per_type"]
    if per_type:
        types = sorted(per_type[list(per_type)[0]], key=lambda t: -per_type[list(per_type)[0]][t]["f1"])
        cols = list(per_type)
        lines.append("| 事件类型 | dev 句数 | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 2))
        from collections import Counter
        cnt = Counter(r["event"]["type"] for r in dev)
        for t in types:
            vals = [f"{per_type[c][t]['f1']:.3f}" if t in per_type[c] else "—" for c in cols]
            lines.append(f"| {t} | {cnt[t]} | " + " | ".join(vals) + " |")
    (out / "分类型对比.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── 2) 分角色 F1（全量 dev 930）──────────────────────────────────────
    lines = ["# 分论元角色 F1（全量 dev 930 句）", "",
             "只列支撑数 ≥ 20 的角色；按规则基线 F1 升序（越靠前越难）。", ""]
    per_role = {}
    for name, p in preds.items():
        if p.exists():
            per_role[name] = evaluate(dev, load_jsonl(str(p)))["per_role"]
    if per_role:
        cols = list(per_role)
        base = cols[0]
        roles = sorted(per_role[base], key=lambda r: per_role[base][r]["f1"])
        cols = list(per_role)
        lines.append("| 角色 | 金标数 | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 2))
        for r in roles:
            if per_role[cols[0]][r]["tp"] + per_role[cols[0]][r]["fn"] < 20:
                continue
            vals = [f"{per_role[c][r]['f1']:.3f}" if r in per_role[c] else "—" for c in cols]
            lines.append(f"| {r} | {per_role[cols[0]][r]['tp'] + per_role[cols[0]][r]['fn']} | "
                         + " | ".join(vals) + " |")
    (out / "分角色对比.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── 3) LLM 链路分类型（n=200 子集）────────────────────────────────
    lines = ["# LLM 链路分事件类型 F1（固定 n=200 子集）", ""]
    llm_pt = {}
    for name, p in llm_preds.items():
        if p.exists():
            llm_pt[name] = evaluate(gold_n200, load_jsonl(str(p)))["per_type"]
    if llm_pt:
        from collections import Counter
        cnt = Counter(r["event"]["type"] for r in gold_n200)
        types = sorted(cnt, key=lambda t: -cnt[t])
        cols = list(llm_pt)
        lines.append("| 事件类型 | 句数 | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 2))
        for t in types:
            vals = [f"{llm_pt[c][t]['f1']:.3f}" if t in llm_pt[c] else "—" for c in cols]
            lines.append(f"| {t} | {cnt[t]} | " + " | ".join(vals) + " |")
    (out / "LLM分类型对比.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("已产出：分类型对比.md / 分角色对比.md / LLM分类型对比.md")


if __name__ == "__main__":
    main()

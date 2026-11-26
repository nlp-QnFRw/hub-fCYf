# -*- coding: utf-8 -*-
"""
把本周作业跑出的所有指标（outputs/metrics/{disc,llm}/hw_*.json）汇总成对比表。

设计原则：
  1. **只读指标文件，不写死任何数字**——表里每个格子都能反查到具体 json；
     缺失的方法显式标注「未跑」，绝不估算。
  2. 统一口径：所有方法都用课程项目 common/evaluate_events.py 算的同一套
     trigger / argument / event_exact / type_macro F1。
  3. 两条评测线分开报：
     - 全量 dev 930（规则/BiLSTM/BERT 这类可以跑全量的方法）
     - 固定 n=200 子集（所有方法都能投影到这里，含只跑了 200 条的 LLM 链路）
     两条线的数字**不能混在一张表里比**，这是本项目明确的口径要求。

用法：
  python 代码/summarize.py                 # 输出 结果/汇总表.md + 汇总表.csv
  python 代码/summarize.py --project <课程项目 code 目录>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# tag -> (展示名, 方法族, 训练数据规模, 备注)
REGISTRY = {
    # ── 规则基线 ────────────────────────────────────────────────────────
    "hw_rule_full":   ("L0 规则（词典+正则）",        "rule",      "全量 5747", "零训练；触发词统计+人工同义词，论元正则/词典"),
    "hw_rule_sample": ("L0 规则（小样本设定）",        "rule",      "小样本 800", "同一个规则系统，只换评测数据，看数据规模的影响"),
    "hw_rule_holdout": ("L0 规则（类型留出集）",       "rule",      "—",         "留出 3 个事件类型，仅用于测迁移"),
    "hw_rule_frac25": ("L0 规则（25% 训练数据）",      "rule",      "25% 1436",  "同一规则系统，训练统计只喂 1/4 数据"),
    "hw_bilstm_crf_frac25": ("L1 BiLSTM-CRF（25% 训练数据）", "bilstm", "25% 1436", "同配置 6 epoch，数据降到 1/4"),
    # ── 无预训练 ────────────────────────────────────────────────────────
    "hw_bilstm_crf":  ("L1 BiLSTM-CRF（无预训练）",   "bilstm",    "全量 5747", "字符级 BiLSTM-CRF，6 epoch @ max_len 256（本机 MPS）"),
    "hw_bilstm_crf_sample": ("L1 BiLSTM-CRF（小样本 800）", "bilstm", "小样本 800", "同配置 6 epoch，训练集只有 800 句"),
    "hw_bert_linear_sample": ("L2 BERT+Linear（小样本 800）", "bert", "小样本 800", "同结构 3 epoch；66 个训练步，明显训练不足"),
    # ── 预训练判别式 ────────────────────────────────────────────────────
    "hw_bert_linear": ("L2 BERT+Linear（联合抽取）",  "bert",      "全量 5747", "bert-base-chinese，149 标签 BIO，3 epoch @ max_len 128"),
    "hw_bert_linear_e1_256": ("L2 BERT+Linear（1 epoch 欠拟合对照）", "bert", "全量 5747", "同结构但只训 1 epoch：用来说明训练预算对结果的影响"),
    "hw_bert_linear_sample": ("L2 BERT+Linear（小样本设定）", "bert", "小样本 800", "同结构，训练集只给 800 句，看数据量的边际收益"),
    "hw_bert_linear_holdout": ("L2 BERT+Linear（类型留出集）", "bert", "留出训练 5008", "训练时删掉 3 个类型，测未见类型迁移"),
    # ── 生成式 / API ────────────────────────────────────────────────────
    "hw_api_flash_zero_n200": ("L3 DeepSeek-Flash zero-shot", "llm-api", "0", "DeepSeek-V4.1-Flash，prompt 只给 schema，不给示例"),
    "hw_api_flash_few_n200":  ("L3 DeepSeek-Flash few-shot",  "llm-api", "0", "同上 + 6 个 few-shot 示例"),
    "hw_api_pro_zero_n200":   ("L3 DeepSeek-V4-Pro zero-shot", "llm-api", "0", "同一条 prompt，换更强/更贵的 API 模型"),
    "hw_api_flash_zero_holdout_n116": ("L3 DeepSeek-Flash zero-shot（类型留出集）", "llm-api", "0", "同 prompt，测未见类型"),
}

FAMILY_ORDER = {"rule": 0, "bilstm": 1, "bert": 2, "pipeline": 3, "llm-api": 4}

# 表 1 只放「全量数据训练 + 全量 dev 评测」的方法：
# 小样本/类型留出集上的结果是**不同的评测集**，混进同一张表会被误读成"方法变差了"。
MAIN_N200 = {"hw_rule_full", "hw_bilstm_crf", "hw_bert_linear",
             "hw_api_flash_zero_n200", "hw_api_flash_few_n200", "hw_api_pro_zero_n200"}

ORDER = [
    "hw_rule_full", "hw_bilstm_crf", "hw_bert_linear", "hw_bert_linear_e1_256",
    "hw_api_flash_zero_n200", "hw_api_flash_few_n200", "hw_api_pro_zero_n200",
    "hw_rule_frac25", "hw_bilstm_crf_frac25",
    "hw_rule_sample", "hw_bilstm_crf_sample", "hw_bert_linear_sample",
    "hw_rule_holdout", "hw_api_flash_zero_holdout_n116",
]


def load_metrics(project: Path) -> dict[str, dict]:
    out = {}
    for sub in ("disc", "llm"):
        d = project / "outputs" / "metrics" / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("hw_*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"[warn] 指标文件损坏，跳过：{p}")
    return out


def row_of(tag: str, m: dict) -> dict:
    name, family, data, note = REGISTRY.get(tag, (tag, "?", "?", ""))
    meta = m.get("meta") or {}
    return {
        "tag": tag,
        "方法": name,
        "方法族": family,
        "训练数据": data,
        "n": m.get("num_samples"),
        "触发词F1": m.get("trigger", {}).get("f1"),
        "触发词P": m.get("trigger", {}).get("precision"),
        "触发词R": m.get("trigger", {}).get("recall"),
        "论元F1": m.get("argument", {}).get("f1"),
        "论元P": m.get("argument", {}).get("precision"),
        "论元R": m.get("argument", {}).get("recall"),
        "事件级F1": m.get("event_exact", {}).get("f1"),
        "类型macroF1": m.get("type_macro_f1"),
        "参数(M)": meta.get("params_m"),
        "训练(s)": meta.get("train_s"),
        "延迟(ms/句)": meta.get("ms_per_sentence"),
        "epoch": meta.get("epochs", meta.get("arg_epochs")),
        "max_len": meta.get("max_len"),
        "备注": note,
    }


def sub200_row(tag: str, m: dict) -> dict | None:
    s = m.get("n200_summary")
    if not s:
        return None
    name, family, data, note = REGISTRY.get(tag, (tag, "?", "?", ""))
    meta = m.get("meta") or {}
    return {
        "tag": tag + "@n200",
        "方法": name,
        "方法族": family,
        "训练数据": data,
        "n": s.get("num_samples", 200),
        "触发词F1": s.get("trigger_f1"),
        "触发词P": None, "触发词R": None,
        "论元F1": s.get("argument_f1"),
        "论元P": None, "论元R": None,
        "事件级F1": s.get("event_f1"),
        "类型macroF1": None,
        "参数(M)": meta.get("params_m"),
        "训练(s)": meta.get("train_s"),
        "延迟(ms/句)": meta.get("ms_per_sentence"),
        "epoch": meta.get("epochs"),
        "max_len": meta.get("max_len"),
        "备注": note,
    }


ND = {"参数(M)": 1, "训练(s)": 1, "延迟(ms/句)": 2, "epoch": 0, "max_len": 0, "n": 0}


def fmt(v, col=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        nd = ND.get(col, 4)
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str], title: str) -> str:
    if not rows:
        return f"### {title}\n\n（暂无数据）\n"
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c), c) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="", help="课程项目 code 目录（含 outputs/）")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 结果/）")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    project = Path(args.project) if args.project else (here.parent.parent / "week7-信息抽取" / "code")
    out_dir = Path(args.out_dir) if args.out_dir else here.parent / "结果"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(project)
    if not metrics:
        raise SystemExit(f"没有找到任何 hw_*.json 指标，检查 --project：{project}")

    full, n200, other = [], [], []
    for tag in ORDER + [t for t in metrics if t not in ORDER]:
        m = metrics.get(tag)
        if not m:
            continue
        r = row_of(tag, m)
        if r["n"] and r["n"] >= 900:
            full.append(r)
        elif r["n"] and r["n"] == 200:
            n200.append(r)
        else:
            other.append(r)
        s = sub200_row(tag, m)

    # n=200 主表：LLM 直接跑的就是 200 条；判别式方法的 n200 从 n200_summary 取
    main200 = []
    for tag in ORDER:
        m = metrics.get(tag)
        if not m or tag not in MAIN_N200:
            continue
        if tag.startswith("hw_api") and m.get("num_samples") == 200:
            main200.append(row_of(tag, m))
        else:
            s = sub200_row(tag, m)
            if s:
                main200.append(s)
    rank = {t: i for i, t in enumerate(ORDER)}
    main200.sort(key=lambda r: (FAMILY_ORDER.get(r["方法族"], 9), rank.get(r["tag"].replace("@n200", ""), 99)))

    cols_main = ["方法", "触发词F1", "论元F1", "事件级F1", "类型macroF1", "训练数据", "参数(M)", "训练(s)", "延迟(ms/句)", "备注"]
    cols_full = ["方法", "训练数据", "触发词F1", "论元F1", "事件级F1", "类型macroF1",
                 "参数(M)", "训练(s)", "延迟(ms/句)", "epoch", "max_len"]

    md = ["# 信息抽取方法对比汇总（自动生成，数字来自 outputs/metrics/*/hw_*.json）", ""]
    md.append(md_table(main200, cols_main,
                       "表 1  主对比：固定 n=200 子集（所有方法唯一可比口径）"))
    md.append(md_table(full, cols_full, "表 2  同一评测集 dev 930 句：换训练数据量"))
    md.append(md_table(other, cols_main, "表 3  其它评测集（小样本 / 类型留出集）"))

    # 自动结论
    def get(tag, field="论元F1"):
        for r in main200:
            if r["tag"].startswith(tag):
                return r[field]
        return None

    md.append("### 自动生成的关键对比（n=200 子集，论元 F1）")
    pairs = [
        ("hw_rule_full", "hw_bilstm_crf", "规则 → 无预训练 BiLSTM-CRF（建模本身的价值）"),
        ("hw_bilstm_crf", "hw_bert_linear", "BiLSTM-CRF → BERT 预训练（预训练的价值）"),
        ("hw_bert_linear", "hw_api_flash_zero_n200", "监督 BERT → LLM zero-shot（生成式的价值）"),
        ("hw_api_flash_zero_n200", "hw_api_flash_few_n200", "zero-shot → few-shot（示例的价值）"),
        ("hw_api_flash_zero_n200", "hw_api_pro_zero_n200", "Flash → V4-Pro（换更大 API 模型）"),
    ]
    for a, b, desc in pairs:
        va, vb = get(a), get(b)
        if va is not None and vb is not None:
            md.append(f"- {desc}：{va:.4f} → {vb:.4f}（{vb - va:+.4f}，相对 "
                      f"{(vb - va) / max(abs(va), 1e-9):+.1%}）")
    md.append("")

    (out_dir / "汇总表.md").write_text("\n".join(md), encoding="utf-8")

    # 长表 CSV：所有行都留痕
    fields = ["tag", "方法", "方法族", "训练数据", "n", "触发词P", "触发词R", "触发词F1",
              "论元P", "论元R", "论元F1", "事件级F1", "类型macroF1",
              "参数(M)", "训练(s)", "延迟(ms/句)", "epoch", "max_len", "备注"]
    with open(out_dir / "汇总表.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in full + n200 + other:
            w.writerow(r)
        for r in main200:
            if r["tag"].endswith("@n200"):
                w.writerow(r)

    print("\n".join(md))
    print(f"已写入：{out_dir / '汇总表.md'} / {out_dir / '汇总表.csv'}")


if __name__ == "__main__":
    main()

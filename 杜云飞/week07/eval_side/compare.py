# -*- coding: utf-8 -*-
"""
compare.py — D 组收口：汇总所有方法的指标 JSON → 统一对比表（Markdown + JSON）

口径纪律：
  - 只读 outputs/metrics/disc/*.json 的真实产物，缺文件的方法标"未跑"，绝不填估算值；
  - 主口径 dev 全量（判别式）与 n200 同批口径（LLM 侧 + 判别式投影）分开成两张表，
    LLM 数字只允许出现在 n200 表（课程坑：不同样本量的数字不能混排）；
  - 所有数字带溯源（run_name → metrics 文件名，可逐条反查）。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics" / "disc"

A_ORDER = [
    "bert12L_linear", "bert12L_crf", "bert4L_linear", "bert4L_crf", "bert4Lrand_crf",
    "roberta12L_linear", "roberta12L_crf", "roberta4L_linear", "roberta4L_crf",
    "macbert12L_linear", "macbert12L_crf", "macbert4L_linear", "macbert4L_crf",
]
# 顺序 = 报告表格顺序：先云端 API（基线），再本地模型按参数量从小到大，最后微调版
LLM_ORDER = [
    "api_deepseek-flash_zero_rep1", "api_deepseek-flash_zero_rep2", "api_deepseek-flash_few",
    "local_Qwen2-0.5B-Instruct_zero", "local_Qwen3-0.6B_zero", "local_Qwen3-0.6B_few",
    "local_Qwen3-0.6B_lora", "local_Qwen3-1.7B_zero", "local_Qwen3-4B_zero",
]


def load(name: str):
    p = METRICS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def row_of(name: str, r: dict, n200: bool) -> dict:
    src = r.get("n200_summary") if n200 else r
    meta = r.get("meta", {})
    if n200:
        return {
            "方法": meta.get("display_name", name), "run": name,
            "trigger F1": src.get("trigger_f1", ""), "argument F1": src.get("argument_f1", ""),
            "event F1": src.get("event_f1", ""),
        }
    return {
        "方法": meta.get("display_name", name), "run": name,
        "trigger F1": r["trigger"]["f1"], "argument F1": r["argument"]["f1"],
        "event F1": r["event_exact"]["f1"], "type_macro F1": r.get("type_macro_f1", ""),
        "训练(s)": meta.get("train_s", ""), "延迟(ms/句)": meta.get("ms_per_sentence", ""),
        "峰值显存(MiB)": meta.get("peak_gpu_mb", ""),
    }


def to_md(rows: list[dict], title: str) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        return "\n".join(lines + ["（全部未跑）", ""])
    header = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "|".join(["---"] * len(rows[0])) + "|"
    body = ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
    return "\n".join(lines + [header, sep, *body, ""])


def main():
    out = {"A_dev_full": [], "all_n200": []}
    md = ["# FewFC 事件抽取 方法对比（自动汇总，全部来自 outputs/metrics/disc/*.json）", ""]

    for name in A_ORDER:
        r = load(name)
        if r is None:
            md.append(f"- A 组 `{name}`：**未跑**")
            continue
        out["A_dev_full"].append(row_of(name, r, n200=False))
        out["all_n200"].append(row_of(name, r, n200=True))
    md.append("")
    md.append(to_md(out["A_dev_full"], "表 1：A 组判别式（dev 全量 2548 句口径）"))

    for name in LLM_ORDER:
        r = load(name)
        if r is None:
            md.append(f"- LLM `{name}`：**未跑**")
            continue
        out["all_n200"].append(row_of(name, r, n200=True))
    md.append("")
    md.append(to_md(out["all_n200"], "表 2：全方法 n200 同批口径（200 句，跨方法唯一可比）"))

    # 可复现性：两次重跑的差
    reps = [(n, load(n)) for n in LLM_ORDER if "rep" in n]
    md.append("## 附：API 重跑可复现性（temperature=0 下的服务端波动）")
    md.append("")
    for n, r in reps:
        if r:
            m = r["meta"]
            md.append(f"- {n}: trigger={r['trigger']['f1']} argument={r['argument']['f1']} "
                      f"json_valid={m.get('json_valid_rate')}")
    md.append("")

    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "compare_table.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "outputs" / "compare_table.md").write_text("\n".join(md), encoding="utf-8")
    print("→ outputs/compare_table.md / compare_table.json")


if __name__ == "__main__":
    main()

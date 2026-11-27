# -*- coding: utf-8 -*-
"""fill_report.py — 把 outputs/metrics/disc/*.json 的真实数字回填进 HOMEWORK_REPORT.md

为什么用脚本而不是手抄：
  报告里的每个数字必须能反查到 metrics 文件（课程纪律：口径溯源）。手抄一遍就会出现
  "图注写 2 epoch、实际跑了 3 epoch"这类自相矛盾。这里只做一件事：
  marker 之间的表格由脚本生成，正文结论人工写——数字和叙事分离，改数不用改文。

用法：python eval_side/fill_report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics" / "disc"
REPORT = ROOT / "HOMEWORK_REPORT.md"

A_ORDER = [
    "bert12L_crf", "bert12L_linear", "bert4L_crf", "bert4L_linear", "bert4Lrand_crf",
    "roberta12L_crf", "roberta4L_crf", "roberta12L_linear", "roberta4L_linear",
    "macbert12L_crf", "macbert4L_crf", "macbert12L_linear", "macbert4L_linear",
]

LLM_ROWS = [
    ("api_deepseek-flash_zero_rep1", "DeepSeek API zero-shot #1", "thinking 关，temp=0"),
    ("api_deepseek-flash_zero_rep2", "DeepSeek API zero-shot #2", "同参重跑，测波动"),
    ("api_deepseek-flash_few", "DeepSeek API few-shot(3 例)", "prompt 预算受限的折中"),
    ("local_Qwen2-0.5B-Instruct_zero", "本地 Qwen2-0.5B zero-shot", "上一代同量级"),
    ("local_Qwen3-0.6B_zero", "本地 Qwen3-0.6B zero-shot", "8GB 全量驻留"),
    ("local_Qwen3-0.6B_few", "本地 Qwen3-0.6B few-shot(9 例)", "示例是否撑爆上下文"),
    ("local_Qwen3-0.6B_lora", "本地 Qwen3-0.6B LoRA-SFT", "train 3552 句 SFT"),
    ("local_Qwen3-1.7B_zero", "本地 Qwen3-1.7B zero-shot", "bf16 全量上卡"),
    ("local_Qwen3-4B_zero", "本地 Qwen3-4B zero-shot", "device_map=auto，输出被 800 token 截断"),
]


def load(name: str):
    p = METRICS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def a_table() -> str:
    head = ("| 配置 | 编码器 | 层数 | 初始化 | 解码器 | trigger F1 | argument F1 | event F1 | "
            "内层验证 spanF1 | 训练(min) | 峰值显存(MiB) | 截断句 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines, missing = [head], []
    for name in A_ORDER:
        r = load(name)
        if r is None:
            missing.append(name)
            continue
        m = r.get("meta", {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            m.get("display_name", name), m.get("encoder", ""), m.get("bert_layers", ""),
            m.get("init", ""), str(m.get("decoder", "")).upper(),
            fmt(r["trigger"]["f1"]), fmt(r["argument"]["f1"]), fmt(r["event_exact"]["f1"]),
            fmt(m.get("inner_val_span_f1")), fmt(float(m.get("train_s", 0)) / 60, 1),
            fmt(m.get("peak_gpu_mb"), 0), m.get("n_truncated", "")))
    tail = ""
    if missing:
        tail = "\n\n未跑：" + "、".join(missing) + "（如实标注，不填估算值）"
    return "\n".join(lines) + tail


def n200_table() -> str:
    head = ("| 方法 | trigger F1 | argument F1 | event F1 | JSON 合法率 | 每句耗时 | 备注 |\n"
            "|---|---|---|---|---|---|---|")
    lines = [head]
    for name, label, note in LLM_ROWS:
        r = load(name)
        if r is None:
            lines.append("| {} | 未跑 | — | — | — | — | {} |".format(label, note))
            continue
        m = r.get("meta", {})
        s = r.get("n200_summary") or {}
        sec = m.get("sec_per_sentence")
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            label, fmt(s.get("trigger_f1", r["trigger"]["f1"])),
            fmt(s.get("argument_f1", r["argument"]["f1"])),
            fmt(s.get("event_f1", r["event_exact"]["f1"])),
            fmt(m.get("json_valid_rate"), 2),
            (f"{float(sec):.2f}s" if sec else "—"), note))
    # n200 表只放 5 个有故事线的判别式行（全 13 行见 2.1，避免表格被近似行淹没）
    for name in ["bert12L_crf", "roberta12L_crf", "macbert12L_crf", "bert4L_crf", "bert4Lrand_crf"]:
        r = load(name)
        if not r:
            continue
        s = r.get("n200_summary") or {}
        if not s:
            continue
        lines.append("| {}（n200 投影） | {} | {} | {} | — | — | 单事件解码 |".format(
            r.get("meta", {}).get("display_name", name),
            fmt(s.get("trigger_f1")), fmt(s.get("argument_f1")), fmt(s.get("event_f1"))))
    return "\n".join(lines)


def replace_block(text: str, marker: str, body: str) -> str:
    close = "<!-- /" + marker[len("<!-- "):]
    start = text.index(marker) + len(marker)
    end = text.index(close, start)
    return text[:start] + "\n\n" + body + "\n\n" + text[end:]


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    text = replace_block(text, "<!-- TABLE:A_GROUP -->", a_table())
    text = replace_block(text, "<!-- TABLE:ALL_N200 -->", n200_table())
    REPORT.write_text(text, encoding="utf-8")
    print("已回填 A_GROUP / ALL_N200 两张表 →", REPORT)


if __name__ == "__main__":
    main()

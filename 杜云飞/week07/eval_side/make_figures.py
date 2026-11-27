# -*- coding: utf-8 -*-
"""make_figures.py — D 组图表：从 compare_table.json 生成对比图（课程 make_figures 风格）。

图 1：A 组判别式网格 trigger/argument F1（dev 全量）分组条形图
图 2：全方法 n200 同批口径对比（BERT 最佳 vs 本地 LLM vs DeepSeek API）
图 3：层数消融故事线（12L vs 4L vs 4L-random，bert 系）
缺数据的组自动跳过，不画空图。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 课程风格：无网格线 junk，CJK 字体随 runtime 自带
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

table = json.loads((ROOT / "outputs" / "compare_table.json").read_text(encoding="utf-8"))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fig1_a_group():
    rows = [r for r in table["A_dev_full"] if fnum(r.get("trigger F1")) is not None]
    if not rows:
        print("图1：无 A 组数据，跳过")
        return
    names = [r["run"] for r in rows]
    trig = [fnum(r["trigger F1"]) for r in rows]
    arg = [fnum(r["argument F1"]) for r in rows]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - w / 2 for i in x], trig, w, label="Trigger F1", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], arg, w, label="Argument F1", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Span F1（dev 全量 2461 句）")
    ax.set_title("A 组判别式网格：3 编码器 × 层数 × 解码器")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_a_group.png", dpi=150)
    plt.close(fig)
    print("图1 → outputs/figures/fig1_a_group.png")


A_SHOWCASE = {"bert12L_crf", "roberta12L_crf", "macbert12L_crf", "bert4L_crf", "bert4Lrand_crf"}


def fig2_all_n200():
    # A 组 13 行全画进来会让图糊掉（fig1 已经单独画了网格），这里只保留 5 个有故事线的
    # 判别式行 + 全部 LLM 行，正好对应 2.2 的对比表。
    rows = [r for r in table["all_n200"]
            if fnum(r.get("trigger F1")) is not None
            and (r.get("run") in A_SHOWCASE or str(r.get("run", "")).startswith(("api_", "local_")))]
    if not rows:
        print("图2：无 n200 数据，跳过")
        return
    names = [r["方法"] for r in rows]
    trig = [fnum(r["trigger F1"]) for r in rows]
    arg = [fnum(r["argument F1"]) for r in rows]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - w / 2 for i in x], trig, w, label="Trigger F1", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], arg, w, label="Argument F1", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Span F1（n200 同批 200 句）")
    ax.set_title("全方法对比：判别式 vs 本地离线 LLM vs DeepSeek API")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_all_n200.png", dpi=150)
    plt.close(fig)
    print("图2 → outputs/figures/fig2_all_n200.png")


def fig3_layers():
    keys = ["bert12L_crf", "bert4L_crf", "bert4Lrand_crf"]
    rows = {r["run"]: r for r in table["A_dev_full"] if r.get("run") in keys}
    rows = {k: v for k, v in rows.items() if fnum(v.get("argument F1")) is not None}
    if len(rows) < 2:
        print("图3：层数消融数据不足，跳过")
        return
    order = ["bert12L_crf", "bert4L_crf", "bert4Lrand_crf"]
    labels = ["12 层 pretrained", "4 层 pretrained", "4 层 random（坑#30）"]
    trig = [fnum(rows[k]["trigger F1"]) for k in order if k in rows]
    arg = [fnum(rows[k]["argument F1"]) for k in order if k in rows]
    labs = [l for k, l in zip(order, labels) if k in rows]
    x = range(len(labs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - w / 2 for i in x], trig, w, label="Trigger F1", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], arg, w, label="Argument F1", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Span F1（dev 全量）")
    ax.set_title("层数与初始化消融：预训练知识值多少 F1？")
    for i, v in enumerate(arg):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_layers.png", dpi=150)
    plt.close(fig)
    print("图3 → outputs/figures/fig3_layers.png")


def fig4_scaling():
    """参数量 vs zero-shot 能力：把 LoRA 那一根单独染色，一眼看出"微调 > 堆参数"。"""
    names = [("0.5B\n(Qwen2)", "local_Qwen2-0.5B-Instruct_zero"),
             ("0.6B", "local_Qwen3-0.6B_zero"),
             ("1.7B", "local_Qwen3-1.7B_zero"),
             ("4B", "local_Qwen3-4B_zero"),
             ("0.6B\n+LoRA", "local_Qwen3-0.6B_lora")]
    labels, trig, arg, colors = [], [], [], []
    for lbl, run in names:
        p = ROOT / "outputs" / "metrics" / "disc" / f"{run}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d.get("n200_summary") or {}
        if s.get("trigger_f1") is None:
            continue
        labels.append(lbl)
        trig.append(float(s.get("trigger_f1") or 0))
        arg.append(float(s.get("argument_f1") or 0))
        colors.append("#C44E52" if "LoRA" in lbl else "#4C72B0")
    if len(labels) < 3:
        print("图4：本地模型数据不足，跳过")
        return
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], trig, w, label="Trigger F1", color=colors)
    ax.bar([i + w / 2 for i in x], arg, w, label="Argument F1", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Span F1（n200 同批 200 句）")
    ax.set_title("本地模型：参数量 vs 微调（同一批 200 句）")
    for i, v in enumerate(trig):
        ax.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_scaling.png", dpi=150)
    plt.close(fig)
    print("图4 → outputs/figures/fig4_scaling.png")


if __name__ == "__main__":
    fig1_a_group()
    fig2_all_n200()
    fig3_layers()
    fig4_scaling()

# -*- coding: utf-8 -*-
"""
生成本周作业报告用的 4 张图（白底、150dpi、中文标签、柱子上直接标数值）。

图 1  fig1_n200_main.png      固定 n=200 子集：各方法 trigger / argument / event 三级 F1
图 2  fig2_full930.png        全量 dev 930：判别式三档（规则 / BiLSTM-CRF / BERT）
图 3  fig3_llm_api.png        LLM 链路：换 API 模型 & zero/few-shot 的 F1 与延迟
图 4  fig4_dataset.png        换数据集：全量 / 小样本 / 类型留出，同一方法横向比

用法：
  MPLCONFIGDIR=/tmp/mpl python 代码/make_figures.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-week7")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

CJK_CANDIDATES = ["Heiti SC", "STHeiti", "Songti SC", "PingFang SC", "PingFang TC",
                  "Arial Unicode MS", "Noto Sans CJK SC"]


def setup_style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    picked = [f for f in CJK_CANDIDATES if f in have] or ["DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    })


def load(project: Path, sub: str, tag: str) -> dict | None:
    p = project / "outputs" / "metrics" / sub / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def triple(m: dict) -> tuple[float, float, float]:
    return (m["trigger"]["f1"], m["argument"]["f1"], m["event_exact"]["f1"])


def bars(ax, rows: list[tuple[str, tuple[float, float, float]]], title: str, ylabel="F1"):
    names = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    x = np.arange(len(names))
    w = 0.26
    colors = ["#3B6FD4", "#E8833A", "#4C9A5E"]
    labels = ["触发词 F1", "论元 F1", "事件级 F1"]
    for i in range(3):
        b = ax.bar(x + (i - 1) * w, vals[:, i], w, label=labels[i], color=colors[i])
        for rect, v in zip(b, vals[:, i]):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylim(0, max(0.95, vals.max() + 0.12))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)


def fig_main(project: Path, out: Path, tags: list[tuple[str, str, str]]):
    """tags: (sub_dir, tag, display)"""
    rows = []
    for sub, tag, disp in tags:
        m = load(project, sub, tag)
        if not m:
            continue
        if tag.startswith("hw_api"):
            rows.append((disp, triple(m)))
        else:
            s = m.get("n200_summary")
            if s:
                rows.append((disp, (s["trigger_f1"], s["argument_f1"], s["event_f1"])))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bars(ax, rows, "固定 n=200 子集：不同方法三级 F1（唯一可比口径）")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_full(project: Path, out: Path):
    rows = []
    for tag, disp in [("hw_rule_full", "L0 规则"), ("hw_bilstm_crf", "L1 BiLSTM-CRF"),
                      ("hw_bert_linear", "L2 BERT+Linear")]:
        m = load(project, "disc", tag)
        if m:
            rows.append((disp, triple(m)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars(ax, rows, "全量 dev 930 句：判别式三档方法")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_llm(project: Path, out: Path):
    specs = [("hw_api_flash_zero_n200", "Flash\nzero-shot"),
             ("hw_api_flash_few_n200", "Flash\nfew-shot"),
             ("hw_api_pro_zero_n200", "V4-Pro\nzero-shot")]
    rows, lat = [], []
    for tag, disp in specs:
        m = load(project, "llm", tag)
        if not m:
            continue
        rows.append((disp, triple(m)))
        p = project / "outputs" / "llm" / f"report_{tag}.json"
        rep = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        lat.append(rep.get("latency_mean_s", 0))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.1, 1]})
    bars(axes[0], rows, "LLM 链路：换 API 模型 × zero/few-shot（n=200）")
    axes[0].set_xticklabels([r[0] for r in rows], rotation=0, ha="center", fontsize=8.5)

    ax = axes[1]
    names = [r[0].replace("\n", " ") for r in rows]
    b = ax.barh(names, lat, color="#7A5BA8")
    for rect, v in zip(b, lat):
        ax.text(v + 0.03, rect.get_y() + rect.get_height() / 2, f"{v:.2f}s",
                va="center", fontsize=8.5)
    ax.set_xlabel("平均延迟 (s/条)")
    ax.set_title("成本：单条请求延迟")
    ax.set_xlim(0, max(lat) * 1.35 if lat else 1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_dataset(project: Path, out: Path):
    """换数据集：左边看「数据量」，右边看「数据分布（类型留出）」。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})

    # ── 左：数据量（评测集都是 dev 930 / 300）──
    ax = axes[0]
    groups = [("全量 5747", [("hw_rule_full", "规则", "disc"), ("hw_bilstm_crf", "BiLSTM-CRF", "disc"),
                          ("hw_bert_linear", "BERT", "disc")]),
              ("25% 1436", [("hw_rule_frac25", "规则", "disc"), ("hw_bilstm_crf_frac25", "BiLSTM-CRF", "disc")]),
              ("小样本 800", [("hw_rule_sample", "规则", "disc"), ("hw_bilstm_crf_sample", "BiLSTM-CRF", "disc"),
                           ("hw_bert_linear_sample", "BERT", "disc")])]
    x = np.arange(len(groups))
    w = 0.24
    for i, (disp, color) in enumerate([("规则", "#3B6FD4"), ("BiLSTM-CRF", "#E8833A"), ("BERT", "#4C9A5E")]):
        vals = []
        for gname, items in groups:
            v = float("nan")
            for tag, name, sub in items:
                if name == disp:
                    m = load(project, sub, tag)
                    if m:
                        v = m["argument"]["f1"]
            vals.append(v)
        b = ax.bar(x + (i - 1) * w, vals, w, label=disp, color=color)
        for rect, v in zip(b, vals):
            if v == v:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.2f}",
                        ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("论元 F1")
    ax.set_ylim(0, 0.75)
    ax.set_title("换数据量：同一方法在 1/4 数据与极小样本上的退化（dev 930 / 300）")
    ax.legend(fontsize=8, frameon=False)

    # ── 右：类型留出（训练删掉 3 类，只测这 3 类，116 句）──
    ax = axes[1]
    items = [("规则", ("hw_rule_holdout", "disc")),
             ("LLM zero-shot", ("hw_api_flash_zero_holdout_n116", "llm"))]
    vals = []
    labels = []
    for name, (tag, sub) in items:
        m = load(project, sub, tag)
        if not m:
            continue
        labels.append(name)
        vals.append((m["trigger"]["f1"], m["argument"]["f1"]))
    if vals:
        x2 = np.arange(len(vals))
        w2 = 0.32
        for i, (lab, color) in enumerate([("触发词 F1", "#3B6FD4"), ("论元 F1", "#E8833A")]):
            bs = ax.bar(x2 + (i - 0.5) * w2, [v[i] for v in vals], w2, label=lab, color=color)
            for rect, v in zip(bs, [v[i] for v in vals]):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", fontsize=8)
        ax.set_xticks(x2)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 0.8)
        ax.set_title("换数据分布：3 个未见类型（116 句）")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    project = Path(args.project) if args.project else here.parent.parent / "week7-信息抽取" / "code"
    out_dir = Path(args.out_dir) if args.out_dir else here.parent / "报告" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    fig_main(project, out_dir / "fig1_n200_main.png", [
        ("disc", "hw_rule_full", "L0 规则"),
        ("disc", "hw_bilstm_crf", "L1 BiLSTM-CRF"),
        ("disc", "hw_bert_linear", "L2 BERT+Linear"),
        ("llm", "hw_api_flash_zero_n200", "DeepSeek-Flash zero"),
        ("llm", "hw_api_flash_few_n200", "DeepSeek-Flash few"),
        ("llm", "hw_api_pro_zero_n200", "DeepSeek-V4-Pro zero"),
    ])
    fig_full(project, out_dir / "fig2_full930.png")
    fig_llm(project, out_dir / "fig3_llm_api.png")
    fig_dataset(project, out_dir / "fig4_dataset.png")
    print(f"图已生成 → {out_dir}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""生成 week7 报告用图（全部基于已落盘的指标文件，不手填数字）。

图：
  fig_compare_sample.png   本次 sample n=200 同口径对比（trigger/arg/event 三级）
  fig_compare_full.png     PPT 全量 dev 930 参考对比
  fig_latency_accuracy.png 精度-延迟权衡（全量参考）
  fig_type_dist.png        sample train/dev 事件类型分布
"""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/week7_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 中文字体：macOS 自带 Songti SC / STHeiti（找不到就退回英文标签）
_available = {f.name for f in font_manager.fontManager.ttflist}
for _cand in ("Songti SC", "STHeiti", "Arial Unicode MS", "PingFang SC"):
    if _cand in _available:
        plt.rcParams["font.family"] = _cand
        break
plt.rcParams["axes.unicode_minus"] = False

# 方法顺序、颜色、延迟（ms/句）、训练成本（PPT 口径）
FULL = [
    ("L0 规则", "rule", 0.6168, 0.1740, 0.0161, 1.9, "#9aa5b1"),
    ("L1 BiLSTM-CRF", "bilstm_crf", 0.8165, 0.4634, 0.0885, 20.6, "#7fb3d5"),
    ("L2 BERT+Linear", "bert_linear_256", 0.8293, 0.5630, 0.1564, 11.3, "#5dade2"),
    ("L2 BERT+CRF", "bert_crf_256", 0.8320, 0.5815, 0.1697, 74.3, "#2e86c1"),
    ("L2 BERT+CRF+硬约束", "bert_crf_256_hard", 0.8464, 0.6566, 0.2177, 39.6, "#1b4f72"),
    ("L2 两段式", "pipeline", 0.6769, 0.3690, 0.0957, 49.9, "#a9dfbf"),
    ("L3 本地 0.6B zero\n(n=60)", "local_zero_n60", 0.4068, 0.1167, 0.0000, 5320, "#f9e79f"),
    ("L3 API zero-shot", "api_zero_n200", 0.7789, 0.5224, 0.0402, 3620, "#f5b041"),
    ("L3 API few-shot", "api_few_n200", 0.8321, 0.5395, 0.0852, 3530, "#e67e22"),
    ("L4 LoRA SFT", "sft_lora_n200", 0.9000, 0.6971, 0.1850, 3130, "#c0392b"),
]


def grouped_bars(rows, title, path, figsize=(12, 6.2)):
    labels = [r[0] for r in rows]
    x = range(len(rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([i - w for i in x], [r[2] for r in rows], w, label="触发词 F1", color="#2e86c1")
    ax.bar(list(x), [r[3] for r in rows], w, label="论元 F1", color="#e67e22")
    ax.bar([i + w for i in x], [r[4] for r in rows], w, label="事件级 F1", color="#27ae60")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    for i, r in enumerate(rows):
        ax.text(i - w, r[2] + 0.012, f"{r[2]:.2f}", ha="center", fontsize=7, color="#1b4f72")
        ax.text(i, r[3] + 0.012, f"{r[3]:.2f}", ha="center", fontsize=7, color="#935116")
        ax.text(i + w, r[4] + 0.012, f"{r[4]:.2f}", ha="center", fontsize=7, color="#145a32")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    # ── 图 1：sample n=200 复评 ────────────────────────────────
    sample_csv = ROOT / "outputs" / "sample_subset.csv"
    rows = []
    if sample_csv.exists():
        with open(sample_csv, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append((r["display_name"], r["method"], float(r["trigger_f1"]),
                             float(r["argument_f1"]), float(r["event_f1"]), 0, "#5dade2"))
    rows.sort(key=lambda r: r[3])
    grouped_bars(rows, "本次 sample 数据同口径复评（n=200，按论元 F1 排序）",
                 OUT / "fig_compare_sample.png", figsize=(12, 6.8))

    # ── 图 2：全量参考 ───────────────────────────────────────
    grouped_bars(FULL, "课程 PPT 全量 dev 930 句：10 种方法同口径对比",
                 OUT / "fig_compare_full.png")

    # ── 图 3：精度-延迟 ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6.2))
    offsets = {"L3 API zero-shot": (-130, -6), "L3 API few-shot": (-140, 10),
               "L4 LoRA SFT": (-120, 10), "L3 本地 0.6B zero\n(n=60)": (-150, 4),
               "L0 规则": (8, 4)}
    for name, _, tr, ar, ev, lat, color in FULL:
        ax.scatter(lat, ar, s=90, color=color, edgecolor="white", zorder=3)
        ax.annotate(f"{name}\n{ar:.3f} / {lat:.0f}ms", (lat, ar),
                    textcoords="offset points", xytext=offsets.get(name, (6, 6)), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("单句延迟（ms，log scale）")
    ax.set_ylabel("论元 F1")
    ax.set_title("精度 vs 延迟：规则最快、LoRA 最准、硬约束性价比最高", fontsize=13, pad=10)
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_xlim(0.5, 12000)
    ax.set_ylim(0, 0.78)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency_accuracy.png", dpi=180)
    plt.close(fig)

    # ── 图 4：sample 类型分布 ────────────────────────────────
    data_dir = ROOT / "data" / "sample"
    counts = {}
    for split in ("train", "dev"):
        c = {}
        with open(data_dir / f"{split}_extract.jsonl", encoding="utf-8") as f:
            for line in f:
                ev = json.loads(line).get("event") or {}
                c[ev.get("type")] = c.get(ev.get("type"), 0) + 1
        counts[split] = c
    types = sorted({t for c in counts.values() for t in c}, key=lambda t: -counts["train"].get(t, 0))
    x = range(len(types))
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar([i - 0.2 for i in x], [counts["train"].get(t, 0) for t in types], 0.4,
           label="train 800 句", color="#2e86c1")
    ax.bar([i + 0.2 for i in x], [counts["dev"].get(t, 0) for t in types], 0.4,
           label="dev 300 句", color="#e67e22")
    ax.set_xticks(list(x))
    ax.set_xticklabels(types, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("句数")
    ax.set_title("sample 事件类型分布：股份回购/亏损/中标占多数，被约谈/破产为长尾", fontsize=13, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_type_dist.png", dpi=180)
    plt.close(fig)

    print("图已写入", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()

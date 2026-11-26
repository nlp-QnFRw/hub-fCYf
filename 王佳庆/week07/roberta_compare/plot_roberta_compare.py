"""把 10 份 RoBERTa 系列评测 JSON 画成对比图。

数据：outputs/metrics/disc/roberta_*.json
  - 不带 _n200 的文件：全量 dev（930 句）
  - 带   _n200 的文件：固定抽样的 200 句（跨方法同批对比）

用法：
  python roberta_compare/plot_roberta_compare.py
  python roberta_compare/plot_roberta_compare.py --only overall f1
  python roberta_compare/plot_roberta_compare.py --metrics-dir ... --out-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前：无界面后端
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_METRICS_DIR = ROOT / "outputs" / "metrics" / "disc"
DEFAULT_OUT_DIR = HERE / "figures"

# (文件主名, 图例标签, 统一配色)
METHODS = [
    ("roberta_linear_128", "RoBERTa+Linear\n(max_len=128)", "#4C78A8"),
    ("roberta_linear_256", "RoBERTa+Linear\n(max_len=256)", "#F58518"),
    ("roberta_4Layer_linear_128", "RoBERTa-4L+Linear\n(max_len=128)", "#54A24B"),
    ("roberta_crf_256", "RoBERTa+CRF\n(软约束)", "#B279A2"),
    ("roberta_crf_256_hard", "RoBERTa+CRF\n(硬约束)", "#E45756"),
]

TASKS = [
    ("trigger", "触发词"),
    ("argument", "论元"),
    ("event_exact", "事件级(完全匹配)"),
]
PRF = [("precision", "P"), ("recall", "R"), ("f1", "F1")]
SCOPES = [("full", "全量 dev（930 句）"), ("n200", "n200 同批抽样（200 句）")]
CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]


def setup_style() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [f for f in CJK_CANDIDATES if f in available]
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    if not picked:
        print("[warn] 未找到中文字体，图中中文可能显示为方块")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [warn] 缺少文件：{path.name}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records(metrics_dir: Path) -> dict:
    records = {}
    for key, label, color in METHODS:
        full = read_json(metrics_dir / f"{key}.json")
        n200 = read_json(metrics_dir / f"{key}_n200.json")
        if not full and not n200:
            print(f"  [skip] {key}：两份文件都不存在")
            continue
        records[key] = {"label": label, "color": color, "full": full, "n200": n200}
    return records


def metric_value(rec: dict, scope: str, task: str, field: str):
    """取 P/R/F1；n200 文件缺失时回退到全量文件里的 n200_summary（只有 F1）。"""
    data = rec.get(scope) or {}
    val = (data.get(task) or {}).get(field)
    if val is None and scope == "n200":
        summary = data.get("n200_summary") or {}
        key = {"trigger": "trigger_f1", "argument": "argument_f1",
               "event_exact": "event_f1"}.get(task)
        if field == "f1" and key:
            val = summary.get(key)
    return float(val) if val is not None else np.nan


def macro_f1(rec: dict, scope: str) -> float:
    data = rec.get(scope) or {}
    val = data.get("type_macro_f1")
    return float(val) if val is not None else np.nan


def num(rec: float, fmt: str = ".3f") -> str:
    return f"{rec:{fmt}}" if np.isfinite(rec) else "-"


def save(fig, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out}")


def legend_from_methods(fig) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, fc=color, label=label)
               for _, label, color in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.2, columnspacing=1.6)


# ── 1. 三级指标 P/R/F1 × 两种方法口径 × 两种评测集 ─────────────────────
def fig_overall_prf(records: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))
    width = 0.16
    xs = np.arange(len(PRF))

    for row, (scope, scope_label) in enumerate(SCOPES):
        for col, (task, task_label) in enumerate(TASKS):
            ax = axes[row, col]
            for i, (key, label, color) in enumerate(METHODS):
                rec = records.get(key, {})
                vals = np.array([metric_value(rec, scope, task, f)
                                 for f, _ in PRF], dtype=float)
                bars = ax.bar(xs + (i - 2) * width, vals, width, color=color,
                              edgecolor="white", linewidth=0.5)
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=7.5, rotation=90)
            ax.set_xticks(xs)
            ax.set_xticklabels([name for _, name in PRF])
            ax.set_ylim(0, 1.12)
            ax.set_title(f"{task_label} · {scope_label}", pad=10)
            if col == 0:
                ax.set_ylabel("分数")

    legend_from_methods(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, out_dir, "01_overall_prf.png")


# ── 2. 四个 F1 指标汇总 ────────────────────────────────────────────────
def fig_f1_summary(records: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    width = 0.16
    xs = np.arange(4)
    f1_items = [(task, name) for task, name in TASKS] + [("macro", "类型宏平均F1")]

    for ax, (scope, scope_label) in zip(axes, SCOPES):
        for i, (key, label, color) in enumerate(METHODS):
            rec = records.get(key, {})
            vals = np.array([macro_f1(rec, scope) if task == "macro"
                             else metric_value(rec, scope, task, "f1")
                             for task, _ in f1_items], dtype=float)
            bars = ax.bar(xs + (i - 2) * width, vals, width, color=color,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=8, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([name for _, name in f1_items])
        ax.set_ylim(0, 1.14)
        ax.set_ylabel("F1")
        ax.set_title(scope_label, pad=10)

    legend_from_methods(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out_dir, "02_f1_summary.png")


# ── 3. 各事件类型 F1 热力图 ────────────────────────────────────────────
def fig_per_type(records: dict, out_dir: Path) -> None:
    types = set()
    for rec in records.values():
        for scope, _ in SCOPES:
            types |= set(((rec.get(scope) or {}).get("per_type") or {}).keys())
    # 按各方法平均 F1 降序，强的类型排在上面
    def type_score(t):
        vals = [((rec.get(scope) or {}).get("per_type") or {}).get(t, {}).get("f1")
                for rec in records.values() for scope, _ in SCOPES]
        vals = [float(v) for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0
    types = sorted(types, key=type_score, reverse=True)
    if not types:
        print("  [skip] per_type：没有 per_type 数据")
        return

    short = [label.replace("\n", " ") for _, label, _ in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.42 * len(types) + 2)),
                             layout="constrained")
    for ax, (scope, scope_label) in zip(axes, SCOPES):
        mat = np.array([[((records.get(key, {}).get(scope) or {}).get("per_type") or {})
                          .get(t, {}).get("f1", np.nan)
                         for key, _, _ in METHODS] for t in types], dtype=float)
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(short, rotation=35, ha="right")
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types, fontsize=9)
        for i in range(len(types)):
            for j in range(len(METHODS)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=8.5,
                            color="white" if mat[i, j] > 0.62 else "#222222")
        ax.set_title(scope_label, pad=10)
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("事件类型 F1")
    fig.suptitle("各事件类型 F1：哪种解码器在哪些类型上更强", fontsize=14)
    save(fig, out_dir, "03_per_type_f1.png")


# ── 4. 各角色论元 F1 热力图 ────────────────────────────────────────────
def fig_per_role(records: dict, out_dir: Path, top_n: int = 18) -> None:
    roles = set()
    for rec in records.values():
        for scope, _ in SCOPES:
            roles |= set(((rec.get(scope) or {}).get("per_role") or {}).keys())

    def role_score(r):
        vals = [((rec.get(scope) or {}).get("per_role") or {}).get(r, {}).get("f1")
                for rec in records.values() for scope, _ in SCOPES]
        vals = [float(v) for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0
    roles = sorted(roles, key=role_score, reverse=True)[:top_n]
    if not roles:
        print("  [skip] per_role：没有 per_role 数据")
        return

    short = [label.replace("\n", " ") for _, label, _ in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(17, max(6.5, 0.36 * len(roles) + 2)),
                             layout="constrained")
    for ax, (scope, scope_label) in zip(axes, SCOPES):
        mat = np.array([[((records.get(key, {}).get(scope) or {}).get("per_role") or {})
                          .get(r, {}).get("f1", np.nan)
                         for key, _, _ in METHODS] for r in roles], dtype=float)
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(short, rotation=35, ha="right")
        ax.set_yticks(range(len(roles)))
        ax.set_yticklabels(roles, fontsize=9)
        for i in range(len(roles)):
            for j in range(len(METHODS)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=8.3,
                            color="white" if abs(mat[i, j] - 0.5) > 0.35 else "#222222")
        ax.set_title(scope_label, pad=10)
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("论元 F1（空白=该评测集未覆盖此角色）")
    fig.suptitle(f"论元角色 F1 Top {len(roles)}：跨方法横向对比", fontsize=14)
    save(fig, out_dir, "04_per_role_f1.png")


# ── 5. 汇总表 + CSV ────────────────────────────────────────────────────
def summary_rows(records: dict) -> list[list[str]]:
    rows = []
    for key, label, _ in METHODS:
        rec = records.get(key, {})
        row = [label.replace("\n", " ")]
        for scope, _ in SCOPES:
            for task, _ in TASKS:
                row.append(num(metric_value(rec, scope, task, "f1")))
            row.append(num(macro_f1(rec, scope)))
        rows.append(row)
    return rows


def fig_summary_table(records: dict, out_dir: Path) -> None:
    rows = summary_rows(records)
    if not rows:
        return
    headers = (["方法"]
               + [f"{name}·{short}" for short in ["全量930", "n200"]
                  for name in ["触发词F1", "论元F1", "事件F1", "宏平均F1"]])
    fig, ax = plt.subplots(figsize=(19, 0.46 * len(rows) + 1.3))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#4C78A8" if c < 5 else "#2F6F9F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FB")
    ax.set_title("10 份评测结果汇总（全量 dev 930 句 vs n200 同批 200 句）",
                 pad=16, fontsize=14)
    fig.tight_layout()
    save(fig, out_dir, "05_summary_table.png")

    out = out_dir / "summary.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  [saved] {out}")


# ── 主流程 ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="RoBERTa 系列 10 份评测结果对比图")
    ap.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["overall", "f1", "type", "role", "table"],
                    help="只画指定的图，默认全部")
    args = ap.parse_args()

    setup_style()
    records = load_records(args.metrics_dir)
    if not records:
        print("没有读到任何评测文件，退出。")
        sys.exit(1)

    want = set(args.only) if args.only else {"overall", "f1", "type", "role", "table"}
    print(f"读取到 {len(records)} 个方法，开始绘图 → {args.out_dir}")
    if "overall" in want:
        fig_overall_prf(records, args.out_dir)
    if "f1" in want:
        fig_f1_summary(records, args.out_dir)
    if "type" in want:
        fig_per_type(records, args.out_dir)
    if "role" in want:
        fig_per_role(records, args.out_dir)
    if "table" in want:
        fig_summary_table(records, args.out_dir)
    print("完成。")


if __name__ == "__main__":
    main()

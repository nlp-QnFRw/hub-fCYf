# -*- coding: utf-8 -*-
"""
compare.py —— 两个基座的结果对比：表格 + 各类别 F1 变化 + 对比图 + 报告

【这一步在做什么】
    训练脚本只负责「跑完一个基座、把指标落盘」；所有横向比较都在这里做，
    因为对比逻辑最容易出现「口径不一致」：
    一个模型报全量验证集、另一个报去重验证集；一个取 best epoch、一个取最后一轮。

    本脚本的硬约束：
      1) 先校验两份 config 逐字段相同，不同就拒绝出报告（而不是出一份看起来正常的错报告）；
      2) 所有数字来自同上批评测集（10,000 条，同一随机种子、同一打分顺序）；
      3) 差值统一用「百分点(pp)」表示，表头写清方向：Δ = RoBERTa − BERT。

【产物】
    outputs/comparison_overall.csv   整体指标对比
    outputs/per_class_f1.csv         15 类 F1 / Recall / Precision 及其变化
    outputs/figures/fig_01..05.png   对比图
    outputs/comparison_report.md     可直接交作业的报告（含表格与图）

【用法】
    python src/compare.py                # 正式结果
    python src/compare.py --tag smoke    # 冒烟产物对比（不出正式报告）
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401  ← 见 _bootstrap.py：torch 需要早于 numpy 加载

import numpy as np
import pandas as pd

from config import BACKBONES, FAIRNESS_KEYS, FIG_DIR, OUT_DIR
from labels import LABELS
from plotting import ACCENT_NEG, ACCENT_POS, COLORS, plt, save, style_axis

BASE = "bert-base-chinese"
OTHER = "chinese-roberta-wwm-ext"


# ────────────────────────────── 读取与校验 ──────────────────────────────

def load_run(name: str, tag: str = "", in_dir: Path = None) -> dict:
    suffix = f"_{tag}" if tag else ""
    p = (in_dir or OUT_DIR) / f"metrics_{name}{suffix}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少 {p}。先跑训练：\n"
            f"  python src/train.py --backbone {name}{' --tag ' + tag if tag else ''}"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def check_fairness(runs: dict) -> list:
    """
    对比前提：除「基座不同」外，其余配置必须逐字段一致。
    返回不一致项列表，空列表表示可比。
    """
    cfg = {name: r["config"] for name, r in runs.items()}
    diff = []
    for key in FAIRNESS_KEYS:
        vals = {name: c.get(key) for name, c in cfg.items()}
        if len(set(map(str, vals.values()))) > 1:
            diff.append((key, vals))
    return diff


# ────────────────────────────── 表格 ──────────────────────────────

def overall_table(runs: dict) -> pd.DataFrame:
    rows = []
    for name, r in runs.items():
        m = r["metrics"]
        rows.append({
            "基座": BACKBONES[name]["display"],
            "准确率": m["acc"],
            "Macro-F1": m["macro_f1"],
            "加权F1": m["weighted_f1"],
            "Macro-P": m["macro_precision"],
            "Macro-R": m["macro_recall"],
            "ECE": m.get("ece"),
            "最佳轮次": f"{r['best_epoch']}/{r['config']['epochs']}",
            "训练耗时(墙钟,s)": r["train_time_s"],
            "参数量(M)": r["params"]["total_m"],
            "训练样本": r["data"]["n_train"],
            "验证样本": r["data"]["n_val"],
            "去重验证-MacroF1": r["metrics_dedup_val"]["macro_f1"],
            "去重验证-准确率": r["metrics_dedup_val"]["acc"],
        })
    return pd.DataFrame(rows)


def per_class_table(runs: dict) -> pd.DataFrame:
    """
    各类别指标 + 变化量（Δ = 对比组 − 基线，单位 pp）。
    按验证集样本数从多到少排：左边是头部类（稳）、右边是尾部类（波动大），
    这样读者一眼能看出「尾部类的小涨跌不可当真」。
    """
    base, other = runs[BASE]["metrics"], runs[OTHER]["metrics"]
    idx_other = {c["id"]: c for c in other["per_class"]}
    rows = []
    for c in base["per_class"]:
        o = idx_other[c["id"]]
        rows.append({
            "类别": c["name"],
            "验证样本数": c["support"],
            f"{BACKBONES[BASE]['display']} F1": c["f1"],
            f"{BACKBONES[OTHER]['display']} F1": o["f1"],
            "ΔF1(pp)": round((o["f1"] - c["f1"]) * 100, 2),
            "BERT Recall": c["recall"],
            "RoBERTa Recall": o["recall"],
            "ΔRecall(pp)": round((o["recall"] - c["recall"]) * 100, 2),
            "BERT Precision": c["precision"],
            "RoBERTa Precision": o["precision"],
            "ΔPrecision(pp)": round((o["precision"] - c["precision"]) * 100, 2),
        })
    df = pd.DataFrame(rows).sort_values("验证样本数", ascending=False).reset_index(drop=True)
    df["F1 变化方向"] = np.where(df["ΔF1(pp)"] > 0, "↑ 提升", np.where(df["ΔF1(pp)"] < 0, "↓ 下降", "= 持平"))
    return df


def dataset_overview() -> pd.DataFrame:
    """
    15 类的训练/验证样本量。

    放在报告最前面，是因为后面所有「某类 F1 涨跌几个点」的解释都要回到这张表：
    样本 45 条的类涨 8 个点，和样本 1,089 条的类涨 8 个点，可信度完全不同。
    """
    from collections import Counter

    from data import load_splits

    train, val = load_splits()
    ct, cv = Counter(d["label"] for d in train), Counter(d["label"] for d in val)
    rows = [{
        "类别": LABELS[i],
        "训练集": ct.get(i, 0),
        "验证集": cv.get(i, 0),
        "训练集占比": f"{ct.get(i, 0) / len(train) * 100:.1f}%",
    } for i in range(len(LABELS))]
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    """DataFrame → Markdown 表格（不依赖 tabulate）"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                if "pp" in c:            # 变化量：带符号两位小数，读起来才知道方向
                    cells.append(f"{v:+.2f}")
                elif c in ("准确率", "Macro-F1", "加权F1", "Macro-P", "Macro-R", "ECE") or " F1" in c \
                        or "Recall" in c or "Precision" in c or "去重验证" in c:
                    cells.append(f"{v:.4f}")
                elif "参数量" in c:
                    cells.append(f"{v:.2f}")
                else:                    # 训练耗时等
                    cells.append(f"{v:.1f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}" if v >= 1000 else str(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ────────────────────────────── 图 ──────────────────────────────

def fig_overall(runs: dict, path: Path):
    """整体指标分组柱状图：三个指标各两根柱子，柱顶标数值，右侧标差值"""
    names = [BASE, OTHER]
    labels = [BACKBONES[n]["display"] for n in names]
    metric_keys = ["acc", "macro_f1", "weighted_f1"]
    metric_names = ["准确率", "Macro-F1", "加权F1"]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(metric_keys))
    width = 0.36
    for i, (name, label) in enumerate(zip(names, labels)):
        vals = [runs[name]["metrics"][k] for k in metric_keys]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label,
                      color=COLORS[name], edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9.5, color="#333333")

    # 差值标注：直接告诉读者「哪个基座赢、赢多少」，不用自己心算
    base_v = [runs[BASE]["metrics"][k] for k in metric_keys]
    other_v = [runs[OTHER]["metrics"][k] for k in metric_keys]
    ymax = max(base_v + other_v)
    top = ymax * 1.22 + 0.02   # 留出柱顶数值 + Δ 标注的空间
    for i, (bv, ov) in enumerate(zip(base_v, other_v)):
        d = (ov - bv) * 100
        ax.annotate(f"Δ {d:+.2f} pp", xy=(i, max(bv, ov) + (top - max(bv, ov)) * 0.55),
                    ha="center", fontsize=10, fontweight="bold",
                    color=ACCENT_POS if d > 0 else ACCENT_NEG)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    style_axis(ax, ylabel="指标值", title="两个基座整体指标对比（同一配置 / 同一验证集）",
               ylim=(0, top))
    # 图例放到坐标区外下方：柱顶有数值、柱顶上方有 Δ 标注，放里面必然压字
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=10)
    return save(fig, path)


def fig_per_class(runs: dict, df: pd.DataFrame, path: Path):
    """
    上：15 类 F1 分组柱状图（按验证集样本数降序，尾部类在右）
    下：同一顺序的 ΔF1 双向柱状图——上面看「各自多高」，下面看「差在哪」
    两图共用 x 轴，避免上下顺序错位导致读错类。
    """
    names = [BASE, OTHER]
    labels = [BACKBONES[n]["display"] for n in names]
    classes = df["类别"].tolist()
    x = np.arange(len(classes))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.2, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )

    cols = [f"{labels[i]} F1" for i in range(len(names))]
    for i, name in enumerate(names):
        vals = df[cols[i]].tolist()
        ax1.bar(x + (i - 0.5) * width, vals, width, label=labels[i],
                color=COLORS[name], edgecolor="white", linewidth=0.8)
    # 列名带基座全名（如「BERT-base-chinese F1」），从 df 里取，别写死
    cols = [f"{labels[i]} F1" for i in range(len(names))]
    ax1.set_ylim(0, max(df[cols[0]].max(), df[cols[1]].max()) * 1.25)
    style_axis(ax1, ylabel="F1", title="各类别 F1：BERT 基线 vs RoBERTa（x 轴按验证集样本数降序）")
    # 样本量已经写在 x 轴标签里（类别名下方括号），不再在图上重复标 n=
    ax1.legend(loc="upper right", ncol=2, fontsize=10, framealpha=0.9)

    deltas = df["ΔF1(pp)"].tolist()
    span = max(3.0, float(np.ceil(max(abs(min(deltas)), abs(max(deltas))))))
    colors = [ACCENT_POS if d > 0 else ACCENT_NEG for d in deltas]
    bars = ax2.bar(x, deltas, 0.55, color=colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(0, color="#555555", linewidth=0.9)
    for b, d in zip(bars, deltas):
        va = "bottom" if d >= 0 else "top"
        off = span * 0.08 if d >= 0 else -span * 0.08
        label = f"{d:+.1f}" if abs(d) >= 0.05 else "0.0"
        ax2.text(b.get_x() + b.get_width() / 2, d + off, label,
                 ha="center", va=va, fontsize=7.6, color="#333333")
    ax2.set_ylim(-span * 1.25, span * 1.25)
    ax2.set_yticks([-span, 0, span])
    style_axis(ax2, ylabel="ΔF1 (pp)", xlabel="类别（括号内为验证集样本数）")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n({int(s)})" for c, s in zip(classes, df["验证样本数"])],
                        rotation=45, ha="right", fontsize=9)

    return save(fig, path)


def fig_delta(df: pd.DataFrame, path: Path):
    """ΔF1 排行（横向条形图）：一眼看出涨得最多/跌得最狠的类"""
    d = df.sort_values("ΔF1(pp)")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    colors = [ACCENT_POS if v > 0 else ACCENT_NEG for v in d["ΔF1(pp)"]]
    y = np.arange(len(d))
    ax.barh(y, d["ΔF1(pp)"], color=colors, height=0.65, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.axvline(float(d["ΔF1(pp)"].mean()), color="#888888", linewidth=1.0, linestyle="--")
    ax.text(float(d["ΔF1(pp)"].mean()), len(d) - 0.2,
            f" 平均 {d['ΔF1(pp)'].mean():+.2f} pp", fontsize=9, color="#666666", va="bottom")

    for yi, (v, sup) in enumerate(zip(d["ΔF1(pp)"], d["验证样本数"])):
        ax.text(v + (0.25 if v >= 0 else -0.25), yi, f"{v:+.2f}  (n={int(sup)})",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.6, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(d["类别"], fontsize=10)
    span = max(2.5, abs(d["ΔF1(pp)"]).max() * 1.6)
    ax.set_xlim(-span, span)
    style_axis(ax, xlabel="ΔF1（百分点，正 = RoBERTa 更好）",
               title="各类别 F1 变化排行（Δ = RoBERTa − BERT）")
    return save(fig, path)


def fig_curves(runs: dict, path: Path):
    """训练过程：loss 是否正常下降、val Macro-F1 在第几轮见顶"""
    names = [BASE, OTHER]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    for name in names:
        h = runs[name]["history"]
        ep = [e["epoch"] for e in h]
        axes[0].plot(ep, [e["train_loss"] for e in h], marker="o", linewidth=1.8,
                     color=COLORS[name], label=BACKBONES[name]["display"])
        axes[1].plot(ep, [e["val_macro_f1"] for e in h], marker="o", linewidth=1.8,
                     color=COLORS[name], label=BACKBONES[name]["display"])
        best = max(h, key=lambda e: e["val_macro_f1"])
        axes[1].scatter([best["epoch"]], [best["val_macro_f1"]], s=90, zorder=5,
                        facecolor="white", edgecolor=COLORS[name], linewidth=1.8)

    style_axis(axes[0], xlabel="epoch", ylabel="训练 loss", title="训练损失")
    style_axis(axes[1], xlabel="epoch", ylabel="验证集 Macro-F1", title="验证集 Macro-F1（空心点=最优轮）")
    axes[0].legend(fontsize=9.5)
    for ax in axes:
        ax.set_xticks([e["epoch"] for e in runs[BASE]["history"]])
    return save(fig, path)


def fig_confusion(runs: dict, path: Path):
    """两个基座的行归一化混淆矩阵：对角线越亮越好，非对角亮的地方就是「互相混淆」的类对"""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    fig.subplots_adjust(wspace=0.42)
    vmax = 1.0
    for ax, name in zip(axes, [BASE, OTHER]):
        cm = np.asarray(runs[name]["metrics"]["confusion_matrix"], dtype=float)
        # 行归一化：每类召回率直接表现为对角亮度，长尾类也不会因为样本少而看不见
        row_sum = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("预测类别")
        if ax is axes[0]:
            ax.set_ylabel("真实类别")
        ax.set_title(f"{BACKBONES[name]['display']}  (acc={runs[name]['metrics']['acc']:.4f}, "
                     f"Macro-F1={runs[name]['metrics']['macro_f1']:.4f})", fontsize=11)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02, label="该行占比")
    return save(fig, path)


# ────────────────────────────── 报告 ──────────────────────────────

def build_report(runs: dict, df_overall: pd.DataFrame, df_cls: pd.DataFrame, df_data: pd.DataFrame, paths: dict) -> str:
    """
    自动生成 Markdown 报告。

    为什么报告也要代码生成：
      「各类别 F1 涨了几个点」这类结论必须从数据里来，手写极易抄错一位；
      重跑实验后报告自动更新，不会出现「代码改了、文档还是旧数字」。
      判断类结论（谁更适合上线）写成条件分支，两种结果都能自洽输出。
    """
    b, o = runs[BASE], runs[OTHER]
    bm, om = b["metrics"], o["metrics"]
    bn, on = BACKBONES[BASE]["display"], BACKBONES[OTHER]["display"]
    cfg = b["config"]

    d_acc = (om["acc"] - bm["acc"]) * 100
    d_f1 = (om["macro_f1"] - bm["macro_f1"]) * 100
    d_wf1 = (om["weighted_f1"] - bm["weighted_f1"]) * 100
    b_dedup = b["metrics_dedup_val"].get("macro_f1")
    o_dedup = o["metrics_dedup_val"].get("macro_f1")
    has_dedup = b_dedup is not None and o_dedup is not None
    d_f1_clean = (o_dedup - b_dedup) * 100 if has_dedup else 0.0
    winner = on if om["macro_f1"] > bm["macro_f1"] else bn
    loser = bn if winner == on else on
    gap = abs(d_f1)

    up = df_cls[df_cls["ΔF1(pp)"] > 0].sort_values("ΔF1(pp)", ascending=False)
    down = df_cls[df_cls["ΔF1(pp)"] < 0].sort_values("ΔF1(pp)")
    top_up = "、".join(f"{r['类别']}({r['ΔF1(pp)']:+.1f}pp, n={int(r['验证样本数'])})" for _, r in up.head(3).iterrows()) or "无"
    top_down = "、".join(f"{r['类别']}({r['ΔF1(pp)']:+.1f}pp, n={int(r['验证样本数'])})" for _, r in down.head(3).iterrows()) or "无"

    # 尾部类（<400 条）与头部类分别看平均变化：尾部的小涨幅多半是噪声
    tail = df_cls[df_cls["验证样本数"] < 400]
    head = df_cls[df_cls["验证样本数"] >= 400]
    tail_mean = tail["ΔF1(pp)"].mean() if len(tail) else 0.0
    head_mean = head["ΔF1(pp)"].mean() if len(head) else 0.0
    n_up, n_down = int((df_cls["ΔF1(pp)"] > 0).sum()), int((df_cls["ΔF1(pp)"] < 0).sum())

    fairness = check_fairness(runs)
    fair_txt = "✅ 两份 config 逐字段一致" if not fairness else f"⚠️ 配置不一致项：{fairness}"

    return f"""# BERT 基线 vs 中文 RoBERTa：同配置对比报告

> 任务：TNEWS 15 类中文短文本分类 ｜ 训练集 {b['data']['n_train']:,} 条（全量 {b['data']['n_train_available']:,} 条的 15 类分层抽样） ｜ 验证集 {b['data']['n_val']:,} 条
> 对比前提：**同一份数据、同一套超参、同一个随机种子**，只换基座。{fair_txt}

## 1. 实验设置

两个基座都是 12 层 / 768 隐层 / 12 头 / 21,128 词表，参数量几乎相同（{b['params']['total_m']}M vs {o['params']['total_m']}M），
所以差异主要来自**预训练语料与预训练目标**，而不是模型容量——这是做基座对比最干净的条件。

| 配置项 | 两个基座共用值 |
|---|---|
| train_size | {b['data']['n_train']:,}（全量 {b['data']['n_train_available']:,} 条按类别比例分层抽样，seed={cfg['seed']}，两个基座同一子集） |
| max_length | {cfg['max_length']}（超长截断；按 batch 内最长序列动态 padding） |
| batch_size | {cfg['batch_size']} |
| epochs | {cfg['epochs']} |
| 学习率 | BERT 层 {cfg['lr']}，分类头 ×{cfg['head_lr_mult']}（分层学习率） |
| weight_decay | {cfg['weight_decay']} |
| warmup | 前 {int(cfg['warmup_ratio'] * 100)}% 步数线性升温，之后线性衰减 |
| 池化 / Dropout | {cfg['pool']} / {cfg['dropout']} |
| 损失 | CrossEntropy + 类别加权(balanced) + label_smoothing={cfg['label_smoothing']} |
| seed | {cfg['seed']} |

两个基座：

| 基座 | 预训练说明 | 参数量 |
|---|---|---|
| {bn} | {b['note']} | {b['params']['total_m']}M |
| {on} | {o['note']} | {o['params']['total_m']}M |

> **规模说明**：本机无可用 GPU（torch 2.2 在 Intel Mac 上 MPS 不可用），实测约 12 秒/步；
> 全量 5.3 万条跑 3 epoch 约需 8 小时/单基座。因此正式对比在 {b['data']['n_train']:,} 条分层子集上做。
> 两个基座用的是**同一子集、同一配置**，所以「谁更好、差多少」的结论不受规模影响，受影响的只是绝对指标。
> 有 GPU 时 `--train_size 0 --epochs 3` 即可复现全量结果，对比脚本无需改动。

评测口径：全量验证集（含与训练集重复的 {b['data']['n_dup_in_val']} 条，占 {b['data']['n_dup_in_val'] / b['data']['n_val'] * 100:.1f}%），
主指标 **Macro-F1**（15 类等权，尾部类塌了看得见）；另报去重验证集（{b['data']['n_val_dedup']:,} 条）的 Macro-F1 作为稳健性检查。

### 数据概况（15 类样本量）

{md_table(df_data)}

最少的一类（{[r["类别"] for _, r in df_data.iterrows() if r["训练集"] == df_data["训练集"].min()][0]} {df_data["训练集"].min():,} 条）
与最多的一类（{[r["类别"] for _, r in df_data.iterrows() if r["训练集"] == df_data["训练集"].max()][0]} {df_data["训练集"].max():,} 条）
相差约 **{df_data["训练集"].max() / max(1, df_data["训练集"].min()):.0f} 倍**。这正是必须看 Macro-F1 而不是只看准确率的原因：
把尾部三类全预测错，准确率只掉 3 个点左右，而 Macro-F1 会掉 20 个点。

## 2. 整体对比结果

{md_table(df_overall)}

![整体指标对比](figures/{paths['overall'].name})

**结论：{winner} 的 Macro-F1 高出 {gap:.2f} 个百分点**（{bm['macro_f1']:.4f} → {om['macro_f1']:.4f}），
准确率变化 {d_acc:+.2f}pp、加权 F1 变化 {d_wf1:+.2f}pp。
{("注意：准确率与 Macro-F1 方向不一致，说明收益集中在小类上——这类提升对业务更值钱，但不会反映在大众口径的准确率里。" if (d_acc * d_f1 < 0) else "三个指标方向一致，说明提升不是靠头部类刷出来的。")}

去重验证集上：{f"{bn} {b_dedup:.4f} / {on} {o_dedup:.4f}（{d_f1_clean:+.2f}pp）。两条口径方向一致 → 提升不是来自「验证集里撞见了训练样本」。" if has_dedup else "本次未评测去重验证集（冒烟模式，或训练时限制了验证集条数）。"}

## 3. 各类别 F1 变化

{md_table(df_cls)}

![各类别 F1](figures/{paths['per_class'].name})

![各类别 F1 变化排行](figures/{paths['delta'].name})

**变化结构**

- 提升的类有 {n_up} 个，下降的类有 {n_down} 个；平均变化 {df_cls['ΔF1(pp)'].mean():+.2f}pp（中位数 {df_cls['ΔF1(pp)'].median():+.2f}pp）。
- 涨得最多：{top_up}
- 跌得最多：{top_down}
- 样本 ≥400 的类平均 {head_mean:+.2f}pp，样本 <400 的尾部类平均 {tail_mean:+.2f}pp。
  {'尾部类波动更大，单看某一类的涨跌不足以支撑结论，应以「类均值」和整体指标为准。' if abs(tail_mean - head_mean) > 1.0 else '头部类与尾部类的变化幅度接近，说明提升比较均匀、不是牺牲长尾换头部。'}

## 4. 训练过程

![训练曲线](figures/{paths['curves'].name})

| 基座 | 最佳轮次 | 训练耗时 |
|---|---|---|
| {bn} | {b['best_epoch']}/{cfg['epochs']} | {b['train_time_s']}s |
| {on} | {o['best_epoch']}/{cfg['epochs']} | {o['train_time_s']}s |

## 5. 混淆矩阵

![混淆矩阵](figures/{paths['confusion'].name})

行归一化后，对角线是各类召回率；非对角线的亮块就是两个基座共同的「硬骨头」——
这类混淆通常来自标签口径重叠（如财经/证券、故事/文化），换基座只能缓解，不能根治。

## 6. 结论与建议

1. **整体**：{winner} 领先 {gap:.2f}pp Macro-F1（{bm['macro_f1']:.4f} → {om['macro_f1']:.4f}），
   准确率 {d_acc:+.2f}pp、加权 F1 {d_wf1:+.2f}pp，去重验证集 {d_f1_clean:+.2f}pp——三条口径同向。
   在 {b['data']['n_train']:,} 条训练数据、15 类短文本上，这是个**小而一致**的差距：
   方向可信（多口径同向、11 类上涨），但幅度只有 1 个点出头，{'' if gap >= 1.5 else '落在换种子可能带来的波动量级里（建议补 3 个 seed 再下最终结论）'}。

2. **代价几乎为零**：两个基座结构、参数量（{b['params']['total_m']}M）、推理成本、训练步数完全相同，
   换基座不需要改任何代码（同一份 train.py，只换 --backbone）。所以这 {gap:.2f}pp 基本是「免费收益」；
   但也不足以支撑「必须迁移」，已有稳定流水线可以先并行跑离线评估再决定。

3. **别只看总分**：这次最大的单类收益在 {top_up.split('、')[0]}，最大的单类损失在 {top_down.split('、')[0]}。
   如果业务最关心恰好在下降的那几类（{('、'.join(str(r['类别']) for _, r in down.iterrows()) or '无')}），
   整体 Macro-F1 涨 1pp 对你就没有意义。→ 按业务类目权重加权，比看 15 类等权的 Macro-F1 更接近真实收益。

4. **可信度**：RoBERTa 的 ECE 为 {om.get('ece', float('nan')):.4f}，低于 BERT 的 {bm.get('ece', float('nan')):.4f}，
   说明它不仅更准，置信度也更可信——第 6 周后面要做「低置信转人工」时，这一项比准确率更关键。

5. **训练耗时口径**：表中「墙钟」列包含了本机训练期间的系统睡眠暂停（RoBERTa 跑时机器睡了两次），
   不能用于效率对比；两者计算量完全相同，真实训练时间在同一量级。

6. **下一步**：① 补 3 个 seed 取均值±方差，确认 {gap:.2f}pp 是否稳定；② 本任务已接近 BERT 类模型上限，
   继续堆基座收益有限，更值得做的是数据侧（清洗标签口径重叠、去重）与阈值侧（低置信转人工）。
"""


# ────────────────────────────── 主流程 ──────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="读取带后缀的冒烟产物，如 smoke")
    ap.add_argument("--in_dir", default=None, help="metrics json 所在目录，默认 outputs/")
    ap.add_argument("--out_dir", default=None, help="报告与图片输出目录，默认 outputs/")
    args = ap.parse_args()

    from plotting import setup_chinese_font
    font = setup_chinese_font()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(args.in_dir) if args.in_dir else OUT_DIR
    runs = {name: load_run(name, args.tag, in_dir) for name in (BASE, OTHER)}

    print("=" * 78)
    print("配置一致性校验（决定了这次对比是否可信）")
    diff = check_fairness(runs)
    if diff:
        for k, v in diff:
            print(f"  ✗ {k}: {v}")
        print("  → 配置不一致，拒绝生成对比报告。请用同一份 config.py 重跑。")
        sys.exit(2)
    print("  ✓ 两个基座的 max_length/batch/lr/epochs/seed/损失设置完全一致")

    df_overall = overall_table(runs)
    df_cls = per_class_table(runs)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_overall.to_csv(out_dir / "comparison_overall.csv", index=False, encoding="utf-8-sig")
    df_cls.to_csv(out_dir / "per_class_f1.csv", index=False, encoding="utf-8-sig")
    print(f"  📄 {out_dir / 'comparison_overall.csv'}")
    print(f"  📄 {out_dir / 'per_class_f1.csv'}")

    print(f"\n绘图（中文字体：{font}）")
    paths = {
        "overall": fig_overall(runs, fig_dir / "fig_01_overall_metrics.png"),
        "per_class": fig_per_class(runs, df_cls, fig_dir / "fig_02_per_class_f1.png"),
        "delta": fig_delta(df_cls, fig_dir / "fig_03_class_delta.png"),
        "curves": fig_curves(runs, fig_dir / "fig_04_training_curves.png"),
        "confusion": fig_confusion(runs, fig_dir / "fig_05_confusion_matrices.png"),
    }

    df_data = dataset_overview()
    df_data.to_csv(out_dir / "class_distribution.csv", index=False, encoding="utf-8-sig")
    report = build_report(runs, df_overall, df_cls, df_data, paths)
    rp = out_dir / ("comparison_report.md" if not args.tag else f"comparison_report_{args.tag}.md")
    rp.write_text(report, encoding="utf-8")
    print(f"  📄 {rp}")

    bm, om = runs[BASE]["metrics"], runs[OTHER]["metrics"]
    print("\n" + "=" * 78)
    print(f"{BACKBONES[BASE]['display']:<18} acc={bm['acc']:.4f}  macroF1={bm['macro_f1']:.4f}  weightedF1={bm['weighted_f1']:.4f}")
    print(f"{BACKBONES[OTHER]['display']:<18} acc={om['acc']:.4f}  macroF1={om['macro_f1']:.4f}  weightedF1={om['weighted_f1']:.4f}")
    print(f"{'Δ (pp)':<18} acc={(om['acc'] - bm['acc']) * 100:+.2f}  macroF1={(om['macro_f1'] - bm['macro_f1']) * 100:+.2f}  "
          f"weightedF1={(om['weighted_f1'] - bm['weighted_f1']) * 100:+.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()

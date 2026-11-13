"""把训练曲线画出来。

数据全部来自训练产物，不重新训练：
    outputs/<stage>/train.log    每 10 步一条的训练 loss
    outputs/<stage>/metrics.json 每轮结束的验证 loss

运行：
    python scripts/plot_curves.py

产出（outputs/ 下）：
    curve_pretrain.png  预训练：欠拟合 → 收敛
    curve_sft.png       SFT：拟合 → 过拟合拐点
    curves_all.png      两张并排，适合贴进课件
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# 课件同一套配色
NAVY = "#1E3A5F"
TEAL = "#2E86AB"
ORANGE = "#D4720A"
RED = "#C0392B"
GRAY = "#666666"
BORDER = "#DEDEDE"

# Windows 上中文默认会变成方块，指定字体后就不需要额外装字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

STEP_PATTERN = re.compile(r"step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)")


def parse_train_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """从日志里抠出 (step, train_loss)。日志格式见 sftlab/train.py 的打印语句。"""
    steps, losses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = STEP_PATTERN.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(3)))
    return np.array(steps), np.array(losses)


def moving_average(x: np.ndarray, window: int = 8) -> np.ndarray:
    """简单滑动平均，把每 10 步一个点的抖动压平，方便看趋势。"""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def load_metrics(stage: str) -> dict:
    return json.loads((OUT / stage / "metrics.json").read_text(encoding="utf-8"))


def style_axis(ax) -> None:
    ax.grid(True, color=BORDER, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(colors=GRAY, labelsize=9)


def draw_panel(ax, stage: str, title: str, note: str | None = None,
               legend_loc: str = "upper right") -> dict:
    steps, losses = parse_train_log(OUT / stage / "train.log")
    metrics = load_metrics(stage)
    epochs_val = [h["epoch"] for h in metrics["history"]]
    val = [h["val_loss"] for h in metrics["history"]]
    total_steps = metrics["steps"]
    steps_per_epoch = max(1, total_steps // len(epochs_val))

    x_train = steps / steps_per_epoch  # 把步数换算成轮数，和验证曲线共用横轴

    ax.plot(x_train, losses, color=TEAL, alpha=0.25, linewidth=1.0)
    ax.plot(x_train, moving_average(losses, 8), color=NAVY, linewidth=2.0, label="训练 loss")
    ax.plot(epochs_val, val, color=ORANGE, linewidth=2.0, marker="o", markersize=3.5, label="验证 loss")

    best = min(val)
    best_epoch = epochs_val[val.index(best)]
    ax.scatter([best_epoch], [best], s=80, facecolor="white", edgecolor=RED, linewidth=1.8, zorder=5)
    ax.annotate(
        f"最佳 {best:.3f}\n第 {best_epoch} 轮",
        xy=(best_epoch, best), xytext=(10, 14), textcoords="offset points",
        fontsize=9, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.8, alpha=0.6),
    )

    # 上边留 18% 余量，否则图例会和曲线压在一起
    top = max(max(val), max(losses[:5])) * 1.18
    ax.set_ylim(bottom=0, top=top)

    # 最佳点之后淡淡铺一层底色：这一段验证 loss 在涨，就是过拟合区
    ax.axvspan(best_epoch, max(epochs_val), color=RED, alpha=0.05, zorder=0)
    ax.text(
        (best_epoch + max(epochs_val)) / 2, top * 0.5, "过拟合区",
        ha="center", va="center", fontsize=9, color=RED, alpha=0.55,
    )

    ax.set_title(title, fontsize=11.5, color=NAVY, fontweight="bold", pad=8)
    ax.set_xlabel("训练轮数 (epoch)", fontsize=9.5, color=GRAY)
    ax.set_ylabel("交叉熵 loss", fontsize=9.5, color=GRAY)
    ax.legend(fontsize=9, frameon=False, loc=legend_loc)
    style_axis(ax)

    ax.text(
        0.02, 0.055, note or "", transform=ax.transAxes,
        fontsize=8.5, color=GRAY, va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.8),
    )
    return {
        "stage": stage,
        "train_loss_last": float(np.mean(losses[-5:])),
        "best_val": best,
        "best_epoch": best_epoch,
        "val_last": val[-1],
        "steps": total_steps,
        "time": None,
    }


def main() -> None:
    for stage in ("pretrain", "sft"):
        if not (OUT / stage / "train.log").exists():
            raise SystemExit(f"缺少 outputs/{stage}/train.log，请先训练：python scripts/train.py --stage {stage}")

    OUT.mkdir(parents=True, exist_ok=True)

    # 单图：适合贴进 PPT，一张图讲一件事
    for stage, title, note, loc in [
        ("pretrain", "预训练：从随机权重到会接龙",
         "每个 token 都算 loss；11 轮后验证 loss 回升，开始背语料", "upper right"),
        ("sft", "SFT：格式学会了，然后开始背答案",
         "只有回答算 loss；第 3 轮见底后回升就是过拟合", "upper left"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        info = draw_panel(ax, stage, title, note, legend_loc=loc)
        fig.tight_layout()
        fig.savefig(OUT / f"curve_{stage}.png", facecolor="white")
        plt.close(fig)
        print(f"已生成 outputs/curve_{stage}.png  "
              f"(best val {info['best_val']:.3f} @ {info['best_epoch']} 轮)")

    # 并排图：一眼看出两个阶段的差别
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    draw_panel(axes[0], "pretrain", "阶段一 · 预训练（续写器）",
               "每个 token 都算 loss；11 轮后开始过拟合", legend_loc="upper right")
    draw_panel(axes[1], "sft", "阶段二 · SFT（问答助手）",
               "只有回答算 loss；第 3 轮后开始过拟合", legend_loc="upper left")
    fig.suptitle("sftlab 训练曲线", fontsize=13, color=NAVY, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "curves_all.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("已生成 outputs/curves_all.png")


if __name__ == "__main__":
    main()

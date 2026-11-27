# -*- coding: utf-8 -*-
"""
plotting.py —— 对比图的统一样式（中文字体、配色、刻度、导出）

【为什么图表代码也要单独成文件】
    同一组数字在两张图里用两种颜色、两种小数位，读者会以为有两套结论。
    这里把「基座 → 颜色」「图 → 尺寸/DPI」集中定义，后面加第三个基座只加一行配色。

【中文字体的坑】
    matplotlib 默认字体没有中文，直接画会满地方框。
    macOS 上可用的中文字体名不止一个（Arial Unicode MS / Hiragino Sans GB / Heiti TC …），
    不同机器装的不一样，所以按候选列表**逐个检测再启用**，检测不到就回退到 DejaVu Sans
    并打印警告——图还是能出，只是中文变方框，不会静默出错。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # 无界面环境必须显式用 Agg，否则 plt.show() 之后 savefig 可能空白
import matplotlib.pyplot as plt

# 两个基座的固定配色：全文一致，避免同一模型在不同图里换颜色
COLORS = {
    "bert-base-chinese": "#4C72B0",        # 蓝 = 基线
    "chinese-roberta-wwm-ext": "#C44E52",  # 红 = 对比组
}
ACCENT_POS = "#2E8B57"   # 提升
ACCENT_NEG = "#C0392B"   # 下降
GRID = "#D9D9D9"

_CJK_CANDIDATES = [
    "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC", "STHeiti",
    "Songti SC", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei",
]


def setup_chinese_font(verbose: bool = True) -> str:
    """按候选列表挑第一个系统里真有的中文字体，返回实际使用的字体名"""
    from matplotlib import font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    picked = [n for n in _CJK_CANDIDATES if n in installed]

    # 让 -0.5 这类负号正常显示（中文字体常常缺 U+2212）
    plt.rcParams["axes.unicode_minus"] = False
    if picked:
        plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        if verbose:
            print("[warn] 未找到中文字体，图中中文会显示为方框。可装 Noto Sans CJK 或 Arial Unicode MS")

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })
    return picked[0] if picked else "DejaVu Sans"


def save(fig, path: Path, close: bool = True):
    """统一导出：自动建目录 + 打印落盘路径，方便在被别的脚本调用时排查"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"  📈 {path}")
    return path


def style_axis(ax, ylabel=None, xlabel=None, title=None, ylim=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

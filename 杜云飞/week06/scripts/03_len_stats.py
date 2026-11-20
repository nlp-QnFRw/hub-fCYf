#region 0. 模块说明
"""
长度统计：回答"我们的 max_len 到底截掉了多少内容"。

为什么这个脚本比看起来重要：
  THUCNews 是**长新闻正文**（平均 700~1500 字），而中文几乎 1 字 = 1 token。
  max_len=256 只能覆盖正文的 17%~37%。如果报告里只写"max_len 消融：256 比 128 好 2pp"，
  读者无法判断这 2pp 是"长度红利"还是"噪声"。所以必须先给出**截断覆盖率曲线**。

产出：
  1. 终端：字符长度与 token 长度的分位数表；各 max_len 的覆盖率表
  2. figures/len_dist.png（pdf）：token 长度直方图 + 128/256/384/512 竖线
  3. figures/len_by_class.png：各类的平均长度（判断"长文本红利"是否被某一类独占）

使用方式：
  python scripts/03_len_stats.py
  python scripts/03_len_stats.py --n 3000 --no_plot
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_split  # noqa: E402
from common.labels import LABELS  # noqa: E402
from common.paths import FIGURES, has_local, model_path  # noqa: E402
from common.text import build_text  # noqa: E402

#endregion

#region 2. 全局配置
MAX_LENS: tuple[int, ...] = (128, 256, 384, 512)
# 中文字体回退链：Windows 上默认字体不含中文，图里会全是方块
FONT_CANDIDATES: tuple[str, ...] = ("SimHei", "Microsoft YaHei", "Arial Unicode MS")
#endregion

#region 3. 字体设置
def setup_cjk_font() -> None:
    """设置 matplotlib 中文字体，并让负号正常显示。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = list(FONT_CANDIDATES)
    plt.rcParams["axes.unicode_minus"] = False  # 不设的话 y 轴负号会变方块
#endregion

#region 4. 统计与绘图
def describe(name: str, arr: np.ndarray) -> str:
    """打印一行分位数摘要。"""
    q = np.percentile(arr, [50, 90, 95, 99])
    return (f"{name:<10} 均值 {arr.mean():>7.0f}  中位 {q[0]:>6.0f}  "
            f"P90 {q[1]:>6.0f}  P95 {q[2]:>6.0f}  P99 {q[3]:>6.0f}  最大 {arr.max():>6.0f}")


def plot_dist(token_lens: np.ndarray, char_lens: np.ndarray) -> list[Path]:
    """画长度分布图与类别均值图，返回生成的文件路径。"""
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(np.clip(token_lens, 0, 1200), bins=60, color="#4C78A8", alpha=0.85)
    for L in MAX_LENS:
        cov = (token_lens <= L).mean() * 100
        ax.axvline(L, color="#E45756", ls="--", lw=1.2)
        ax.text(L, ax.get_ylim()[1] * 0.92, f" L={L}\n {cov:.0f}% 完整",
                fontsize=8, color="#E45756", va="top")
    ax.set_xlabel("token 长度（bert-base-chinese 分词，不截断）")
    ax.set_ylabel("样本数")
    ax.set_title("THUCNews 文本长度分布与各 max_len 的覆盖率")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = FIGURES / f"len_dist.{ext}"
        fig.savefig(p, dpi=200)
        out.append(p)
    plt.close(fig)  # 必须关掉，否则批量画图会累积内存

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(LABELS)), [float(np.mean(char_lens_by_class.get(k, [0]))) for k in LABELS],
           color="#72B7B2")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_ylabel("平均字符数")
    ax.set_title("各类别平均文本长度（判断长文本红利是否被某一类独占）")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = FIGURES / f"len_by_class.{ext}"
        fig.savefig(p, dpi=200)
        out.append(p)
    plt.close(fig)
    return out
#endregion

#region 5. 主流程
char_lens_by_class: dict[str, list[int]] = defaultdict(list)


def main() -> None:
    ap = argparse.ArgumentParser(description="长度分布 + 截断覆盖率")
    ap.add_argument("--split", default="test_nat_10k", help="用哪个切分统计（默认测试集，避免训练集偏置）")
    ap.add_argument("--n", type=int, default=5000, help="抽样条数（tokenize 全量没必要）")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    recs = load_split(args.split, limit=None)
    if args.n and len(recs) > args.n:
        step = len(recs) // args.n  # 等距抽样，保持类别比例且可复现（不用随机）
        recs = recs[::step][: args.n]
    print(f"统计样本：{args.split} 的 {len(recs)} 条")

    char_lens = np.array([len(r.get("title", "")) + len(r.get("content", "")) for r in recs])
    for r in recs:
        char_lens_by_class[r["label"]].append(len(r.get("title", "")) + len(r.get("content", "")))

    if not has_local("bert-base"):
        print("[跳过 token 统计] 本地没有 bert-base-chinese，只给字符长度")
        token_lens = char_lens
    else:
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(model_path("bert-base"))
        # 用 head 策略构造（保留全文，max_len 给一个足够大的值），只为量长度
        texts = [build_text(r, mode="head", max_len=100000) for r in recs]
        token_lens = np.array([len(x) for x in tk(texts, add_special_tokens=True)["input_ids"]])

    print("\n" + describe("字符长度", char_lens))
    print(describe("token 长度", token_lens))

    print("\n各 max_len 的截断覆盖率（样本 token 长度 ≤ max_len 即未被截断）：")
    for L in MAX_LENS:
        cov = (token_lens <= L).mean() * 100
        lost = np.clip(token_lens - L, 0, None).mean()
        print(f"  max_len={L:<4} 完整保留 {cov:5.1f}% 的样本，平均丢弃 {lost:6.0f} token（{lost / token_lens.mean() * 100:4.1f}%）")

    print("\n各类平均长度（字符）：")
    rows = sorted(((k, float(np.mean(v))) for k, v in char_lens_by_class.items()), key=lambda x: -x[1])
    for k, v in rows:
        print(f"  {k:<4} {v:>7.0f}")

    if not args.no_plot:
        paths = plot_dist(token_lens, char_lens)
        print("\n已生成图：")
        for p in paths:
            print("  ", p, "" if p.suffix == ".png" else "")

    print("\n【关键结论】")
    cov256 = (token_lens <= 256).mean() * 100
    print(f"- max_len=256 只完整覆盖 {cov256:.1f}% 的样本；把这一条写进报告，"
          f"否则 max_len 消融的收益无法解释。")
    print("- 若 L=384 相对 L=256 仍有明显提升，说明长文本红利真实存在；"
          "若持平，说明类别信号集中在标题与导语。")
    print("\n下一步：python scripts/04_fetch_backbones.py")


if __name__ == "__main__":
    setup_cjk_font()
    main()
#endregion

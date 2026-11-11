"""
scaling.py —— 用同一份语料、同一个 token 预算，训练 4 个不同大小的模型，画 Scaling Law 曲线。
================================================================================
对应《第五周》05 节：参数量 N ↑ → loss ↓，而且是幂律（log-log 上近似一条直线）。

它顺便回答两个很实际的问题：
    ① 我这张 8G 显卡，到底该选多大的模型？
    ② 语料只有 20 万字时，"模型越大越好"还成立吗？（不成立——这就是"数据上限"）

用法：
    python scaling.py                      # 4 档，每档 8M token
    python scaling.py --tokens 12e6 --with-lstm
================================================================================
"""

#region 0. 导入库与常量   【模块:标准库 argparse/json/os/subprocess/sys + torch】
import argparse
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
PRESETS = ["tiny", "mini", "small", "base", "large"]  # 要对比的五档规模
# 每档用不同的 batch × 梯度累积，保证"每步 token 数"完全一致（32768），
# 这样对比才是公平的：唯一的变量是参数量 N，而不是 batch 大小。
# 大模型必须缩小 batch，否则 8G 显存会被撑爆 → Windows 会用共享内存兜底，速度直接掉一个数量级。
BATCH = {"tiny": 32, "mini": 32, "small": 32, "base": 32, "large": 16, "lstm": 32}  # batch 大小表
ACCUM = {"tiny": 2,  "mini": 2,  "small": 2,  "base": 2,  "large": 4,  "lstm": 2}  # 梯度累积步数表
#endregion


#region 1. 跑一档训练：起子进程调 train.py   【函数:run(预设名,token预算,序列长度,dropout,输出目录,架构),return None】
def run(preset, tokens, seq_len, dropout, out, arch="gpt"):
    """用 subprocess 起一个新的 python 进程跑 train.py。

    为什么用子进程而不是直接调函数：每档训练都要独占显存，
    在同一个进程里反复建/删模型容易留下显存碎片，跑几档就 OOM；
    子进程退出时显存会被系统完整回收。
    """
    key = "lstm" if arch == "lstm" else preset  # LSTM 用单独一套 batch 配置
    # -u 参数：关掉输出缓冲，否则子进程的进度要等结束才一起吐出来
    cmd = [sys.executable, "-u", os.path.join(HERE, "train.py"),
           "--preset", preset, "--arch", arch, "--tokens", str(tokens),
           "--seq_len", str(seq_len), "--dropout", str(dropout),
           "--batch", str(BATCH[key]), "--grad_accum", str(ACCUM[key]),
           "--eval_every", "60", "--out", out]
    print("\n" + "=" * 78 + f"\n▶ {' '.join(cmd[2:])}\n" + "=" * 78, flush=True)  # 打印实际执行的命令
    r = subprocess.run(cmd, cwd=HERE)  # 阻塞等它跑完
    if r.returncode != 0:
        # 不直接退出：画图用的是 metrics.jsonl，只要指标写出来了就能继续
        print(f"⚠ train.py 返回码 {r.returncode}（产物可能已正常写出，继续读取 metrics）")
#endregion

#region 2. 从 metrics.jsonl 读结果   【函数:read_metrics(输出目录),return dict 或 None】
def read_metrics(out):
    """从 metrics.jsonl 里取：参数量、最佳 val loss、最后一步的 train loss。
    参数量优先读日志里记录的（新版 train.py 会写），这样删掉 ckpt 也能复现图表。

    返回：{"params": int, "best_val": float, "final_train": float, "minutes": float}，文件缺失返回 None。
    """
    path = os.path.join(out, "metrics.jsonl")
    if not os.path.exists(path):
        return None  # 这一档没跑过，调用方会补跑
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]  # 一行一条记录
    if not rows:
        return None
    best_val = min(r["val_loss"] for r in rows)  # 整个训练过程中的最优验证 loss
    params = rows[0].get("params")  # 新版日志第一步就记了参数量
    if params is None:                          # 老日志没有记录，只能翻 ckpt
        ckpt = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        if "n_layer" in cfg:                    # Transformer：把结构搭起来数参数
            from model import GPT, GPTConfig
            params = GPT(GPTConfig(**cfg)).num_params()
        else:                                   # LSTM：没有共享权重，直接求和
            params = sum(v.numel() for v in ckpt["model"].values())
    return {"params": params, "best_val": best_val,
            "final_train": rows[-1]["train_loss"], "minutes": rows[-1]["elapsed_min"]}
#endregion


#region 3. 主流程：逐档训练 → 画图 → 幂律拟合   【函数:main(),return None】
def main():
    #region 3.1 命令行参数   【集合:args】
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=8e6)  # 每档的 token 预算，必须一致才公平
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)  # 小模型 + 小预算，dropout 调大些防过拟合
    ap.add_argument("--with-lstm", action="store_true", help="额外跑一个同参数量的 LSTM 作对照")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已有 ckpt 的档位直接复用结果，只补跑缺的（改图时很省时间）")
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))  # 输出根目录
    args = ap.parse_args()
    #endregion

    #region 3.2 逐档训练 / 复用结果   【集合:results】
    results = {}  # 字典，预设名 → read_metrics 的结果
    for p in PRESETS:
        out = os.path.join(args.out, f"scaling_{p}")  # 每档一个独立目录
        if args.skip_existing and os.path.exists(os.path.join(out, "metrics.jsonl")) \
                and read_metrics(out) is not None:
            # 已经跑过且指标可读，直接复用，不浪费 GPU 时间
            results[p] = read_metrics(out)
            print(f"↻ {p}: 复用已有结果 | best val {results[p]['best_val']:.4f}")
            continue
        run(p, args.tokens, args.seq_len, args.dropout, out)
        results[p] = read_metrics(out)
        print(f"✔ {p}: 参数量 {results[p]['params']/1e6:.2f}M | "
              f"best val {results[p]['best_val']:.4f} | 用时 {results[p]['minutes']:.1f}min")
    #endregion

    #region 3.3 可选的 LSTM 对照
    if args.with_lstm:
        out = os.path.join(args.out, "scaling_lstm")
        if args.skip_existing and os.path.exists(os.path.join(out, "metrics.jsonl")) \
                and read_metrics(out) is not None:
            results["lstm"] = read_metrics(out)
            print(f"↻ lstm: 复用已有结果 | best val {results['lstm']['best_val']:.4f}")
        else:
            # preset 传 "base" 只是为了拿到 d_model=384 作为 embed 维度，
            # 真正的容量由 lstm_hidden / lstm_layers 决定
            run("base", args.tokens, args.seq_len, args.dropout, out, arch="lstm")
            results["lstm"] = read_metrics(out)
            print(f"✔ lstm: 参数量 {results['lstm']['params']/1e6:.2f}M | "
                  f"best val {results['lstm']['best_val']:.4f}")
    #endregion

    # ---------- 画图 ----------
    #region 3.4 画 Scaling Law 曲线
    import math
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # 中文标签
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    import matplotlib.pyplot as plt

    # pts 是 [(参数量, 最佳val loss, 名字), ...]，只放真的跑出结果的档位
    pts = [(results[p]["params"], results[p]["best_val"], p) for p in PRESETS if results.get(p)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    xs = [x for x, _, _ in pts]  # x 轴：参数量
    ys = [y for _, y, _ in pts]  # y 轴：最佳验证 loss
    ax.plot(xs, ys, "o-", color="#c0392b", label="best val loss")  # 实线圆点：验证曲线
    ax.plot([x for x, _, _ in pts], [results[p]["final_train"] for _, _, p in pts],
            "s--", color="#7f8c8d", label="train loss (end)")  # 虚线方块：训练曲线
    for x, y, n in pts:
        off = (-20, 11) if n == "base" else (0, 11)   # base 的标签往左挪，避开 LSTM 星标
        # annotate 的 xytext 单位是"点"（offset points），不是数据坐标，所以能稳定避让
        ax.annotate(f"{n}\n{x/1e6:.1f}M", (x, y), textcoords="offset points",
                    xytext=off, ha="center", fontsize=8)
    if "lstm" in results:
        # 同参数量的 LSTM 用蓝色五角星标出来，一眼看出它比 Transformer 差多少
        ax.scatter([results["lstm"]["params"]], [results["lstm"]["best_val"]],
                   marker="*", s=220, color="#2980b9", zorder=5, label="LSTM（同参数量）")
        ax.annotate(f"LSTM {results['lstm']['best_val']:.2f}", 
                    (results["lstm"]["params"], results["lstm"]["best_val"]),
                    textcoords="offset points", xytext=(14, -2), ha="left", fontsize=8,
                    color="#2980b9")
    ax.set_xscale("log")  # 幂律在 log-log 上才是直线，所以横轴取 log
    ax.set_xlabel("参数量 N（log）")
    ax.set_ylabel("loss")
    ax.set_title(f"Scaling Law 实测（同一语料，固定 {args.tokens/1e6:.0f}M token 预算）")
    ax.grid(alpha=.3, which="both")  # which="both" 连次网格线也画
    ax.set_ylim(top=max(ys + [results[p]["final_train"] for _, _, p in pts]) * 1.07)  # 顶部留 7% 空白
    png = os.path.join(args.out, "scaling.png")
    #endregion

    #region 3.5 幂律拟合 + 标注过拟合区
    # 幂律拟合：log L = log a - α·log N。只拟合 loss 还在单调下降的那一段——
    # 过了最优点就是"过拟合区"，loss 反弹不是因为 Scaling Law 失效，
    # 而是 token 预算/语料固定，模型大到一定程度只能开始背语料。这一点必须画在图里。
    if len(pts) >= 2:
        import numpy as np
        imin = min(range(len(ys)), key=lambda i: ys[i])  # loss 最低的那一档的下标
        if imin >= 1:  # 至少要两个点才能拟合一条直线
            b, a = np.polyfit(np.log(xs[:imin + 1]), np.log(ys[:imin + 1]), 1)  # 斜率 b、截距 a
            # b 是负数（参数越多 loss 越低），所以 α = -b
            print(f"\n幂律拟合（{pts[0][2]}→{pts[imin][2]}）: L ≈ {math.exp(a):.2f} · N^({b:.3f})"
                  f"   （α≈{-b:.3f}，讲义给的参考值 ~0.076）")
            xfit = [xs[0], xs[imin]]  # 只在拟合区间内画线，不外推
            ax.plot(xfit, [math.exp(a) * x ** b for x in xfit], ":", color="#2c3e50",
                    label=f"幂律拟合 L∝N^({b:.2f})")
        if imin < len(pts) - 1:
            # 最优点之后还有档位 → 说明进入了过拟合区，用黄色底纹标出来
            ax.axvspan(xs[imin], xs[-1] * 1.25, color="#f1c40f", alpha=0.12)  # 半透明底色
            ax.annotate("过拟合区\n预算固定时，模型再大\n只会开始背语料",
                        xy=(xs[imin] * 2.6, ys[imin] - 0.42), fontsize=8, color="#7f6000",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#f1c40f", alpha=0.85))
        ax.legend(fontsize=7, loc="lower left")  # 图例放左下角，避开曲线
    #endregion

    #region 3.6 存图 + 存数据
    fig.tight_layout()
    fig.savefig(png)
    print(f"曲线: {png}")  # 打印图片路径
    with open(os.path.join(args.out, "scaling.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)  # 原始数值也存一份，方便复查
    #endregion
#endregion


#region 4. 入口
if __name__ == "__main__":
    main()
#endregion

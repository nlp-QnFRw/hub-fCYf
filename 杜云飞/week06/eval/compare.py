#region 0. 模块说明
"""
汇总：把所有 run 的 metrics.json 拼成一张表 + 图 + summary.md。

三条硬约定（规范 §3.8 / §4.3）：
  1. **只读不造**：本脚本不产生新指标，所有数字来自 runs/<run_id>/metrics.json；
     报告里写数字只能引用这里的产出。
  2. **缺失不影响其它**：某个 run 没跑或没评测，就打一行"[未评测]"，不抛异常。
  3. **口径检查**：各 run 的 n_eval 不一致时打印 warning（样本不同，结果不可直接比）。

图（figures/）：
  compare_macro_f1.png    各方法 macro-F1 横向条形图（带 acc 对照）
  data_curve.png          数据量曲线（只有 ≥3 个不同 n_train 的同类方法时才画）
  confusion_best.png      最优 run 的 14×14 混淆矩阵热力图

使用方式：
  python eval/compare.py
  python eval/compare.py --top 10 --no_plot
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import list_runs, load_metrics  # noqa: E402
from common.labels import LABELS  # noqa: E402
from common.metrics import to_markdown_table  # noqa: E402
from common.paths import FIGURES, RUNS  # noqa: E402

#endregion

#region 2. 字体
def setup_cjk_font() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
#endregion

#region 3. 收集数据
def load_meta(run_id: str) -> dict:
    p = RUNS / run_id / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def collect() -> list[dict]:
    """把每个 run 的 metrics + meta 合成一行记录（缺字段就留 None）。"""
    rows: list[dict] = []
    for rid in list_runs():
        m = load_metrics(rid)
        if m is None:
            print(f"[未评测] {rid}（缺 metrics.json，先跑 eval/evaluate.py --run {rid}）")
            continue
        meta = load_meta(rid)
        rows.append({
            "run_id": rid,
            "method": meta.get("method", rid.split("__")[1] if "__" in rid else "?"),
            "backbone": meta.get("backbone", "?"),
            "n_train": meta.get("n_train", 0),
            "hyperparams": meta.get("hyperparams") or None,
            "seed": meta.get("seed"),
            "macro_f1": m.get("macro_f1"),
            "acc": m.get("acc"),
            "weighted_f1": m.get("weighted_f1"),
            "group_macro_f1": m.get("group_macro_f1"),
            "parse_fail_rate": m.get("parse_fail_rate"),
            "ece": m.get("ece"),
            "ms_per_sample": m.get("ms_per_sample"),
            "n_eval": m.get("n_eval"),
            "elapsed_min": round((meta.get("elapsed_s") or 0) / 60, 1),
            "peak_mem_gb": meta.get("peak_mem_gb"),
            "cost_cny": meta.get("cost_usd", 0.0),
        })
    rows.sort(key=lambda r: -(r["macro_f1"] or -1))
    return rows
#endregion

#region 4. 画图
def plot_per_class(top: list[dict], k: int = 3) -> Path | None:
    """逐类 F1 对比（前 k 名），用来看"难点集中在哪些类"。
    比总表有用得多：总 macro-F1 相同，可能是一个模型全线均衡、另一个小类全崩。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    sel = [r for r in top[:k] if load_metrics(r["run_id"])]
    if len(sel) < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(LABELS))
    w = 0.8 / len(sel)
    for i, r in enumerate(sel):
        pc = (load_metrics(r["run_id"]) or {}).get("per_class") or {}
        f1 = [pc.get(name, {}).get("f1", 0.0) for name in LABELS]
        ax.bar(x + i * w, f1, width=w, label=r["run_id"])
    ax.set_xticks(x + w * (len(sel) - 1) / 2)
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("逐类 F1 对比：难点集中在哪些类")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "per_class_f1.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_bars(rows: list[dict]) -> Path | None:
    import matplotlib.pyplot as plt

    if not rows:
        return None
    top = rows[:15][::-1]
    labels = [r["run_id"].replace("__", " | ") for r in top]
    f1 = [r["macro_f1"] or 0 for r in top]
    acc = [r["acc"] or 0 for r in top]
    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.42 * len(top) + 1)))
    y = range(len(top))
    ax.barh([i + 0.2 for i in y], f1, height=0.38, color="#E45756", label="macro-F1 (test_bal)")
    ax.barh([i - 0.2 for i in y], acc, height=0.38, color="#4C78A8", label="acc (test_nat)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    for i, (a, b) in enumerate(zip(f1, acc)):
        ax.text(a, i + 0.2, f" {a:.3f}", va="center", fontsize=7)
        ax.text(b, i - 0.2, f" {b:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, 1.14)  # 留出右侧空间，避免最高分的数值标签被裁掉
    ax.set_xlabel("分数")
    ax.set_title("各方法对比（红=主指标 macro-F1，蓝=辅助指标 acc）")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "compare_macro_f1.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_data_curve(rows: list[dict]) -> Path | None:
    """数据量曲线：同一 method+backbone+**轮数**下，n_train → macro_f1。

    为什么必须把轮数放进分组键：30k 那个点是 2 epoch 的消融锚点，而 5k/20k/50k/100k
    都是 3 epoch。混在一条线上会得到"20k→30k 毫无提升、30k→100k 突然暴涨"的假曲线，
    真实原因是轮数不同 + 30k 在第 2 轮已经过拟合。
    """
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict]] = {}
    for r in rows:
        if not r["n_train"]:
            continue
        ep = (r.get("hyperparams") or {}).get("epochs")
        if ep is None:  # 免训练方法（TF-IDF/LoRA）没有 epochs 字段，单独成组
            key = f"{r['method']}-{r['backbone']}"
        else:
            key = f"{r['method']}-{r['backbone']}-{ep}ep"
        groups.setdefault(key, []).append(r)
    groups = {k: v for k, v in groups.items() if len({x["n_train"] for x in v}) >= 3}
    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda x: x["n_train"])
        ax.plot([x["n_train"] for x in rs], [x["macro_f1"] or 0 for x in rs],
                marker="o", label=key)
        for x in rs:  # 每个点标出具体分数，避免只看形状不看量级
            ax.annotate(f"{x['macro_f1']:.3f}", (x["n_train"], x["macro_f1"]),
                        textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax.set_xscale("log")  # 数据量跨两个数量级，log 轴才看得出边际收益递减
    ax.set_xlabel("训练集规模（条，log 轴）")
    ax.set_ylabel("macro-F1（test_bal）")
    ax.set_title("数据量曲线：加数据还值不值")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "data_curve.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_confusion(best: dict) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    m = load_metrics(best["run_id"])
    cm = m.get("confusion_matrix") if m else None
    if not cm:
        return None
    cm = np.array(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)  # 行归一化：看召回而非绝对量

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(norm * 100, cmap="Reds", vmin=0, vmax=100)  # 直接按百分数着色，与色标一致
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=9)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if norm[i, j] > 0.05:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵（行归一化 %）：{best['run_id']}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = FIGURES / "confusion_best.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out
def plot_ablation_delta(abl: list[dict]) -> Path | None:
    """消融的 Δ 条形图：一眼看出哪个旋钮真正有用、哪些只是噪声。

    Δ = 该配置 macro-F1 − 锚点 macro-F1（单位 pp）。红=变好，绿=变差（A 股配色）。
    """
    import matplotlib.pyplot as plt

    if len(abl) < 2:
        return None
    abl = sorted(abl, key=lambda x: x["delta_pp"])
    labels = [x["change"][:34] for x in abl]
    vals = [x["delta_pp"] for x in abl]
    colors = ["#D5484B" if v >= 0 else "#3A9E5F" for v in vals]

    fig, ax = plt.subplots(figsize=(9.2, max(2.6, 0.42 * len(abl) + 1.2)))
    ax.barh(range(len(vals)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#444", lw=0.9)
    for i, v in enumerate(vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.set_xlabel("Δ macro-F1（百分点，相对锚点）")
    ax.set_title("消融：每个改动相对基线的收益（红=变好，绿=变差）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "ablation_delta.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out
#endregion

#region 5. 消融分析
# 这些字段在消融里不构成"变量"：它们要么是分组键，要么只是评估规模
ABL_IGNORE: set[str] = {"train_split", "val_split", "num_val", "max_steps", "seed"}

# 超参名 → 人话（表格里直接读得懂）
HP_CN: dict[str, str] = {
    "max_len": "max_len", "truncation": "截断", "pool": "池化", "batch_size": "batch",
    "epochs": "轮数", "lr": "lr", "head_lr_mult": "头lr倍数", "weight_decay": "wd",
    "warmup_ratio": "warmup", "dropout": "dropout", "freeze_layers": "冻结层数",
    "class_weight": "类权重", "backbone": "主干", "grad_accum": "累积",
}


def _fmt(v) -> str:
    """超参值格式化：2e-05 → 2e-5，0.0 → 0，'none' 原样。"""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_ablation(rows: list[dict]) -> tuple[dict | None, list[dict], float | None]:
    """自动识别消融锚点并计算每个变体的 Δ。

    为什么自动求差而不是手写映射：手写表一改参数就过期，而且容易"结论先行"。
    这里直接读 meta.json 里的 hyperparams，**只有真正不同的字段才会出现在表格里**，
    等于让实验自己说话。

    返回：(锚点行, 消融行列表, seed 标准差或 None)
    """
    # 只看带完整超参的 BERT run（缺 hyperparams 的免训练方法不参与）
    cands = [r for r in rows if r.get("hyperparams")]
    if not cands:
        return None, [], None

    # 锚点判定：与"默认配置"完全一致的那一条（不看 seed），且它是 t1 的 30k×2ep 基准
    def is_anchor(hp: dict) -> bool:
        return (hp.get("train_split") == "sub_30k" and hp.get("epochs") == 2
                and hp.get("max_len") == 256 and hp.get("truncation") == "head_tail"
                and hp.get("pool") == "cls" and hp.get("batch_size") == 32
                and abs(float(hp.get("lr", 0)) - 2e-5) < 1e-12
                and abs(float(hp.get("dropout", 0)) - 0.1) < 1e-12
                and int(hp.get("freeze_layers", 0)) == 0
                and hp.get("class_weight") == "none"
                and abs(float(hp.get("warmup_ratio", 0)) - 0.06) < 1e-12)

    anchors = [r for r in cands if is_anchor(r["hyperparams"])]
    if not anchors:
        return None, [], None
    anchor = sorted(anchors, key=lambda r: (r.get("seed") != 42, r["run_id"]))[0]
    a_hp = anchor["hyperparams"]

    # seed 噪声：与锚点配置完全相同、只有 seed 不同的 run
    seed_f1 = [r["macro_f1"] for r in anchors if r["macro_f1"] is not None]
    seed_std = None
    if len(seed_f1) >= 2:
        mu = sum(seed_f1) / len(seed_f1)
        seed_std = (sum((x - mu) ** 2 for x in seed_f1) / (len(seed_f1) - 1)) ** 0.5

    abl: list[dict] = []
    for r in cands:
        if r["run_id"] == anchor["run_id"]:
            continue
        hp = r["hyperparams"]
        # 只与同预算的锚点比：训练集规模或轮数不同 = 不同实验，不能算"单因素消融"
        if hp.get("train_split") != a_hp.get("train_split") or hp.get("epochs") != a_hp.get("epochs"):
            continue
        diffs = []
        for k in sorted(set(a_hp) | set(hp)):
            if k in ABL_IGNORE:
                continue
            a, b = a_hp.get(k), hp.get(k)
            if a != b:
                diffs.append(f"{HP_CN.get(k, k)} {_fmt(a)}→{_fmt(b)}")
        if not diffs:  # 只有 seed 不同 → 属于噪声组，不进 Δ 表
            continue
        if r["macro_f1"] is None or anchor["macro_f1"] is None:
            continue
        abl.append({
            "run_id": r["run_id"],
            "change": "; ".join(diffs),
            "macro_f1": r["macro_f1"],
            "acc": r["acc"],
            "delta_pp": (r["macro_f1"] - anchor["macro_f1"]) * 100,
            "delta_acc_pp": ((r["acc"] or 0) - (anchor["acc"] or 0)) * 100,
            "elapsed_min": r["elapsed_min"],
        })
    abl.sort(key=lambda x: -x["delta_pp"])
    return anchor, abl, seed_std
#endregion

#region 7. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="汇总所有 run")
    ap.add_argument("--top", type=int, default=None, help="表格只显示前 N 行")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        sys.exit("没有已评测的 run。先跑 python eval/evaluate.py --all")

    # ---- 口径一致性检查 ----
    ns = {r["n_eval"] for r in rows if r["n_eval"]}
    if len(ns) > 1:
        print(f"[WARNING] 各 run 的 n_eval 不一致：{sorted(ns)} —— 结果不可直接比较！")

    show = rows[: args.top] if args.top else rows
    print("\n" + "=" * 100)
    print("方法对比（主指标 = test_bal 上的 macro-F1；acc = test_nat 上的自然分布准确率）")
    print("=" * 100)
    print(to_markdown_table(
        show, ["run_id", "macro_f1", "acc", "group_macro_f1", "parse_fail_rate", "ece",
               "elapsed_min", "peak_mem_gb", "n_train"],
        ["run_id", "macro-F1", "acc", "主题F1", "拒答率", "ECE", "耗时(min)", "显存(GB)", "n_train"]))

    if not args.no_plot:
        setup_cjk_font()
        p1 = plot_bars(rows)
        p2 = plot_data_curve(rows)
        p3 = plot_confusion(rows[0])
        p4 = plot_per_class(rows)
        for p in (p1, p2, p3, p4):
            if p:
                print(f"[图] {p}")

    # ---- 消融表（自动求差）----
    anchor, abl, seed_std = build_ablation(rows)
    if anchor and abl:
        print("\n" + "=" * 100)
        print(f"消融（锚点 {anchor['run_id']}，macro-F1={anchor['macro_f1']:.4f}）")
        if seed_std is not None:
            print(f"seed 噪声：同配置换种子的 macro-F1 标准差 = {seed_std * 100:.3f}pp "
                  f"（Δ 小于 2×该值 ≈ {seed_std * 200:.2f}pp 时不应下结论）")
        print("=" * 100)
        print(to_markdown_table(abl, ["change", "macro_f1", "delta_pp", "delta_acc_pp", "elapsed_min"],
                                ["改动（相对锚点）", "macro-F1", "ΔF1(pp)", "Δacc(pp)", "耗时(min)"]))
        if not args.no_plot:
            p5 = plot_ablation_delta(abl)
            if p5:
                print(f"[图] {p5}")

    # ---- summary.md ----
    md: list[str] = ["# 实验结果汇总（自动生成）", "",
                     "> 由 `python eval/compare.py` 生成。**报告中的数字只能引用这里**，不手工抄写。", "",
                     "## 主表", "",
                     to_markdown_table(
                         show, ["run_id", "macro_f1", "acc", "group_macro_f1",
                                "parse_fail_rate", "ms_per_sample", "elapsed_min", "peak_mem_gb"],
                         ["run_id", "macro-F1", "acc", "主题F1", "拒答率", "ms/条", "耗时(min)", "显存(GB)"]),
                     "", "## 评估口径", "",
                     f"- 主指标：`test_bal` 上 14 类 macro-F1（每类 ≥300 条，长尾类 F1 才有统计意义）",
                     f"- 辅指标：`test_nat_10k` 上 acc（按自然分布，贴近线上真实分布）",
                     "- 生成式方法的拒答/解析失败**计入分母**（算作错），失败率单列",
                     "- 同一批样本、固定 seed=42；相邻方法的差异需用 `eval/paired_test.py` 做 McNemar 检验",
                     "", "## 结论", ""]
    best = rows[0]
    md.append(f"- 最优：`{best['run_id']}`，macro-F1 **{best['macro_f1']:.4f}**、acc {best['acc']:.4f}"
              if best["macro_f1"] is not None else "")
    cheapest = min([r for r in rows if r["macro_f1"]], key=lambda r: (r["elapsed_min"] or 1e9))
    md.append(f"- 性价比最高：`{cheapest['run_id']}`（{cheapest['elapsed_min']} min 达到 "
              f"macro-F1 {cheapest['macro_f1']:.4f}）" if cheapest.get("macro_f1") else "")
    md.append(f"- 共 {len(rows)} 个 run；" + (f"其中 {sum(1 for r in rows if (r['parse_fail_rate'] or 0) > 0)} "
              f"个生成式 run 存在拒答，需按全样本口径解读。" if any(r["parse_fail_rate"] for r in rows) else ""))
    md += ["", "## 思考题", "",
           "1. 与「最强基线」比，L4（LoRA）多花的训练时间换来了几个点？这个交换在你的场景里成立吗？",
           "2. 数据量曲线的拐点在哪？如果只有 5k 数据，你会选 BERT 还是 LoRA？",
           "3. 主题层 F1 与细分类 F1 的差距，说明了什么业务取舍？"]

    if anchor and abl:
        md += ["", "## 消融：每个旋钮值多少分", "",
               f"锚点 `{anchor['run_id']}`（sub_30k × 2 epoch，macro-F1 **{anchor['macro_f1']:.4f}**）。"
               "表由 `meta.json` 里的 `hyperparams` **自动求差**生成，只列真正变过的字段。", ""]
        if seed_std is not None:
            md += [f"> **seed 噪声**：同配置换种子的 macro-F1 标准差 = {seed_std * 100:.3f}pp。"
                   f"因此 |Δ| < {seed_std * 200:.2f}pp 的格子应视为**噪声**，不构成结论。", ""]
        md += [to_markdown_table(abl, ["change", "macro_f1", "delta_pp", "delta_acc_pp", "elapsed_min"],
                                 ["改动（相对锚点）", "macro-F1", "ΔF1(pp)", "Δacc(pp)", "耗时(min)"]), ""]
        best_abl = abl[0]
        md += [f"- 收益最大的单项改动：**{best_abl['change']}**（+{best_abl['delta_pp']:.2f}pp）。",
               f"- 最差的改动：**{abl[-1]['change']}**（{abl[-1]['delta_pp']:+.2f}pp）。"]
        if seed_std is not None:
            sig = [x for x in abl if abs(x["delta_pp"]) > 2 * seed_std * 100]
            md += [f"- 在 seed 噪声之上仍然站得住的改动只有 **{len(sig)}/{len(abl)}** 项："
                   + ("、".join(f"`{x['change']}`" for x in sig) if sig else "无") + "。"]

    out_md = RUNS / "summary.md"
    out_md.write_text("\n".join([x for x in md if x is not None]) + "\n", encoding="utf-8")
    print(f"\n[写入] {out_md}")

    print("\n【关键结论】")
    if len(rows) >= 2:
        gap = (rows[0]["macro_f1"] or 0) - (rows[-1]["macro_f1"] or 0)
        print(f"- 最好与最差相差 {gap * 100:.1f}pp；第一名 {rows[0]['run_id']}。")
    print("- 差异 < 1pp 的两行，**不要**直接下结论，先跑 eval/paired_test.py 看 p 值。")
    print("\n下一步：python eval/paired_test.py --all-pairs")


if __name__ == "__main__":
    main()
#endregion

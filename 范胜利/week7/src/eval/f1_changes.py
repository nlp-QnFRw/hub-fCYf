# -*- coding: utf-8 -*-
"""输出「对比结果 + 各模型 F1 变化 + 对比图」。

产出：
  outputs/f1_changes.csv       全量 dev 与 sample n=200 的 F1、相对提升
  （不再写 .md；学习/结果类 md 统一放桌面合集）
  outputs/figures/fig_f1_progression.png  判别式主线 / LLM 线的 F1 递进
  outputs/figures/fig_f1_delta.png        相对规则、相对 BERT+Linear 的 F1 变化
  outputs/figures/fig_training_curves.png 本地重训的逐 epoch F1（数据量诊断）

指标来源分三层，不要混：
  - `dev930_*`：课程 PPT 的全量 dev 930 句参考值（来自 reference/metrics，硬编码在 METHODS 里）；
  - `n200_*`：sample 的 200 句同批复评（来自 outputs/sample_subset.csv）；
  - `local_*`：本次在 sample 上重训的逐 epoch 曲线（来自 outputs/local_run/logs）。

两个 Δ 的定义：
  - Δ vs 规则：衡量“相对于零训练基线，这个方法多拿了多少”；
  - Δ vs BERT+Linear：把“换模型/换解码”从“有没有预训练”里拆出来。
  单位统一为百分点（pp），正数表示更好。
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
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

_avail = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Songti SC", "STHeiti", "Arial Unicode MS", "PingFang SC"):
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# key, 显示名, 层级, trigger, argument, event, 是否判别式主线
# 最后一位只控制 fig_f1_progression 画在左图（判别式链条）还是右图（生成式路线）。
METHODS = [
    ("rule", "L0 规则", "L0", 0.6168, 0.1740, 0.0161, True),
    ("bilstm_crf", "L1 BiLSTM-CRF", "L1", 0.8165, 0.4634, 0.0885, True),
    ("bert_linear_256", "L2 BERT+Linear", "L2", 0.8293, 0.5630, 0.1564, True),
    ("bert_crf_256", "L2 BERT+CRF", "L2", 0.8320, 0.5815, 0.1697, True),
    ("bert_crf_256_hard", "L2 +硬约束", "L2", 0.8464, 0.6566, 0.2177, True),
    ("pipeline", "L2 两段式", "L2", 0.6769, 0.3690, 0.0957, False),
    ("local_zero_n60", "L3 本地 0.6B zero (n=60)", "L3", 0.4068, 0.1167, 0.0000, False),
    ("api_zero_n200", "L3 API zero-shot", "L3", 0.7789, 0.5224, 0.0402, False),
    ("api_few_n200", "L3 API few-shot", "L3", 0.8321, 0.5395, 0.0852, False),
    ("sft_lora_n200", "L4 LoRA SFT", "L4", 0.9000, 0.6971, 0.1850, False),
]

# 关键跃迁（全量 dev）
# 顺序刻意按“只改一个变量”排列：规则→建模→预训练→CRF→硬约束，
# 这样每一行的增量才能归因到那个变量，而不是笼统的“模型更强”。
TRANSITIONS = [
    ("规则 → BiLSTM-CRF", "预训练/建模", 0.1740, 0.4634),
    ("BiLSTM-CRF → BERT+Linear", "预训练语言模型", 0.4634, 0.5630),
    ("BERT+Linear → BERT+CRF", "结构化解码", 0.5630, 0.5815),
    ("BERT+CRF → 解码期硬约束", "合法性约束（不重训）", 0.5815, 0.6566),
    ("BERT+CRF → 两段式", "拆细（检测→类型→论元）", 0.5815, 0.3690),
    ("BERT+CRF → LoRA SFT", "生成式微调", 0.5815, 0.6971),
    ("规则 → LoRA SFT", "端到端最大跨度", 0.1740, 0.6971),
]


def load_bilstm_curve():
    """读 BiLSTM 逐 epoch 的内层验证 span F1（训练日志由 common/log_utils.py 写出）。"""
    p = ROOT / "outputs" / "local_run" / "logs" / "bilstm_crf_sample_train.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return [e["epoch"] for e in d.get("epochs", [])], [e["val"].get("inner_span_f1", 0) for e in d.get("epochs", [])]


def load_bert_curve():
    """读 BERT+Linear 逐 epoch 的内层验证 span F1；本机 135 步全 0，用于说明未收敛。"""
    p = ROOT / "outputs" / "local_run" / "logs" / "bert_linear_sample_train.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    eps = d.get("epochs", [])
    return [e["epoch"] for e in eps], [e["val"].get("inner_span_f1", 0) for e in eps]


def check_against_metrics():
    """硬编码的展示数字必须和 reference/metrics 对得上，防止文档与指标漂移。

    METHODS 里的 F1 是为了让图/表不依赖运行时文件而写死的；一旦有人重跑指标却忘了
    改这里，报告就会“表里一套、metrics 一套”。这里在生成图之前做一次 0.002 容差检查，
    不一致只 warn 不中断——因为出图仍然应该成功，问题要让人看见。
    """
    paths = {
        "rule": "reference/metrics/disc/rule.json",
        "bilstm_crf": "reference/metrics/disc/bilstm_crf.json",
        "bert_linear_256": "reference/metrics/disc/bert_linear_256.json",
        "bert_crf_256": "reference/metrics/disc/bert_crf_256.json",
        "bert_crf_256_hard": "reference/metrics/disc/bert_crf_256_hard.json",
        "pipeline": "reference/metrics/disc/pipeline.json",
        "local_zero_n60": "reference/metrics/llm/local_zero_n60.json",
        "api_zero_n200": "reference/metrics/llm/api_zero_n200.json",
        "api_few_n200": "reference/metrics/llm/api_few_n200.json",
        "sft_lora_n200": "reference/metrics/llm/sft_lora_n200.json",
    }
    bad = []
    for key, name, _, t, a, e, _ in METHODS:
        p = ROOT / paths[key]
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        for label, mine, theirs in (("trigger", t, m["trigger"]["f1"]),
                                    ("argument", a, m["argument"]["f1"]),
                                    ("event", e, m["event_exact"]["f1"])):
            if abs(mine - theirs) > 0.002:
                bad.append(f"{name} {label}: 表内 {mine:.4f} vs 指标 {theirs:.4f}")
    if bad:
        print("[warn] 展示数字与 reference/metrics 不一致：")
        for b in bad:
            print("  -", b)


def main():
    check_against_metrics()
    # ── 1. 表：全量 + sample n=200 + 变化 ─────────────────────
    sample = {}
    sp = OUT / "sample_subset.csv"
    if sp.exists():
        with open(sp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["source"] == "reference":
                    sample[r["method"]] = r
    # 三个基准：规则论元/事件、BERT+Linear 论元；用于计算 Δ 列。
    base_arg, base_ev = 0.1740, 0.0161
    lin_arg = 0.5630
    rows = []
    for key, name, lvl, t, a, e, _ in METHODS:
        s = sample.get(key, {})
        # 本地 0.6B 只评了 n=60，不能和全量 930 做差，Δ 列留空（—）。
        n60 = key == "local_zero_n60"
        rows.append({
            "层级": lvl, "方法": name, "key": key,
            "dev930_trigger": round(t, 4), "dev930_argument": round(a, 4), "dev930_event": round(e, 4),
            "n200_trigger": s.get("trigger_f1", ""), "n200_argument": s.get("argument_f1", ""),
            "n200_event": s.get("event_f1", ""), "n200_n": s.get("n", ""),
            "Δargument_vs_rule_pp": "" if n60 else round((a - base_arg) * 100, 1),
            "Δevent_vs_rule_pp": "" if n60 else round((e - base_ev) * 100, 1),
            "Δargument_vs_BERTLinear_pp": "" if n60 else round((a - lin_arg) * 100, 1),
        })
    with open(OUT / "f1_changes.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── 2. Markdown ───────────────────────────────────────────
    L = ["> **本文档主题：** 各模型三级 F1、相对规则的提升与关键跃迁对比\n",
         "> **初学者导读：** 先读第 3 节“一句话解读”，再看第 1/2 节表；图在 outputs/figures。\n"
         "> **怎么用：** 记“每一步涨多少 pp”，不要背全部数字；正负号表示比基线好/差。\n"
         "> **对应章节：** 初学者教程第 4 章「评测」\n"
         "> **文档级别：** 📊 结果",
         "# 对比结果与 F1 变化（全量 dev 930 句）\n",
         "## 1. 各模型三级 F1\n",
         "| 层级 | 方法 | 触发词 F1 | 论元 F1 | 事件级 F1 | Δ论元 vs 规则 | Δ事件 vs 规则 | Δ论元 vs BERT+Linear |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def _p(v):
            return "—" if v == "" else f"{v:+.1f}pp"
        L.append(f"| {r['层级']} | {r['方法']} | {r['dev930_trigger']:.4f} | {r['dev930_argument']:.4f} | "
                 f"{r['dev930_event']:.4f} | {_p(r['Δargument_vs_rule_pp'])} | "
                 f"{_p(r['Δevent_vs_rule_pp'])} | {_p(r['Δargument_vs_BERTLinear_pp'])} |")

    L += ["\n> 本地 0.6B zero-shot 只评了 n=60，不与全量 930 做差，Δ 列记 —。\n",
          "\n## 2. 关键跃迁（论元 F1）\n",
          "| 跃迁 | 变化 | 论元 F1 | 增量 |",
          "|---|---|---:|---:|"]
    for name, why, before, after in TRANSITIONS:
        L.append(f"| {name} | {why} | {before:.4f} → {after:.4f} | **{(after - before) * 100:+.1f}pp** |")

    L += ["\n## 3. 一句话解读\n",
          "- **最大跃迁**：规则 → LoRA SFT，论元 **+52.3pp**；其中规则 → BiLSTM 就贡献了 +28.9pp。",
          "- **预训练值多少**：BiLSTM → BERT+Linear，论元 **+10.0pp**（0.463→0.563）。",
          "- **CRF 值多少**：Linear → CRF 只 +1.9pp，但非法序列从 66.9% 降到 62.5%；"
          "再加解码期硬约束到 **+7.5pp**、非法 7.4%。",
          "- **拆细的代价**：BERT+CRF → 两段式，论元 **-21.3pp**（0.582→0.369）。",
          "- **生成式微调**：BERT+CRF → LoRA SFT，论元 **+11.6pp**，但事件级反而 -3.3pp（0.2177→0.1850）。",
          "- **非单调**：事件级最高的是硬约束（0.2177），不是 LoRA（0.1850）；"
          "论元最高的是 LoRA。选型要明确优化哪一级指标。"]

    # 不写 .md：结果类 Markdown 统一放在桌面 week7_学习文档合集.md，不进 GitHub。
    # 终端仍打印同一张表，CSV 仍是机器可读的权威结果。

    # ── 3. 图 1：判别式主线 / LLM 线 ──────────────────────────
    disc = [m for m in METHODS if m[6]]
    llm = [m for m in METHODS if m[0] in ("api_zero_n200", "api_few_n200", "sft_lora_n200")]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, group, title in ((axes[0], disc, "判别式主线：规则 → BiLSTM → BERT → CRF → 硬约束"),
                             (axes[1], llm, "生成式路线：zero-shot → few-shot → LoRA SFT")):
        x = list(range(len(group)))
        ax.plot(x, [m[3] for m in group], "o-", color="#2e86c1", label="触发词 F1", lw=2)
        ax.plot(x, [m[4] for m in group], "s-", color="#e67e22", label="论元 F1", lw=2)
        ax.plot(x, [m[5] for m in group], "^-", color="#27ae60", label="事件级 F1", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in group], rotation=22, ha="right", fontsize=9.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1")
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.grid(linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=9.5)
        for i, m in enumerate(group):
            ax.annotate(f"{m[4]:.2f}", (i, m[4]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8.5, color="#935116")
    fig.tight_layout()
    fig.savefig(FIG / "fig_f1_progression.png", dpi=180)
    plt.close(fig)

    # ── 4. 图 2：Δ vs 规则 / Δ vs BERT+Linear ────────────────
    others = [m for m in METHODS if m[0] not in ("rule", "local_zero_n60")]
    others.sort(key=lambda m: m[4])
    y = range(len(others))
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    d_arg = [(m[4] - base_arg) * 100 for m in others]
    d_ev = [(m[5] - base_ev) * 100 for m in others]
    ax.barh([i + 0.2 for i in y], d_arg, 0.4, color="#e67e22", label="Δ论元 F1 vs 规则")
    ax.barh([i - 0.2 for i in y], d_ev, 0.4, color="#27ae60", label="Δ事件级 F1 vs 规则")
    ax.set_yticks(list(y))
    ax.set_yticklabels([m[1] for m in others], fontsize=9.5)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlim(0, max(d_arg) * 1.12)
    ax.set_xlabel("相对规则基线的变化（百分点）")
    ax.set_title("各模型相对规则基线的 F1 提升：论元涨得多，事件级仍很低", fontsize=12, pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    for i, (a, e) in enumerate(zip(d_arg, d_ev)):
        ax.text(a + (1 if a >= 0 else -1), i + 0.2, f"{a:+.1f}", va="center",
                ha="left" if a >= 0 else "right", fontsize=8.3, color="#935116")
        ax.text(e + (1 if e >= 0 else -1), i - 0.2, f"{e:+.1f}", va="center",
                ha="left" if e >= 0 else "right", fontsize=8.3, color="#145a32")
    fig.tight_layout()
    fig.savefig(FIG / "fig_f1_delta.png", dpi=180)
    plt.close(fig)

    # ── 5. 图 3：本地逐 epoch F1（数据量诊断）────────────────
    bil = load_bilstm_curve()
    ber = load_bert_curve()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if bil:
        ax.plot(bil[0], bil[1], "o-", color="#e67e22", lw=2, label="BiLSTM-CRF（800 句 / 6 epoch）")
    if ber:
        ax.plot(ber[0], ber[1], "s--", color="#2e86c1", lw=2, label="BERT+Linear（800 句 / 3 epoch）")
    ax.set_xlabel("epoch")
    ax.set_ylabel("内层验证 span F1")
    ax.set_title("本地重训的 F1 变化：小数据下从零模型需要更多 epoch", fontsize=11.5, pad=8)
    ax.set_ylim(0, 0.55)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=9.5)
    if ber and max(ber[1]) == 0:
        ax.annotate("BERT：135 步全 O\n（未收敛）", xy=(ber[0][-1], 0), xytext=(1.6, 0.06),
                    fontsize=8.5, color="#2e86c1",
                    arrowprops=dict(arrowstyle="->", color="#2e86c1", lw=0.8))
    fig.tight_layout()
    fig.savefig(FIG / "fig_training_curves.png", dpi=180)
    plt.close(fig)

    print("\n".join(L))
    print("\n图：fig_f1_progression.png / fig_f1_delta.png / fig_training_curves.png")


if __name__ == "__main__":
    main()

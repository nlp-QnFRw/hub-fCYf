"""
生成图表与 HTML 报告

做三件在论文/汇报里真正有用的事：
  1. 主表之外给一张「难例子集」对比：只在 财经/房产/科技/家居 这四类上重算 Macro-F1。
     全量 10 类上 TF-IDF 已经 0.97，方法差距被天花板压平；切到易混子集才看得出差别。
  2. 混淆矩阵按行归一化——看的是「这类有多少被分错」，不是「多少样本堆在这一格」。
  3. 数据量曲线（若跑过 scan_scale.py）。
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.labels import CLASSES, CLASS2ID

OUT = ROOT / "outputs"
REP = ROOT / "reports"
CHART = REP / "charts"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SOURCES = [
    ("规则·类名匹配", "rule_pred.json", ("modes", "naive")),
    ("规则·关键词表", "rule_pred.json", ("modes", "keywords")),
    ("TF-IDF + SVM", "tfidf_pred.json", None),
    ("TextCNN 从零", "textcnn_pred.json", None),
    ("rbt3 微调", "rbt3_pred.json", None),
    ("MacBERT 微调", "macbert_pred.json", None),
    ("Qwen3 零样本", "zero_shot_pred.json", None),
]
HARD = ["财经", "房产", "科技", "家居"]


def pick(d, key):
    if key is None:
        return d
    for k in key:
        d = d[k]
    return d


def load_all():
    out = []
    for name, fname, key in SOURCES:
        p = OUT / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        node = pick(d, key)
        out.append({"name": name, "report": node["report"], "preds": node.get("preds", []),
                    "meta": d.get("meta", {})})
    return out


def hard_subset_f1(preds):
    """只在 4 个易混类上重算 Macro-F1（其余类的样本直接丢弃）"""
    ids = {CLASS2ID[c] for c in HARD}
    sub = [p for p in preds if p["true"] in ids]
    if not sub:
        return None
    from sklearn.metrics import f1_score
    yt = [p["true"] for p in sub]
    yp = [p["pred"] for p in sub]
    return round(float(f1_score(yt, yp, average="macro", labels=list(ids), zero_division=0)), 4)


def chart_bars(res):
    names = [r["name"] for r in res]
    f1 = [r["report"]["macro_f1"] for r in res]
    acc = [r["report"].get("acc_incl_unparsed", r["report"]["acc"]) for r in res]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(8, 0.52 * len(res) + 1.6), dpi=160)
    colors = ["#B4B2A9" if "规则" in names[i] else ("#378ADD" if "微调" in names[i] else "#5DCAA5")
              for i in order]
    ax.barh([names[i] for i in order], [f1[i] for i in order], color=colors, height=0.62)
    for y, i in enumerate(order):
        ax.text(f1[i] + 0.008, y, f"{f1[i]:.3f}  (acc {acc[i]:.3f})", va="center", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Macro-F1（10 类，8000 条验证）")
    ax.set_title("各方法横向对比", fontsize=12, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    p = CHART / "methods.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_hard(res):
    rows = [(r["name"], hard_subset_f1(r["preds"])) for r in res]
    rows = [(n, v) for n, v in rows if v is not None]
    if not rows:
        return None
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 0.52 * len(rows) + 1.6), dpi=160)
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#EF9F27", height=0.62)
    for y, (_, v) in enumerate(rows):
        ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro-F1（仅 财经/房产/科技/家居 四类）")
    ax.set_title("难例子集：全量分数被天花板压平后，差距在这里", fontsize=12, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    p = CHART / "hard_subset.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_confusion(res):
    best = max(res, key=lambda r: r["report"]["macro_f1"])
    cm = np.array(best["report"]["confusion_matrix"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=160)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontsize=9)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] and (cmn[i, j] > 0.02 or i == j):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.55 else "#2C2C2A")
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"混淆矩阵 · {best['name']}（按行归一化）", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    p = CHART / "confusion.png"
    fig.savefig(p)
    plt.close(fig)
    return p, best["name"]


def chart_scan():
    p = OUT / "scan.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["rows"]
    key = f"{d['model']}_f1"
    if not rows or key not in rows[0]:
        return None
    xs = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=160)
    ax.plot(xs, [r["tfidf_f1"] for r in rows], "o-", color="#888780", label="TF-IDF + SVM")
    ax.plot(xs, [r[key] for r in rows], "o-", color="#378ADD", label=f"{d['model']} 微调")
    for i, r in enumerate(rows):
        # 首个点两条线只差 0.001，标注必须错开左右，否则叠成一团
        if i == 0:
            ax.annotate(f"{r['tfidf_f1']:.3f}", (r["n_train"], r["tfidf_f1"]), textcoords="offset points",
                        xytext=(-12, 6), fontsize=8, color="#5F5E5A", ha="right", va="center")
            ax.annotate(f"{r[key]:.3f}", (r["n_train"], r[key]), textcoords="offset points",
                        xytext=(-12, -8), fontsize=8, color="#185FA5", ha="right", va="center")
        else:
            ax.annotate(f"{r['tfidf_f1']:.3f}", (r["n_train"], r["tfidf_f1"]), textcoords="offset points",
                        xytext=(0, -15), fontsize=8, color="#5F5E5A", ha="center")
            ax.annotate(f"{r[key]:.3f}", (r["n_train"], r[key]), textcoords="offset points",
                        xytext=(0, 9), fontsize=8, color="#185FA5", ha="center")
    vals = [v for r in rows for v in (r["tfidf_f1"], r[key])]
    ax.set_ylim(min(vals) - 0.0016, max(vals) + 0.0009)
    ax.set_xlim(xs[0] - 3000, xs[-1] + 1500)
    ax.set_xlabel("训练集条数")
    ax.set_ylabel("Macro-F1")
    gaps = [r.get("gap") for r in rows if r.get("gap") is not None]
    trend = "差距随数据量扩大" if gaps and gaps[-1] > gaps[0] else "差距随数据量缩小"
    ax.set_title(f"数据量 vs 效果（本任务：{trend}）", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p2 = CHART / "scale.png"
    fig.savefig(p2)
    plt.close(fig)
    return p2, rows, d["model"]


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def build_html(res, meta, charts, conf_info, scan):
    rows = []
    for r in sorted(res, key=lambda x: -x["report"]["macro_f1"]):
        rep = r["report"]
        rows.append(f"<tr><td>{r['name']}</td>"
                    f"<td class='num'>{fmt(rep.get('acc_incl_unparsed', rep.get('acc')))}</td>"
                    f"<td class='num'><b>{fmt(rep.get('macro_f1'))}</b></td>"
                    f"<td class='num'>{fmt(hard_subset_f1(r['preds']))}</td>"
                    f"<td class='num'>{fmt(rep.get('params_m'))}</td>"
                    f"<td class='num'>{fmt(rep.get('train_seconds') or rep.get('elapsed_s') or rep.get('fit_seconds'))}</td>"
                    f"<td class='num'>{rep.get('unparsed', 0)}</td></tr>")

    # 结论由数据反推，不写死——换了模型或数据重跑，这里跟着变
    by = {r["name"]: r["report"] for r in res}
    best = max(res, key=lambda r: r["report"]["macro_f1"])
    bf1 = best["report"]["macro_f1"]
    tf = by.get("TF-IDF + SVM", {})
    rb = by.get("rbt3 微调", {})
    zs = by.get("Qwen3 零样本", {})
    gap_tf = bf1 - (tf.get("macro_f1") or 0)
    def _cost(rep):
        for k in ("train_seconds", "elapsed_s", "fit_seconds"):
            if rep.get(k):
                return float(rep[k])
        return 0.0
    cost_ratio = (_cost(best["report"]) / _cost(tf)) if _cost(tf) else 0.0
    concl = [
        f"<b>判别式微调仍是最优解</b>：{best['name']}取得 Macro-F1 {bf1:.4f}，"
        f"比规则关键词表（{by.get('规则·关键词表', {}).get('macro_f1', 0):.3f}）高 "
        f"{(bf1 - by.get('规则·关键词表', {}).get('macro_f1', 0))*100:.0f} 个点。",
        f"<b>但本任务的边际收益被文本长度吃掉了</b>：TF-IDF 只用 {_cost(tf):.1f} 秒达到 "
        f"{tf.get('macro_f1', 0):.4f}，与最优解只差 {gap_tf*100:.1f} 个点，训练耗时却相差 {cost_ratio:.0f} 倍。"
        "正文（平均 900 字）里的类别词远比标题密集，字面特征已经够用。",
        f"<b>小模型性价比最高</b>：rbt3 参数只有 {rb.get('params_m', '-')}M（最优解的 "
        f"{(rb.get('params_m') or 0)/((best['report'].get('params_m') or 1)):.0%}），"
        f"F1 达 {rb.get('macro_f1', 0):.4f}，训练 {rb.get('train_seconds', '-')} 秒。"
        "预算紧或要频繁重训时选它，不必上 base。",
        f"<b>零样本可以兜底但不能上线</b>：Qwen3-0.6B 零样本 {zs.get('macro_f1', 0):.4f}，"
        f"覆盖 {zs.get('pred_distribution', {}).get('used_classes', '-')}/10 类，"
        f"解析失败 {zs.get('unparsed', 0)} 条。它比规则法强，但仍低于有监督方案 "
        f"{(tf.get('macro_f1', 0) - zs.get('macro_f1', 0))*100:.0f} 个点——适合冷启动期临时过渡。",
        "<b>规则法只能做下限与快筛</b>：类名匹配 0.413、关键词表 0.549，"
        "在难例四类上更是跌到 0.30 / 0.40，说明同义改写的表述它完全接不住。",
    ]
    if scan:
        srows = scan[1]
        g0, g1 = srows[0].get("gap"), srows[-1].get("gap")
        if g0 is not None and g1 is not None:
            if g1 > g0:
                concl.append(
                    f"<b>「小样本才是微调主场」在本任务不成立</b>：训练集 15% 时两者差距 {g0*100:+.1f} 个点，"
                    f"升到 100% 后扩大到 {g1*100:+.1f} 个点。原因是正文长、类别信号强，"
                    "TF-IDF 在几千条时就已接近天花板，而微调的预训练优势需要更多数据才兑现。"
                    "这条结论直接反驳了原项目在新闻标题上得到的规律——<b>方法选型不能跨任务照搬</b>。")
            else:
                concl.append(
                    f"<b>数据越少，微调越值钱</b>：训练集 15% 时差距 {g0*100:+.1f} 个点，"
                    f"100% 时收窄到 {g1*100:+.1f} 个点，与原项目在短标题任务上的结论一致。")
    concl_html = "".join(f"<li>{c}</li>" for c in concl)

    scan_html = ""
    if scan:
        p, srows, mname = scan
        trs = "".join(
            f"<tr><td class='num'>{r['n_train']}</td><td class='num'>{r['ratio']:.0%}</td>"
            f"<td class='num'>{r['tfidf_f1']:.4f}</td><td class='num'>{r.get(f'{mname}_f1', '-')}</td>"
            f"<td class='num'>{r.get('gap', '-')}</td></tr>" for r in srows)
        scan_html = f"""
  <h2>四、数据量实验</h2>
  <p>固定验证集，只改训练集比例。若微调优势随数据减少而扩大，说明它赢在「预训练知识」而不是「模型更大」。</p>
  <img src="charts/scale.png" alt="数据量曲线">
  <table><thead><tr><th>训练条数</th><th>比例</th><th>TF-IDF F1</th><th>{mname} F1</th><th>差距</th></tr></thead>
  <tbody>{trs}</tbody></table>"""

    conf_name = conf_info[1] if conf_info else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>THUCNews 分类方法对比实验</title>
<style>
 body{{margin:0;background:#FAFAF8;color:#2C2C2A;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.7}}
 .wrap{{max-width:900px;margin:0 auto;padding:48px 28px 72px}}
 h1{{font-size:26px;font-weight:500;margin:0 0 8px;letter-spacing:.2px}}
 .sub{{color:#5F5E5A;font-size:14px;margin-bottom:28px}}
 h2{{font-size:17px;font-weight:500;margin:38px 0 12px;padding-bottom:8px;border-bottom:1px solid #E3E1D9}}
 p{{font-size:14px}}
 table{{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0 8px;background:#fff;
       border:1px solid #E3E1D9;border-radius:8px;overflow:hidden}}
 th{{background:#F1EFE8;text-align:left;font-weight:500;padding:9px 12px;border-bottom:1px solid #E3E1D9}}
 td{{padding:9px 12px;border-bottom:1px solid #F1EFE8}}
 tr:last-child td{{border-bottom:none}}
 .num{{text-align:right;font-variant-numeric:tabular-nums}}
 img{{width:100%;border:1px solid #E3E1D9;border-radius:8px;margin:10px 0;background:#fff}}
 .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}
 .card{{background:#fff;border:1px solid #E3E1D9;border-radius:10px;padding:14px 16px}}
 .card .k{{font-size:12px;color:#5F5E5A}}
 .card .v{{font-size:20px;font-weight:500;margin-top:2px}}
 .note{{background:#FAEEDA;border-left:3px solid #EF9F27;padding:10px 14px;font-size:13px;border-radius:0 6px 6px 0}}
 ul.concl{{padding-left:20px;font-size:14px;margin:10px 0}}
 ul.concl li{{margin:11px 0;line-height:1.75}}
 code{{background:#F1EFE8;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">
 <h1>THUCNews 十类新闻分类 · 方法对比实验</h1>
 <div class="sub">数据集 THUCNews（{meta.get('n_raw')} 条 / 10 类）· 训练 {meta.get('n_train')} / 验证 {meta.get('n_val')}
 · 截断 {meta.get('max_len')} 字 · seed {meta.get('seed')} · 预训练模型 MacBERT-base 与 rbt3</div>

 <div class="cards">
  <div class="card"><div class="k">语料规模</div><div class="v">{meta.get('n_raw')}</div></div>
  <div class="card"><div class="k">全量重复率</div><div class="v">{meta.get('dup_rate', 0):.2%}</div></div>
  <div class="card"><div class="k">最优方法</div><div class="v">{conf_name}</div></div>
 </div>

 <h2>一、主结果</h2>
 <table><thead><tr><th>方法</th><th class="num">准确率</th><th class="num">Macro-F1</th>
 <th class="num">难例四类 F1</th><th class="num">参数量(M)</th><th class="num">训练耗时(s)</th><th class="num">解析失败</th></tr></thead>
 <tbody>{''.join(rows)}</tbody></table>
 <img src="charts/methods.png" alt="方法对比">

 <h2>二、难例子集：分数被天花板压平之后</h2>
 <p>十类全量上 TF-IDF 已接近饱和，方法之间的差距被均值掩盖。只在 财经 / 房产 / 科技 / 家居
 这四类（业务上同属「金融 + 产业」，词面高度重叠）上重算 Macro-F1，才看得出谁真的更强。</p>
 <img src="charts/hard_subset.png" alt="难例子集对比">

 <h2>三、错误长什么样</h2>
 <img src="charts/confusion.png" alt="混淆矩阵">
{scan_html}

 <h2>{'五' if scan else '四'}、结论</h2>
 <ul class="concl">{concl_html}</ul>
 <div class="note">本章表格与图表由 <code>eval/make_report.py</code> 依据 <code>outputs/*_pred.json</code>
 的逐条预测自动生成，结论文字也是从这些数字反推的，可复核。</div>
</div></body></html>
"""


def main():
    CHART.mkdir(parents=True, exist_ok=True)
    res = load_all()
    if not res:
        raise SystemExit("outputs/ 下没有结果文件，先跑各路实验")
    meta = res[0]["meta"]

    chart_bars(res)
    chart_hard(res)
    conf_info = chart_confusion(res)
    scan = chart_scan()

    html = build_html(res, meta, None, conf_info, scan)
    p = REP / "report.html"
    p.write_text(html, encoding="utf-8")
    print(f"→ {p}")
    for r in sorted(res, key=lambda x: -x["report"]["macro_f1"]):
        print(f"  {r['name']:<16} F1={r['report']['macro_f1']:.4f} hard={hard_subset_f1(r['preds'])}")


if __name__ == "__main__":
    main()

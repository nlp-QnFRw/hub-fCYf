"""汇总所有方法 → RESULTS.csv / RESULTS.md / 图。

为什么还要「同批重算」：
    LLM 只跑 300 条抽样，而 BERT 跑全量测试集，直接比 F1 是不公平的。
    这里读 LLM 的 eval_ids，把每个有监督模型在同一批 id 上重算一次（metrics_subset.json），
    表里「F1(同批)」才是可比的，全量 F1 只作为参考。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import FIGURES, OUTPUTS, REPORTS, get_logger, read_json, read_jsonl, write_json  # noqa: E402
from .data import load_labels, load_splits  # noqa: E402
from .metrics import evaluate  # noqa: E402

DATASETS = ["resume", "cluener", "msra"]
METHODS = [("bilstm", "BiLSTM-CRF"), ("bert", "BERT"), ("roberta", "RoBERTa"),
           ("llm_zero", "DeepSeek zero-shot"), ("llm_few", "DeepSeek 3-shot")]
COLORS = {"BiLSTM-CRF": "#8c8c8c", "BERT": "#4c72b0", "RoBERTa": "#dd8452",
          "DeepSeek zero-shot": "#55a868", "DeepSeek 3-shot": "#c44e52"}


def find_llm_tag(dataset: str) -> str | None:
    d = OUTPUTS / dataset / "llm"
    if not d.exists():
        return None
    cands = sorted(d.glob("metrics_*.json"))
    if not cands:
        return None
    # 优先取 n=300 的抽样，其次 all
    for c in cands:
        if "n300" in c.name:
            return c.stem.replace("metrics_", "")
    return cands[0].stem.replace("metrics_", "")


def load_method_rows() -> dict[str, dict[str, dict]]:
    """{dataset: {method: metrics_dict}}"""
    out: dict[str, dict[str, dict]] = {}
    for ds in DATASETS:
        out[ds] = {}
        for key, _ in METHODS:
            if key.startswith("llm"):
                tag = find_llm_tag(ds)
                if not tag:
                    continue
                mode = "zero" if key == "llm_zero" else "few"
                p = OUTPUTS / ds / "llm" / f"metrics_{mode}_n300.json"
                if not p.exists():
                    cands = sorted((OUTPUTS / ds / "llm").glob(f"metrics_{mode}_*.json"))
                    p = cands[0] if cands else None
                if p and p.exists():
                    out[ds][key] = read_json(p)
            else:
                p = OUTPUTS / ds / key / "metrics.json"
                if p.exists():
                    out[ds][key] = read_json(p)
    return out


def recompute_subset(ds: str, method: str, metrics: dict, ids: set[str]) -> dict:
    # @doc *** LLM 只跑 300 条，必须在同一批 id 上重算有监督模型才算同口径
    # 不重算的话 CLUENER 上 BERT 会被高估/低估 3pp（0.7385 vs 0.7699），足以改变排位结论
    pred_path = OUTPUTS / ds / ("llm" if method.startswith("llm") else method) / "pred.jsonl"
    if method.startswith("llm"):
        tag = metrics.get("tag")
        pred_path = OUTPUTS / ds / "llm" / f"pred_{tag}.jsonl"
    pred = read_jsonl(pred_path)
    gold = load_splits(ds)["test"]
    labels = load_labels(ds)
    r = evaluate(gold, pred, labels, subset_ids=ids)
    out = OUTPUTS / ds / ("llm" if method.startswith("llm") else method) / "metrics_subset.json"
    write_json(out, {"dataset": ds, "method": method, "subset_size": len(ids), "test": r})
    return r


def collect() -> list[dict]:
    rows = []
    for ds in DATASETS:
        data = load_method_rows().get(ds, {})
        if not data:
            continue
        llm_metrics = next((v for k, v in data.items() if k.startswith("llm")), None)
        ids = None
        if llm_metrics:
            eval_ids_path = OUTPUTS / ds / "llm" / f"eval_ids_{llm_metrics['tag']}.json"
            ids = set(read_json(eval_ids_path)) if eval_ids_path.exists() else None
        for key, label in METHODS:
            m = data.get(key)
            if not m:
                continue
            full_f1 = m["test"]["overall"]["f1"]
            if ids:
                sub = recompute_subset(ds, key, m, ids)
                f1 = sub["overall"]["f1"]
                p, r, macro = (sub["overall"]["precision"], sub["overall"]["recall"], sub["macro_f1"])
                n = sub["overall"]["n_docs"]
            else:
                f1, p, r, macro = full_f1, m["test"]["overall"]["precision"], m["test"]["overall"]["recall"], m["macro_f1"]
                n = m["test"]["overall"]["n_docs"]
            rows.append({
                "dataset": ds, "method": key, "method_label": label,
                "n_eval": n, "precision": p, "recall": r, "f1": f1,
                "f1_full": full_f1, "macro_f1": macro,
                "n_params": m.get("n_params", 0),
                "train_seconds": m.get("train_seconds", 0),
                "ms_per_sentence": m.get("ms_per_sentence", 0),
                "cost_cny": (m.get("llm") or {}).get("cost_cny", 0.0),
                "json_ok_rate": (m.get("llm") or {}).get("json_ok_rate"),
                "error_attr": m["test"].get("error_attribution", {}),
                "hard_f1": (m["test"].get("hard_subset") or {}).get("overall", {}).get("f1"),
                "per_type": m["test"].get("per_type", {}),
            })
    return rows


def write_tables(rows: list[dict]) -> None:
    csv_path = REPORTS / "RESULTS.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "n_eval", "P", "R", "F1_same_batch", "F1_full",
                    "macroF1", "params_M", "train_s", "ms_per_sentence", "cost_cny"])
        for r in rows:
            w.writerow([r["dataset"], r["method_label"], r["n_eval"],
                        f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                        f"{r['f1_full']:.4f}", f"{r['macro_f1']:.4f}",
                        f"{r['n_params']/1e6:.1f}" if r["n_params"] else "-",
                        f"{r['train_seconds']:.0f}" if r["train_seconds"] else "-",
                        f"{r['ms_per_sentence']:.2f}", f"{r['cost_cny']:.3f}"])

    lines = ["# 方法对比结果（自动生成）", "",
             "> F1(同批) = 在 LLM 抽样 id 上重算；F1(全量) 仅对有监督方法有意义。", "",
             "| 数据集 | 方法 | n | P | R | F1(同批) | F1(全量) | macroF1 | 参数量 | 训练 | ms/句 | 费用 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        params = f"{r['n_params']/1e6:.1f}M" if r["n_params"] else "—"
        train = f"{r['train_seconds']:.0f}s" if r["train_seconds"] else "—"
        cost = f"¥{r['cost_cny']:.3f}" if r["cost_cny"] else "—"
        lines.append(
            f"| {r['dataset']} | {r['method_label']} | {r['n_eval']} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | **{r['f1']:.4f}** | {r['f1_full']:.4f} | {r['macro_f1']:.4f} | "
            f"{params} | {train} | {r['ms_per_sentence']:.1f} | {cost} |")
    (REPORTS / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_f1(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            ax.set_visible(False)
            continue
        xs = np.arange(len(sub))
        vals = [r["f1"] for r in sub]
        ax.bar(xs, vals, color=[COLORS.get(r["method_label"], "#999") for r in sub], width=0.68)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([r["method_label"].replace("DeepSeek ", "DS ").replace("-CRF", "") for r in sub],
                           rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{ds} (n={sub[0]['n_eval']})", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("entity F1 (strict)")
    fig.suptitle("Entity-level F1 by method and dataset", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_f1_by_dataset.png", dpi=150)
    plt.close(fig)


def plot_per_type(rows: list[dict], ds: str = "cluener") -> None:
    sub = [r for r in rows if r["dataset"] == ds]
    if not sub:
        return
    labels = list(sub[0]["per_type"].keys())
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / max(1, len(sub))
    for i, r in enumerate(sub):
        vals = [r["per_type"].get(l, {}).get("f1", 0) for l in labels]
        ax.bar(np.arange(len(labels)) + i * width, vals, width=width,
               label=r["method_label"], color=COLORS.get(r["method_label"], "#999"))
    ax.set_xticks(np.arange(len(labels)) + width * (len(sub) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=len(sub))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Per-type F1 on {ds}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_per_type_f1.png", dpi=150)
    plt.close(fig)


def plot_error_attr(rows: list[dict], ds: str = "cluener") -> None:
    def total(e: dict) -> int:
        return e.get("total_fn") or sum(e.get(k, 0) for k in ("type_err", "boundary_err", "both_err", "miss"))

    sub = [r for r in rows if r["dataset"] == ds and total(r["error_attr"])]
    if not sub:
        return
    keys = ["type_err", "boundary_err", "both_err", "miss"]
    names = ["type error", "boundary error", "both", "missed"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(sub))
    for k, nm in zip(keys, names):
        vals = np.array([r["error_attr"].get(k, 0) for r in sub], dtype=float)
        vals = vals / np.array([total(r["error_attr"]) for r in sub])
        ax.bar([r["method_label"] for r in sub], vals, bottom=bottom, label=nm)
        bottom += vals
    ax.set_ylabel("share of all missed gold entities")
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title(f"How the misses break down on {ds}", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_error_attribution.png", dpi=150)
    plt.close(fig)


def plot_latency(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        if r["ms_per_sentence"] <= 0:
            continue
        ax.scatter(r["ms_per_sentence"], r["f1"], s=60, alpha=0.8,
                   color=COLORS.get(r["method_label"], "#999"),
                   marker={"resume": "o", "cluener": "s", "msra": "^"}[r["dataset"]])
    ax.set_xscale("log")
    ax.set_xlabel("latency (ms / sentence, log scale)")
    ax.set_ylabel("entity F1 (same batch)")
    ax.grid(alpha=0.3)
    ax.set_title("Accuracy vs latency", fontsize=11)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[v], label=v) for v in COLORS]
    handles += [plt.Line2D([], [], marker=m, ls="", color="#444", label=d)
                for d, m in {"resume": "o", "cluener": "s", "msra": "^"}.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_latency_vs_f1.png", dpi=150)
    plt.close(fig)


def plot_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, ds in zip(axes, ["cluener", "msra"]):
        for model in ("bert", "roberta"):
            p = OUTPUTS / ds / model / "train_log.json"
            if not p.exists():
                continue
            log = read_json(p)
            ax.plot([r["epoch"] for r in log], [r["dev_f1"] for r in log], marker="o",
                    color=COLORS["BERT" if model == "bert" else "RoBERTa"], label=model)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dev entity F1")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fine-tuning curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_training_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    log = get_logger("compare")
    rows = collect()
    if not rows:
        log.error("还没有任何 metrics.json，先跑训练/抽取")
        raise SystemExit(1)
    write_tables(rows)
    plot_f1(rows)
    plot_per_type(rows, "cluener")
    plot_error_attr(rows, "cluener")
    plot_latency(rows)
    plot_curves()
    log.info("写出 %s / %s / %d 张图", REPORTS / "RESULTS.csv", REPORTS / "RESULTS.md",
             len(list(FIGURES.glob("fig*.png"))))
    for r in rows:
        log.info("%-8s %-20s F1=%.4f (full %.4f) P=%.3f R=%.3f %.1fms",
                 r["dataset"], r["method_label"], r["f1"], r["f1_full"], r["precision"],
                 r["recall"], r["ms_per_sentence"])


if __name__ == "__main__":
    main()

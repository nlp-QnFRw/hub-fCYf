"""RoBERTa 七组消融实验。运行方式见 roberta/ABLATION_GUIDE.md。"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.data import load_splits, clean_val, sample_subset
from common.labels import LABELS
from common.model_paths import roberta_path
from train_roberta import BertClassifier, TnewsDataset

EXPERIMENTS = [
    ("exp1", "CLS 基线", "cls", 64),
    ("exp2", "Mean 池化", "mean", 64),
    ("exp3", "Max 池化", "max", 64),
    ("exp4", "CLS + Mean 拼接", "cls_mean", 64),
    ("exp5", "学习率 3e-5", "cls", 64),
    ("exp6", "CLS + 长度 32", "cls", 32),
    ("exp8", "Label Smoothing 0.1", "cls", 64),
]
DEFAULT_EXPERIMENTS = ["exp4", "exp5", "exp8"]


def save_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            predictions.extend(logits.argmax(-1).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    return targets, predictions


def run_one(exp, args, train, val, tokenizer, device, model_dir, out, existing):
    exp_id, title, pool, max_length = exp
    set_seed(args.seed)
    train_loader = DataLoader(TnewsDataset(train, tokenizer, max_length), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TnewsDataset(val, tokenizer, max_length), batch_size=args.batch_size)
    model = BertClassifier(model_dir, num_labels=len(LABELS), pool=pool).to(device)
    head = list(model.classifier.parameters())
    lr = 3e-5 if exp_id == "exp5" else args.lr
    smoothing = 0.1 if exp_id == "exp8" else 0.0
    opt = torch.optim.AdamW([
        {"params": list(model.bert.parameters()), "lr": lr},
        {"params": head, "lr": lr * 5},
    ], weight_decay=0.01)
    classes = np.arange(len(LABELS))
    weights = compute_class_weight("balanced", classes=classes, y=[d["label"] for d in train])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device), label_smoothing=smoothing)
    steps = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    total = steps * args.epochs
    warmup = int(total * .1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: step / max(1, warmup) if step < warmup else max(0., (total-step) / max(1, total-warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best_f1, best_state, best_predictions, best_epoch = -1., None, None, 0
    history = []
    start = time.time()
    print(f"\n[{exp_id}] {title} | 设备={device} | 每轮 {steps} 步 | 验证 {len(val)} 条", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.
        for i, batch in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            running += loss.item()
            done = i + 1
            if done == 1 or done % args.log_every == 0 or done == steps:
                elapsed = time.time() - epoch_start
                eta = elapsed / done * (steps - done)
                print(f"  {exp_id} 第 {epoch}/{args.epochs} 轮 {done}/{steps} 步 loss={loss.item():.4f} 本轮训练预计剩余 {eta/60:.1f} 分钟", flush=True)
            if done >= steps:
                break
        print(f"  {exp_id} 开始验证...", flush=True)
        ys, ps = evaluate(model, val_loader, device)
        acc = float(accuracy_score(ys, ps))
        f1 = float(f1_score(ys, ps, labels=list(range(len(LABELS))), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": running / steps, "accuracy": acc, "macro_f1": f1, "seconds": time.time()-epoch_start})
        print(f"  第 {epoch} 轮验证 accuracy={acc:.4f} macro_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_predictions, best_epoch = f1, ps, epoch
            # 只有可能成为全局最优时才在内存保留参数，不为每组保存磁盘权重。
            global_best = max((r["macro_f1"] for r in existing), default=-1.)
            if not args.no_model and f1 > global_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ys = [d["label"] for d in val]
    precision, recall, fscore, support = precision_recall_fscore_support(ys, best_predictions, labels=list(range(len(LABELS))), zero_division=0)
    matrix = confusion_matrix(ys, best_predictions, labels=list(range(len(LABELS))))
    result = {"id": exp_id, "name": title, "pool": pool, "max_length": max_length,
              "lr": lr, "label_smoothing": smoothing, "best_epoch": best_epoch, "accuracy": float(accuracy_score(ys, best_predictions)),
              "macro_f1": best_f1, "seconds": time.time()-start, "history": history,
              "per_class": [{"label": name, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(fscore[i]), "support": int(support[i])} for i, name in enumerate(LABELS)],
              "confusion_matrix": matrix.tolist(),
              "predictions": [{"text": d["sentence"], "true": int(d["label"]), "pred": int(p)} for d, p in zip(val, best_predictions)]}
    if best_state is not None:
        # 临时文件写完后才替换旧模型，避免写入失败破坏已有的最佳模型。
        target = out / "best_model.pt"
        temp = out / "best_model.pt.tmp"
        torch.save({"state_dict": best_state, "pool": pool, "val_acc": result["accuracy"], "val_macro_f1": best_f1,
                    "experiment": exp_id, "args": {"max_length": max_length}, "model_dir": str(model_dir)}, temp)
        temp.replace(target)
    save_json(out / f"{exp_id}.json", result)
    del model, opt, best_state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def make_report(results, out, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if font in fonts:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False
    ids = [r["id"] for r in results]
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(9, len(results) * 1.6), 5))
    for offset, metric, label in [(-.18, "accuracy", "准确率"), (.18, "macro_f1", "Macro-F1")]:
        values = [r[metric] for r in results]
        bars = ax.bar(x + offset, values, .36, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xticks(x, [r["id"] + "\n" + r["name"] for r in results])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("分数")
    ax.set_title("RoBERTa 消融实验对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "metrics_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    width = .8 / len(results)
    for i, r in enumerate(results):
        ax.bar(np.arange(len(LABELS)) + (i-(len(results)-1)/2)*width, [c["f1"] for c in r["per_class"]], width, label=r["id"])
    ax.set_xticks(np.arange(len(LABELS)), LABELS, rotation=45)
    ax.set_ylim(0, 1)
    ax.set_title("各类别 F1 对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_f1_comparison.png", dpi=180)
    plt.close(fig)
    for r in results:
        m = np.array(r["confusion_matrix"])
        norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(LABELS)), LABELS, rotation=45)
        ax.set_yticks(np.arange(len(LABELS)), LABELS)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"{r['id']} {r['name']} 混淆矩阵（颜色：行占比；数字：样本数）")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if m[i, j]:
                    ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=7, color="white" if norm[i,j] > .5 else "black")
        fig.colorbar(im, ax=ax, label="行占比")
        fig.tight_layout()
        fig.savefig(out / f"{r['id']}_confusion.png", dpi=150)
        plt.close(fig)
    with (out / "prediction_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["标题", "真实类别"] + ids)
        for samples in zip(*(r["predictions"] for r in results)):
            writer.writerow([samples[0]["text"], LABELS[samples[0]["true"]]] + [LABELS[s["pred"]] for s in samples])
    base = results[0]
    example_rows = []
    with (out / "examples.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["对比实验", "情况", "标题", "真实类别", "基线预测", "实验预测"])
        for r in results[1:]:
            for kind in ("基线错实验对", "基线对实验错", "均错但预测不同"):
                count = 0
                for b, p in zip(base["predictions"], r["predictions"]):
                    y = b["true"]
                    match = ((kind == "基线错实验对" and b["pred"] != y and p["pred"] == y) or
                             (kind == "基线对实验错" and b["pred"] == y and p["pred"] != y) or
                             (kind == "均错但预测不同" and b["pred"] != y and p["pred"] != y and b["pred"] != p["pred"]))
                    if match and count < 20:
                        row = [r["id"], kind, b["text"], LABELS[y], LABELS[b["pred"]], LABELS[p["pred"]]]
                        writer.writerow(row)
                        if count < 1:
                            example_rows.append(row)
                        count += 1
    if example_rows:
        fig, ax = plt.subplots(figsize=(16, max(3, len(example_rows) * .65 + 1)))
        ax.axis("off")
        display = [[r[0], r[1], (r[2][:24] + "…") if len(r[2]) > 24 else r[2], r[3], r[4], r[5]] for r in example_rows]
        table = ax.table(cellText=display, colLabels=["实验", "情况", "标题（完整文本见 CSV）", "真实", "基线", "实验"],
                         colWidths=[.07, .16, .48, .09, .09, .09], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.set_title("预测差异实例（每种情况各展示一条）")
        fig.tight_layout()
        fig.savefig(out / "examples.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["实验", "配置", "准确率", "Macro-F1", "相对基线F1百分点", "最佳轮次", "耗时分钟"])
        for r in results:
            writer.writerow([r["id"], r["name"], r["accuracy"], r["macro_f1"], (r["macro_f1"]-base["macro_f1"])*100, r["best_epoch"], r["seconds"]/60])
    winner = max(results, key=lambda r: r["macro_f1"])
    lines = [f"# RoBERTa {len(results)} 组消融实验报告", "", f"训练样本：{manifest['n_train']}；去重验证样本：{manifest['n_eval']}；随机种子：{manifest['seed']}；每组最多 {manifest['epochs']} 轮。", "", "各组均从同一预训练模型开始，每组以去重验证集 Macro-F1 选择最佳轮次。该验证集同时用于模型选择和结果比较，结果不能当作独立测试集表现。", "", "| 实验 | 配置 | 准确率 | Macro-F1 | 相对基线 F1（百分点） | 最佳轮次 | 耗时（分钟） |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['name']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {(r['macro_f1']-base['macro_f1'])*100:+.2f} | {r['best_epoch']} | {r['seconds']/60:.1f} |")
    lines += ["", f"本次 Macro-F1 最佳：**{winner['id']} {winner['name']}**。单个随机种子的差异可能有波动，不能直接归因于配置。", "", "![指标对比](metrics_comparison.png)", "", "![类别 F1 对比](class_f1_comparison.png)", "", "## 各组混淆矩阵", ""]
    for r in results:
        lines += [f"### {r['id']} {r['name']}", "", f"![混淆矩阵]({r['id']}_confusion.png)", ""]
    lines += ["## 预测实例", "", "![预测差异实例](examples.png)" if example_rows else "本次没有不同预测的实例。", "", "`examples.csv` 展示基线与各实验的典型差异；`prediction_comparison.csv` 保存所有评估样本的预测。", "", "报告仅包含已完成的实验，未列出的配置没有测试。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="依次运行四组 RoBERTa 消融实验，自动续跑并生成报告")
    ap.add_argument("--experiments", nargs="+", choices=[e[0] for e in EXPERIMENTS], default=DEFAULT_EXPERIMENTS, help="需要补跑的实验编号；已完成的自动跳过")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "roberta_ablation"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="每轮最大步数，0 为完整训练；仅供测试")
    ap.add_argument("--max_eval", type=int, default=0, help="固定抽取验证样本数量，0 为全部；仅供测试")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--no_model", action="store_true", help="不保存模型，仅供快速测试")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.log_every < 1 or args.max_steps < 0 or args.max_eval < 0 or args.lr <= 0:
        ap.error("轮数、批大小、日志间隔和学习率必须为正；样本和步数上限不能为负")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train, val = load_splits()
    val, duplicates = clean_val(train, val)
    if args.max_eval:
        val = sample_subset(val, args.max_eval, args.seed)
    if not val:
        raise ValueError("验证集为空")
    model_dir = roberta_path()
    def fingerprint(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        return h.hexdigest()
    files = sorted(p for p in model_dir.iterdir() if p.is_file() and (p.suffix in {".json", ".txt", ".safetensors"} or p.name == "pytorch_model.bin"))
    manifest = {"version": 1, "model_dir": str(model_dir), "model_files": {p.name: fingerprint(p) for p in files},
                "train_sha256": fingerprint(ROOT / "data/train.json"), "val_sha256": fingerprint(ROOT / "data/val.json"),
                "n_train": len(train), "n_eval": len(val), "duplicates_removed": duplicates,
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
                "max_steps": args.max_steps, "max_eval": args.max_eval, "no_model": args.no_model}
    meta = out / "config.json"
    if meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")) != manifest:
            raise SystemExit(f"配置、数据或模型与已有结果不同。为避免混合实验，请使用新的 --output 目录：{out}")
    else:
        if any((out / f"{e[0]}.json").exists() for e in EXPERIMENTS):
            raise SystemExit("发现已有实验结果但缺少配置记录，请使用新的 --output 目录。")
        save_json(meta, manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}；训练 {len(train)} 条，去重验证 {len(val)} 条；结果目录：{out}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # 先载入所有已有结果，再判断新的模型是否超过历史最佳。
    by_id = {}
    for exp in EXPERIMENTS:
        path = out / f"{exp[0]}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("id") != exp[0] or len(result.get("predictions", [])) != len(val):
                raise SystemExit(f"实验结果文件不完整或不匹配：{path}")
            if any(p["text"] != d["sentence"] or p["true"] != d["label"] for p, d in zip(result["predictions"], val)):
                raise SystemExit(f"实验结果的验证样本不匹配：{path}")
            by_id[exp[0]] = result
    if "exp1" not in by_id and "exp1" not in args.experiments:
        raise SystemExit("报告需要 exp1 基线，请用 --experiments exp1 ... 先运行基线。")
    for exp in EXPERIMENTS:
        if exp[0] not in args.experiments:
            continue
        if exp[0] in by_id:
            print(f"[{exp[0]}] 已完成，跳过。", flush=True)
        else:
            by_id[exp[0]] = run_one(exp, args, train, val, tokenizer, device, model_dir, out, list(by_id.values()))
    results = [by_id[e[0]] for e in EXPERIMENTS if e[0] in by_id]
    if not results or results[0]["id"] != "exp1":
        raise SystemExit("没有可用的 exp1 基线。")
    make_report(results, out, manifest)
    print(f"\n全部完成！报告：{out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

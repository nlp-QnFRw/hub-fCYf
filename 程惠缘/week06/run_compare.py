# -*- coding: utf-8 -*-
"""run_compare.py —— 换预训练模型，做分类任务对比实验（第 6 周作业）

【实验设计】
    只换一个变量：预训练的底座（BERT 权重）。数据、标签、训练配方、评测口径全部不动。
    这样涨的每一个点都能归因到「底座」，而不是别的什么东西。

【为什么不用课件那 500 条】
    课件为课堂演示只抽了 500 条。500 条上 acc≈0.55 的标准误是 2.2%，
    95% 置信区间 ±4.4 个点 —— 也就是说「换个模型涨 2 个点」和「同一个模型换个种子重跑」
    一样大。本脚本默认在**全部 8,901 条干净验证样本**上评（标准误降到 0.53%），
    并且支持多个 seed + 配对检验（McNemar），结论才站得住。

【复用课件代码】
    直接 import 课件的 BertClassifier / TnewsDataset / 指标函数，
    保证和课件同一套池化、同一套指标口径，不是另写一份。

用法:
    python run_compare.py                                  # 三个底座 × 1 seed，全量评测
    python run_compare.py --seeds 42,43                     # 每个底座跑 2 个 seed
    python run_compare.py --models bert-base                # 只跑一个
    python run_compare.py --max-steps 5 --eval-n 200        # 冒烟：不跑全程，只验流程
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
CODE = (HERE.parent / "week6-数据接入与分流" / "code").resolve()

sys.path.insert(0, str(CODE))           # common.* 在这里
sys.path.insert(0, str(CODE / "bert"))  # train_bert.py 在这里
from common.data import load_splits, clean_val, sample_subset  # noqa: E402
from common.labels import LABELS                                # noqa: E402
from common.metrics import (classification_report, expected_calibration_error,  # noqa: E402
                            pred_distribution)
import train_bert as TB                                         # noqa: E402  复用模型与数据集

# 三个底座（路径按本机实际情况写死，换机器只改这里）
CANDIDATES = {
    "bert-base":     r"D:\Dev\Py\AStady\pretrain_models\bert-base-chinese",
    "macbert":       r"D:\Dev\models\chinese-macbert-base",
    "roberta-large": r"D:\Dev\models\chinese-roberta-wwm-ext-large2",
}
# 大模型 8G 卡上要减 batch、加梯度累积，保证「有效 batch」和 base 一致（=64）
BATCH = {"bert-base": 64, "macbert": 64, "roberta-large": 16}
ACCUM = {"bert-base": 1,  "macbert": 1,  "roberta-large": 4}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    """在给定 loader 上跑推理，返回 (逐条预测, 概率矩阵, 标签列表)。

    注意：课件的 TnewsDataset 每条返回的是 dict（input_ids/attention_mask/labels），
    DataLoader 组批之后仍然是 dict——不是 (x, y) 元组。
    """
    model.eval()
    preds, probs, labels = [], [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1).cpu()
        probs.append(p)
        preds += p.argmax(-1).tolist()
        labels += batch["labels"].tolist()
    return preds, torch.cat(probs).numpy(), labels


def train_and_eval(name, model_dir, seed, args, train, eval_items, device):
    """训练一个底座 → 在干净验证集上评测。返回指标 + 逐条预测（供配对检验用）。"""
    from transformers import AutoTokenizer
    t0 = time.time()
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(model_dir)
    train_ds = TB.TnewsDataset(train, tok, args.max_length)
    eval_ds = TB.TnewsDataset(eval_items, tok, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=128)

    model = TB.BertClassifier(model_dir, num_labels=len(LABELS), pool=args.pool).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6

    # 分层学习率 + 类别加权 + warmup：和课件默认配方完全一致
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    bert_params = [p for n, p in model.named_parameters()
                   if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([{"params": bert_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=0.01)
    cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                              y=[d["label"] for d in train])
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(device))

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total = steps_per_epoch * args.epochs
    warm = int(total * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    print(f"  [{name} seed={seed}] {n_param:.0f}M 参数 batch={args.batch_size}"
          f"×累积{ACCUM[name]} steps/epoch={steps_per_epoch}", flush=True)
    best = None
    for ep in range(1, args.epochs + 1):
        model.train()
        run = i = 0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(ids, mask)
            loss = crit(logits, y) / ACCUM[name]
            loss.backward()
            run += loss.item()
            if (i + 1) % ACCUM[name] == 0:
                opt.step(); sched.step(); opt.zero_grad()
            i += 1
            if args.max_steps and i >= args.max_steps:
                break
        preds, probs, labels = evaluate(model, eval_dl, device)
        rep = classification_report(labels, preds, len(LABELS))
        print(f"    epoch {ep}: eval acc={rep['acc']:.4f} f1={rep['macro_f1']:.4f}", flush=True)
        if best is None or rep["macro_f1"] > best["rep"]["macro_f1"]:
            best = {"rep": rep, "preds": preds, "probs": probs, "labels": labels, "epoch": ep}

    rep = best["rep"]
    rep["ece"] = expected_calibration_error(best["labels"], best["probs"])
    rep["pred_distribution"] = pred_distribution(best["preds"], len(LABELS))
    return {
        "model": name, "model_dir": model_dir, "seed": seed,
        "params_m": round(n_param, 1),
        "best_epoch": best["epoch"],
        "acc": rep["acc"], "macro_f1": rep["macro_f1"], "ece": rep["ece"],
        "used_classes": rep["pred_distribution"]["used_classes"],
        "elapsed_s": round(time.time() - t0, 1),
        "preds": best["preds"], "labels": best["labels"],
    }


def mcnemar(pred_a, pred_b, labels):
    """配对检验：只看两个模型预测不同的样本，二项检验 b vs c。"""
    b = sum(1 for a, c, g in zip(pred_a, pred_b, labels) if a == g and c != g)
    c_ = sum(1 for a, c, g in zip(pred_a, pred_b, labels) if a != g and c == g)
    n = b + c_
    if n == 0:
        return b, c_, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(b, n, 0.5).pvalue
    except Exception:                      # 没装 scipy 就退回正态近似
        import math
        z = (abs(b - c_) - 1) / math.sqrt(n)
        p = math.erfc(z / math.sqrt(2))
    return b, c_, round(float(p), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bert-base,macbert,roberta-large")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=0, help="0=按模型自动选")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--eval-n", type=int, default=0, help="评测条数，0=全部 8,901 条干净样本")
    ap.add_argument("--max-steps", type=int, default=0, help="每个 epoch 最多几步（冒烟用）")
    ap.add_argument("--out-dir", default=str(HERE / "runs"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    (Path(args.out_dir)).mkdir(parents=True, exist_ok=True)

    train, val = load_splits()
    clean, n_dup = clean_val(train, val)
    eval_items = sample_subset(clean, args.eval_n or len(clean), seed=42)   # 评测集固定 seed=42
    print(f"训练 {len(train)} 条 | 去重后干净验证 {len(clean)} 条（剔除重复 {n_dup}）"
          f" | 本次评测 {len(eval_items)} 条 | 设备 {device}\n")

    names = [s.strip() for s in args.models.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = []
    for name in names:
        if name not in CANDIDATES:
            raise SystemExit(f"不认识的模型：{name}（可选 {', '.join(CANDIDATES)}）")
        for seed in seeds:
            args.batch_size = args.batch_size or BATCH[name]
            r = train_and_eval(name, CANDIDATES[name], seed, args, train, eval_items, device)
            print(f"  → {name} seed={seed}: acc={r['acc']} f1={r['macro_f1']} "
                  f"ece={r['ece']} {r['elapsed_s']}s\n", flush=True)
            results.append(r)
            (Path(args.out_dir) / f"{name}_s{seed}.json").write_text(
                json.dumps(r, ensure_ascii=False), encoding="utf-8")
        args.batch_size = 0            # 下一个模型重新按表选

    # ── 汇总 ──
    print("=" * 72)
    print(f"{'底座':<16}{'参数':>8}{'acc':>9}{'macro-F1':>10}{'ECE':>8}{'覆盖类':>8}{'耗时':>8}")
    print("=" * 72)
    for r in results:
        print(f"{r['model']:<16}{r['params_m']:>7.0f}M{r['acc']:>9.4f}{r['macro_f1']:>10.4f}"
              f"{r['ece']:>8.4f}{r['used_classes']:>8}{r['elapsed_s']:>7.0f}s")

    if len(results) > 1:
        print("\n配对检验（McNemar，同一批样本）：")
        base = results[0]
        for r in results[1:]:
            b, c, p = mcnemar(base["preds"], r["preds"], base["labels"])
            sig = "显著" if p < 0.05 else "不显著（落在噪声内）"
            print(f"  {base['model']} vs {r['model']}: "
                  f"前者独对 {b} 条 / 后者独对 {c} 条 → p={p}，{sig}")

    out = Path(args.out_dir) / "summary.json"
    out.write_text(json.dumps([{k: v for k, v in r.items() if k != "preds"} for r in results],
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()

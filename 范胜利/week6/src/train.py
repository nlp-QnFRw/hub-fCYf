# -*- coding: utf-8 -*-
"""
train.py —— 单基座微调（同一份代码跑 BERT / RoBERTa，配置只来自 config.py）

【用法】
    # 正式跑（默认 12,000 条训练 / 2 epoch / batch 64，CPU 上单基座约 15~25 分钟）
    python src/train.py --backbone bert-base-chinese
    python src/train.py --backbone chinese-roberta-wwm-ext

    # 全量数据（有 GPU 时用这个，53,360 条，效果最好）
    python src/train.py --backbone bert-base-chinese --train_size 0 --epochs 3

    # 冒烟：2000 条训练、300 条验证、20 步封顶，几十秒验证代码通路
    python src/train.py --backbone bert-base-chinese --smoke

【产物】
    outputs/metrics_<backbone>.json   —— 指标 + 配置 + 逐 epoch 日志 + 混淆矩阵
    outputs/pred_<backbone>.json      —— 逐条预测（idx / 真实 / 预测 / 置信度）
    带 --tag 时文件名加后缀，避免小实验覆盖正式结果。

【为什么这里不把 checkpoint 落盘】
    两个基座各 400 MB，对比实验只需要指标；权重留在内存里取 best 就够。
    真要服务化时加 --save_ckpt，只存 state_dict。
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401  ← 必须第一行：torch 要先于 numpy 加载（见 _bootstrap.py）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import OUT_DIR, TrainConfig, backbone_spec, ensure_backbone, resolve_device
from data import build_eval_loader, build_loaders, dedup_val, load_splits, stratified_subsample
from metrics import compute_metrics
from model import BackboneClassifier, count_params


def set_seed(seed: int):
    """三个随机源都要固定：python / numpy / torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 (y_true, y_pred, probs)；probs 用于 ECE 与置信度分析"""
    model.eval()
    ys, ps, all_probs = [], [], []
    for ids, mask, y in loader:
        logits = model(ids.to(device), mask.to(device))
        probs = F.softmax(logits.float(), dim=-1).cpu()
        all_probs.append(probs)
        ps.extend(probs.argmax(-1).tolist())
        ys.extend(y.tolist())
    return np.array(ys), np.array(ps), torch.cat(all_probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["bert-base-chinese", "chinese-roberta-wwm-ext"])
    # 下面这些默认取 config.TrainConfig，只在做对照实验时才显式覆盖；
    # 正式对比两个基座时**一个都不要传**，否则破坏「同配置」前提。
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--train_size", type=int, default=None, help="训练样本数（分层抽样），0=全量")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_class_weight", action="store_true", help="关掉类别加权（消融用）")
    ap.add_argument("--device", default=None, help="auto/mps/cuda/cpu")
    ap.add_argument("--tag", default="", help="文件名后缀，如 smoke/quick")
    ap.add_argument("--save_ckpt", action="store_true", help="额外保存 best state_dict")
    ap.add_argument("--threads", type=int, default=0, help="CPU 线程数，0=torch 默认")
    # 冒烟/调试
    ap.add_argument("--smoke", action="store_true", help="2000 训练 / 300 验证 / 20 步 / 不落正式产物")
    ap.add_argument("--limit_val", type=int, default=0, help="只评前 N 条验证数据（调试用）")
    ap.add_argument("--max_steps_per_epoch", type=int, default=0, help="每个 epoch 最多多少步（调试用）")
    args = ap.parse_args()

    # ── 配置：默认全部来自 TrainConfig，保证两个基座逐字段一致 ──
    cfg = TrainConfig()
    for field_name in ("epochs", "batch_size", "lr", "max_length", "train_size", "seed", "device"):
        v = getattr(args, field_name)
        if v is not None:
            setattr(cfg, field_name, v)
    if args.no_class_weight:
        cfg.class_weight = False

    max_steps, limit_val = args.max_steps_per_epoch, args.limit_val
    if args.smoke:
        # 冒烟只设默认值：显式传了 --max_steps_per_epoch 就以传的为准
        cfg.train_size = 2000
        limit_val = limit_val or 300
        max_steps = max_steps or 20
        args.tag = args.tag or "smoke"

    spec = backbone_spec(args.backbone)
    out_suffix = f"_{args.tag}" if args.tag else ""

    set_seed(cfg.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(cfg.device)

    model_dir = ensure_backbone(args.backbone)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))

    train_all, val = load_splits()
    clean_val, n_dup = dedup_val(train_all, val)
    train = stratified_subsample(train_all, cfg.train_size, seed=cfg.seed)
    if limit_val:
        val, clean_val = val[:limit_val], [d for d in clean_val if d["idx"] < limit_val]

    train_loader, val_loader = build_loaders(train, val, tok, cfg.max_length, cfg.batch_size, seed=cfg.seed)

    model = BackboneClassifier(model_dir, pool=cfg.pool, dropout=cfg.dropout).to(device)
    params = count_params(model)

    # 分层学习率：底座小步、分类头大步（头是随机初始化的）
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg.lr},
         {"params": head_params, "lr": cfg.lr * cfg.head_lr_mult}],
        weight_decay=cfg.weight_decay,
    )

    # 类别加权：23 倍不均衡，不加权尾部类 recall 接近 0
    weight = None
    if cfg.class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(15), y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float, device=device)

    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

    steps_per_epoch = min(len(train_loader), max_steps) if max_steps else len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: step / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup)),
    )

    print(
        f"[cfg] backbone={args.backbone} device={device} threads={torch.get_num_threads()} "
        f"max_len={cfg.max_length} bs={cfg.batch_size} lr={cfg.lr} head×{cfg.head_lr_mult} "
        f"epochs={cfg.epochs} cw={'on' if weight is not None else 'off'} seed={cfg.seed}\n"
        f"[data] train={len(train)}/{len(train_all)}（分层抽样） val={len(val)} "
        f"(与训练集重复 {n_dup} 条，另报去重口径) params={params['total_m']}M steps/epoch={steps_per_epoch}",
        flush=True,
    )

    history, best = [], {"macro_f1": -1.0, "state": None, "epoch": 0}
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, (ids, mask, y) in enumerate(train_loader):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(ids, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 短文本偶发大梯度，裁一下更稳
            opt.step()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 20 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if max_steps and n_step >= max_steps:
                break

        y_true, y_pred, _ = evaluate(model, val_loader, device)
        val_f1 = compute_metrics(y_true, y_pred)["macro_f1"]
        history.append({
            "epoch": epoch,
            "train_loss": round(running / max(1, n_step), 4),
            "val_acc": round(float((y_true == y_pred).mean()), 4),
            "val_macro_f1": val_f1,
            "lr_last": round(opt.param_groups[0]["lr"], 8),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  → epoch {epoch}: val_acc={history[-1]['val_acc']} val_macro_f1={val_f1} "
              f"({history[-1]['elapsed_s']}s)", flush=True)

        if val_f1 > best["macro_f1"]:
            best = {"macro_f1": val_f1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    train_time = time.time() - t_start

    # ── 用 best 权重在完整验证集上出最终指标 ──
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    y_true, y_pred, probs = evaluate(model, val_loader, device)
    result = compute_metrics(y_true, y_pred, probs=probs)

    # ── 附加口径：去掉与训练集重复的验证样本（11%），确认结论不是靠背答案 ──
    clean_meta = {"n": len(clean_val), "n_dup": n_dup, "macro_f1": None, "acc": None}
    if clean_val and not limit_val:
        cy, cp, _ = evaluate(model, build_eval_loader(clean_val, tok, cfg.max_length), device)
        cm = compute_metrics(cy, cp)
        clean_meta.update({"macro_f1": cm["macro_f1"], "acc": cm["acc"]})

    payload = {
        "backbone": args.backbone,
        "display": spec["display"],
        "note": spec["note"],
        "model_dir": str(model_dir),
        "config": cfg.to_dict(),
        "params": params,
        "data": {
            "n_train": len(train),
            "n_train_available": len(train_all),
            "n_val": len(val),
            "n_val_dedup": len(clean_val),
            "n_dup_in_val": n_dup,
            "limit_val": limit_val,
            "max_steps_per_epoch": max_steps,
        },
        "best_epoch": best["epoch"],
        "train_time_s": round(train_time, 1),
        "history": history,
        "metrics": result,
        "metrics_dedup_val": clean_meta,
        "pred": {
            "idx": [d["idx"] for d in val],
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "conf": [round(float(c), 4) for c in probs.max(axis=1)],
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpath = OUT_DIR / f"metrics_{args.backbone}{out_suffix}.json"
    mpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.tag:
        (OUT_DIR / f"pred_{args.backbone}.json").write_text(
            json.dumps(payload["pred"], ensure_ascii=False), encoding="utf-8")
    if args.save_ckpt and best["state"] is not None:
        # ckpt 里带上 backbone 与指标：推理脚本不用再问「这是哪个基座训的」
        torch.save({
            "state_dict": best["state"],
            "backbone": args.backbone,
            "display": spec["display"],
            "config": cfg.to_dict(),
            "metrics": {k: result[k] for k in ("acc", "macro_f1", "weighted_f1")},
            "best_epoch": best["epoch"],
        }, OUT_DIR / f"ckpt_{args.backbone}{out_suffix}.pt")
        print(f"          权重 → {OUT_DIR / f'ckpt_{args.backbone}{out_suffix}.pt'}", flush=True)

    print(
        f"\n[{spec['display']}] best_epoch={best['epoch']}/{cfg.epochs} "
        f"acc={result['acc']} macro_f1={result['macro_f1']} weighted_f1={result['weighted_f1']}\n"
        f"          去重验证集({clean_meta['n']}条) macro_f1={clean_meta['macro_f1']} "
        f"训练耗时={payload['train_time_s']}s\n          指标 → {mpath}",
        flush=True,
    )


if __name__ == "__main__":
    main()

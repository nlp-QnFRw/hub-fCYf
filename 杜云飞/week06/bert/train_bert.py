#region 0. 模块说明
"""
L2：BERT 微调训练入口（自写训练循环，**不用 transformers 的 Trainer**）。

为什么不用 Trainer：
  transformers 5.x 把 `TrainingArguments` 的一批参数改过名（evaluation_strategy → eval_strategy、
  torch_dtype → dtype 等），依赖它的脚本升一次级就废；而本任务只需要"分组 lr + warmup +
  梯度累积 + 裁剪 + AMP"这四件事，自己写 60 行完全可控，还顺便能加显存断言。

消融开关（一次跑一个变量，其余固定）：
  --backbone {bert-base|roberta-wwm|macbert}   主干
  --max_len 128/256/384/512                    长度
  --truncation head|head_tail|title_only       截断策略
  --pool cls|mean                              池化
  --lr / --warmup_ratio / --dropout / --batch_size / --epochs
  --freeze_layers N                            冻结底部 N 层
  --class_weight none|balanced                 类别不均衡补偿

显存纪律（本机实测，越线时吞吐断崖式下跌而不是报 OOM）：
  bs=32/L=256 → 4.3GB / 121 samples/s；bs=64/L=256 → 7.2GB / 36 samples/s。
  因此每 50 step 断言峰值 < 6.8GB，越线直接抛错，别让它悄悄变成 3 倍慢。

使用方式：
  python bert/train_bert.py --train_split train_100k --epochs 3 --variant 100k-3ep
  python bert/train_bert.py --train_split sub_30k --epochs 2 --lr 3e-5 --variant 30k-lr3e5
  python bert/train_bert.py --demo          # 500 条 / 1 epoch / 30 步，验证流程
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bert.model_bert import build_model, get_tokenized, make_param_groups, run_eval  # noqa: E402
from common.contract import append_index, new_run_id, write_meta, write_preds, write_train_log  # noqa: E402
from common.data import build_dataloaders, label_counts, load_split  # noqa: E402
from common.labels import ID2LABEL, LABELS  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
from common.paths import run_dir  # noqa: E402
from common.seed import set_seed  # noqa: E402

#endregion

#region 2. 全局配置
MEM_LIMIT_GB: float = 6.8  # 显存硬上限（详见模块说明）
VAL_EVAL_LIMIT: int = 10_000  # 每 epoch 评估 val 的条数（全量 10k 约 40s）
#endregion

#region 3. 参数解析
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BERT 微调（多消融开关）")
    ap.add_argument("--backbone", default="bert-base", help="bert-base / roberta-wwm / macbert")
    ap.add_argument("--train_split", default="train_100k")
    ap.add_argument("--val_split", default="val_10k")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--truncation", default="head_tail", choices=["head", "head_tail", "title_only"])
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=3.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freeze_layers", type=int, default=0)
    ap.add_argument("--class_weight", default="none", choices=["none", "balanced"])
    ap.add_argument("--num_val", type=int, default=VAL_EVAL_LIMIT)
    ap.add_argument("--max_steps", type=int, default=None, help="限制每 epoch 步数（冒烟）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tier", default="t2")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--no_save", action="store_true", help="不落盘 checkpoint")
    ap.add_argument("--demo", action="store_true", help="冒烟：500 条 / 1 epoch / 30 步")
    return ap.parse_args()
#endregion

#region 4. 单 epoch 训练
def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum: int,
                    max_steps: int | None, class_weight, epoch: int, total_epochs: int) -> dict:
    """跑一个 epoch，返回 {loss, steps, elapsed_s, peak_mem_gb}。

    要点：
      - 用 `loss / grad_accum` 累积，最后余数批次单独补一次更新（否则末尾梯度被丢掉）；
      - `clip_grad_norm_` 必须在 `scaler.unscale_` 之后、`step` 之前；
      - 每 50 step 检查显存峰值，越线直接抛错（见模块说明）。
    """
    model.train()
    total_loss, n_seen, step = 0.0, 0, 0
    t0 = time.time()
    n_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        ids = batch["input_ids"].to(device, non_blocking=True)
        am = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        # 用 bf16 而不是 fp16：Blackwell 上两者吞吐几乎相同（实测 117 vs 121 samples/s），
        # 但 bf16 动态范围和 fp32 一致，**不需要 GradScaler**，也就没有
        # "Attempting to unscale FP16 gradients" 与"首个 LR 被 scheduler 跳过"这两类坑。
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(ids, am, labels=y, class_weight=class_weight)
            loss = out["loss"]
        (loss / grad_accum).backward()
        total_loss += loss.item() * ids.size(0)
        n_seen += ids.size(0)

        is_last = (i + 1) == n_batches
        if (i + 1) % grad_accum == 0 or is_last:  # 余数批次补更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 50 == 0:
                peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                if peak > MEM_LIMIT_GB:
                    raise RuntimeError(
                        f"显存峰值 {peak:.2f}GB 超过 {MEM_LIMIT_GB}GB 上限。"
                        f"继续跑会静默降级到共享内存（吞吐掉 3 倍），请减小 batch_size 或 max_len。")
                print(f"  ep{epoch}/{total_epochs} step {step:>5} loss {loss.item():.4f} "
                      f"peak {peak:.2f}GB {n_seen / (time.time() - t0):.0f} samp/s", flush=True)

        if max_steps and step >= max_steps:
            print(f"  到达 --max_steps {max_steps}，提前结束本 epoch")
            break

    return {"loss": total_loss / max(1, n_seen), "steps": step,
            "elapsed_s": time.time() - t0,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0}
#endregion

#region 5. 主流程
def main() -> None:
    args = parse_args()
    if args.demo:
        args.train_split, args.epochs, args.max_steps = "train_100k", 1, 30
        args.num_val = 500
        print("[Demo 模式] 500 条训练 / 30 步 / 1 epoch，产物不可用于对比")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = args.variant or (f"{args.backbone}-l{args.max_len}-{args.train_split.replace('train_', '')}"
                              f"-{args.epochs}ep-{args.pool}")
    run_id = args.run_id or new_run_id(args.tier, "bert", variant)
    print(f"run_id = {run_id}\n设备 {device}")

    # ---- 数据 ----
    t0 = time.time()
    train_ds, train_recs = get_tokenized(args.train_split, args.max_len, args.truncation, args.backbone)
    if args.demo:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(500, len(train_ds)))))
        train_recs = train_recs[:500]
    val_ds, _ = get_tokenized(args.val_split, args.max_len, args.truncation, args.backbone,
                              limit=args.num_val)
    print(f"训练 {len(train_ds)} 条 / 验证 {len(val_ds)} 条，tokenize+读盘 {time.time() - t0:.0f}s")

    from transformers import AutoTokenizer

    from common.paths import model_path

    pad_id = AutoTokenizer.from_pretrained(model_path(args.backbone)).pad_token_id
    train_loader = build_dataloaders(train_ds, pad_id, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = build_dataloaders(val_ds, pad_id, args.batch_size, shuffle=False)

    # ---- 模型 ----
    model = build_model(args.backbone, pool=args.pool, dropout=args.dropout,
                        freeze_layers=args.freeze_layers).to(device)

    class_weight = None
    if args.class_weight == "balanced":
        # 反频率加权：count 少的类权重大。会让 macro-F1 涨、acc 可能跌——这正是要观察的取舍
        cnt = label_counts(train_recs)
        w = np.array([len(train_recs) / (len(LABELS) * max(1, cnt[name])) for name in LABELS],
                     dtype=np.float32)
        class_weight = torch.tensor(w, device=device)
        print("[class_weight] " + " ".join(f"{k}:{v:.2f}" for k, v in zip(LABELS, w)))

    groups = make_param_groups(model, args.lr, args.head_lr_mult, args.weight_decay)
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = args.max_steps * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))  # linear warmup + linear decay

    # ---- 训练 ----
    log: list[dict] = []
    best_f1, best_ep = -1.0, -1
    ckpt_path = run_dir(run_id) / "best.pt"
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                             args.grad_accum, args.max_steps, class_weight, ep, args.epochs)
        ev = run_eval(model, val_loader, device)
        golds = [ID2LABEL[int(g)] for g in ev["golds"]]
        preds = [ID2LABEL[int(p)] for p in ev["preds"]]
        m = compute_metrics(golds, preds, ev["probs"])
        rec = {"epoch": ep, "train_loss": round(tr["loss"], 4), "val_loss": round(ev["loss"], 4),
               "val_acc": m["acc"], "val_macro_f1": m["macro_f1"],
               "elapsed_s": round(tr["elapsed_s"], 1), "peak_mem_gb": round(tr["peak_mem_gb"], 2)}
        log.append(rec)
        print(f"[ep{ep}] train_loss {rec['train_loss']:.4f} | val_loss {rec['val_loss']:.4f} "
              f"val_acc {rec['val_acc']:.4f} val_macroF1 {rec['val_macro_f1']:.4f} | "
              f"{rec['elapsed_s']:.0f}s peak {rec['peak_mem_gb']:.2f}GB")

        if m["macro_f1"] > best_f1:
            best_f1, best_ep = m["macro_f1"], ep
            if not args.no_save:
                torch.save({"state_dict": model.state_dict(), "epoch": ep,
                            "val_macro_f1": best_f1, "args": vars(args)}, ckpt_path)

    # ---- 用最优 checkpoint 出最终预测（保证汇报的数字来自选模型的那一版） ----
    if ckpt_path.exists() and not args.no_save:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"已加载最优 checkpoint（ep{best_ep}, val_macroF1={best_f1:.4f}）")

    for split in ("test_nat_10k", "test_bal"):
        recs = load_split(split)
        ds, _ = get_tokenized(split, args.max_len, args.truncation, args.backbone)
        loader = build_dataloaders(ds, pad_id, args.batch_size, shuffle=False)
        ev = run_eval(model, loader, device)
        preds = [{
            "id": r["id"], "gold": r["label"], "pred": ID2LABEL[int(p)],
            "raw": f"p={float(pr.max()):.3f}", "status": "ok",
            "latency_ms": ev["ms_per_sample"],
        } for r, p, pr in zip(recs, ev["preds"], ev["probs"])]
        write_preds(run_id, split, preds)
        np.save(run_dir(run_id) / f"probs_{split}.npy", ev["probs"])  # 供 ECE 使用
        acc = float(np.mean([p["pred"] == p["gold"] for p in preds]))
        macro = compute_metrics([p["gold"] for p in preds], [p["pred"] for p in preds])["macro_f1"]
        print(f"  {split}: acc={acc:.4f} macro_f1={macro:.4f} ({ev['ms_per_sample']:.2f} ms/条)")

    write_meta(run_id, tier=args.tier, method="bert", variant=variant,
               backbone=args.backbone, n_train=len(train_ds), seed=args.seed,
               hyperparams={k: v for k, v in vars(args).items()
                            if k not in ("run_id", "tier", "variant", "demo", "no_save")},
               best_epoch=best_ep, best_val_macro_f1=round(best_f1, 6),
               elapsed_s=round(sum(r["elapsed_s"] for r in log), 1),
               peak_mem_gb=round(max((r["peak_mem_gb"] for r in log), default=0.0), 2),
               device=str(device), cost_usd=0.0)
    write_train_log(run_id, log)
    append_index(run_id, {"method": "bert", "backbone": args.backbone})

    print(f"\nrun_id = {run_id}")
    print(f"下一步：python eval/evaluate.py --run {run_id}")


if __name__ == "__main__":
    main()
#endregion

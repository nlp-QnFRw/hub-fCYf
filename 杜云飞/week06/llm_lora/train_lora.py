#region 0. 模块说明
"""
L4：Qwen2-0.5B-Instruct 的 LoRA 微调。

为什么只做 LoRA、不做全量微调（**这不是偷懒，是算术**）：
  0.5B 参数全量微调 = 494M × (fp16 权重 2 + 梯度 2 + AdamW fp32 状态 8) ≈ 5.9GB，
  再加激活与 logits，8GB 卡上物理不可行。LoRA(r=8) 只训 1.1M 参数（0.22%）。

超参固定项（本机实测的甜点，见 docs/PROBE.md）：
  bs=2 / max_len=384 / grad_accum=8 / lr=2e-4 / fp16 / lora_dropout=0.05
  可调项：r、shots（训练时固定 0）、target_modules、epochs、训练集规模

验收要点：
  1. 训练曲线看 val_loss 是否在第 2~3 个 epoch 反弹（反弹说明过拟合，最优轮数要改）；
  2. 训练完立即在 test_bal/test_nat 上出预测，**用同一个解析口径**与 zero-shot 对齐；
  3. 保存 `adapter/`（含 adapter_config.json + tokenizer），评估端靠目录结构判类型。

使用方式：
  python llm_lora/train_lora.py --train_split sub_10k --epochs 3 --r 8
  python llm_lora/train_lora.py --train_split sub_1k --epochs 3 --variant 1k
  python llm_lora/train_lora.py --demo        # 200 条 / 20 步
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import append_index, new_run_id, write_meta, write_preds, write_train_log  # noqa: E402
from common.data import load_split  # noqa: E402
from common.paths import run_dir  # noqa: E402
from common.prompt import build_messages, parse_label  # noqa: E402
from common.seed import set_seed  # noqa: E402
from llm_lora.model_lora import (SFTDataset, attach_lora, batch_predict,  # noqa: E402
                                 collate_sft, load_base)

#endregion

#region 2. 全局配置
MEM_LIMIT_GB: float = 6.5
#endregion

#region 3. 参数解析
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Qwen2-0.5B LoRA SFT")
    ap.add_argument("--model", default="qwen2-0.5b")
    ap.add_argument("--train_split", default="sub_10k")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--targets", default="qkvo", choices=["qkvo", "qv"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_chars", type=int, default=300)
    ap.add_argument("--truncation", default="head_tail")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--num_val", type=int, default=500, help="每 epoch 验证条数（生成式评估贵，不要全量）")
    ap.add_argument("--splits", nargs="+", default=["test_bal", "test_nat_10k"])
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--tier", default="t3")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--demo", action="store_true")
    return ap.parse_args()
#endregion

#region 4. 评估（生成式，只跑 num_val 条）
def quick_val(tk, model, device, n: int) -> dict:
    """在 val_10k 的前 n 条上算 loss 与准确率（用同一批固定样本，保证跨 epoch 可比）。"""
    recs = load_split("val_10k", limit=n)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, n_tok, n_ok = 0.0, 0, 0
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(recs), 8):
            chunk = recs[i: i + 8]
            ds = SFTDataset(chunk, tk, max_len=args_max_len, max_chars=args_max_chars)
            batch = collate_sft([ds[j] for j in range(len(ds))], tk.pad_token_id)
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lab = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=am)
            logits = out.logits[:, :-1].contiguous()
            tgt = lab[:, 1:].contiguous()
            total_loss += loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1)).item()
            n_tok += 1
            # 逐条判对错：取 response 段的首个 token 做 argmax 不足以判断完整类名，
            # 所以这里只用 loss；准确率由 final 预测阶段给出（避免双口径）
    return {"val_loss": total_loss / max(1, n_tok)}


# 供 quick_val 使用（避免把 20 个参数一路传下去）
args_max_len: int = 384
args_max_chars: int = 300
#endregion

#region 5. 主流程
def main() -> None:
    global args_max_len, args_max_chars
    args = parse_args()
    args_max_len, args_max_chars = args.max_len, args.max_chars
    if args.demo:
        args.train_split, args.epochs, args.max_steps, args.num_val = "sub_5k", 1, 20, 100
        print("[Demo 模式] 5k 训练集 / 20 步 / 1 epoch，产物不可用于对比")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = args.variant or (f"{args.train_split.replace('sub_', '')}-r{args.r}-"
                              f"{args.epochs}ep-{args.targets}")
    run_id = args.run_id or new_run_id(args.tier, "lora", variant)
    print(f"run_id = {run_id}\n设备 {device}")

    tk, base = load_base(args.model)
    base = base.to(device)
    model = attach_lora(base, r=args.r, targets=args.targets)

    train_recs = load_split(args.train_split)
    print(f"训练集 {len(train_recs)} 条（{args.train_split}）")
    t0 = time.time()
    train_ds = SFTDataset(train_recs, tk, max_len=args.max_len, max_chars=args.max_chars,
                          truncation=args.truncation)
    print(f"样本构造完成 {time.time() - t0:.1f}s，平均序列长 "
          f"{sum(x['input_ids'].numel() for x in train_ds.items) / len(train_ds):.0f} token")

    from torch.utils.data import DataLoader

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate_sft(b, tk.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup = max(1, int(total_steps * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    log: list[dict] = []
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, seen, step = 0.0, 0, 0
        t_ep = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lab = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(input_ids=ids, attention_mask=am, labels=lab).loss
            (loss / args.grad_accum).backward()
            ep_loss += loss.item() * ids.size(0)
            seen += ids.size(0)

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    if peak > MEM_LIMIT_GB:
                        raise RuntimeError(
                            f"显存峰值 {peak:.2f}GB > {MEM_LIMIT_GB}GB：继续会静默降级到共享内存。"
                            f"请减小 max_len 或 batch_size。")
                    print(f"  ep{ep} step {step} loss {loss.item():.4f} peak {peak:.2f}GB "
                          f"{seen / (time.time() - t_ep):.1f} samp/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                print(f"  到达 --max_steps {args.max_steps}")
                break

        v = quick_val(tk, model, device, args.num_val)
        rec = {"epoch": ep, "train_loss": round(ep_loss / max(1, seen), 4),
               "val_loss": round(v["val_loss"], 4), "elapsed_s": round(time.time() - t_ep, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
               if torch.cuda.is_available() else 0.0}
        log.append(rec)
        print(f"[ep{ep}] train_loss {rec['train_loss']:.4f} val_loss {rec['val_loss']:.4f} "
              f"{rec['elapsed_s']:.0f}s peak {rec['peak_mem_gb']:.2f}GB")

    # ---- 保存 adapter ----
    adapter_dir = run_dir(run_id) / "adapter"
    if not args.no_save:
        model.save_pretrained(str(adapter_dir))  # 生成 adapter_config.json + 权重
        tk.save_pretrained(str(adapter_dir))     # tokenizer 一起存，评估端不用再传
        print(f"[保存] {adapter_dir}")

    # ---- 出预测 ----
    model.eval()
    for split in args.splits:
        recs = load_split(split)
        preds: list[dict] = []
        t_split = time.time()
        for i in range(0, len(recs), 16):
            chunk = recs[i: i + 16]
            prompts = [tk.apply_chat_template(build_messages(r, shots=0, max_chars=args.max_chars,
                                                             truncation=args.truncation),
                                              tokenize=False, add_generation_prompt=True)
                       for r in chunk]
            texts, dt = batch_predict(tk, model, prompts, device)
            per_ms = dt / max(1, len(chunk)) * 1000
            for r, txt in zip(chunk, texts):
                pred, status = parse_label(txt, mode="lenient")
                preds.append({"id": r["id"], "gold": r["label"], "pred": pred,
                              "raw": txt.replace("\n", " "), "status": status,
                              "latency_ms": per_ms})
        write_preds(run_id, split, preds)
        acc = sum(1 for p in preds if p["pred"] == p["gold"]) / max(1, len(preds))
        n_fail = sum(1 for p in preds if p["pred"] is None)
        print(f"  {split}: acc={acc:.4f} 拒答 {n_fail} ({n_fail / max(1, len(preds)) * 100:.1f}%) "
              f"{len(recs) / max(1e-6, time.time() - t_split):.1f} 条/s")

    write_meta(run_id, tier=args.tier, method="lora", backbone=args.model,
               n_train=len(train_recs), seed=42,
               hyperparams={k: v for k, v in vars(args).items()
                            if k not in ("run_id", "tier", "variant", "demo", "no_save")},
               elapsed_s=round(sum(r["elapsed_s"] for r in log), 1),
               peak_mem_gb=round(max((r["peak_mem_gb"] for r in log), default=0.0), 2),
               device=str(device), cost_usd=0.0)
    write_train_log(run_id, log)
    append_index(run_id, {"method": "lora", "r": args.r})

    print(f"\nrun_id = {run_id}")
    print(f"下一步：python eval/evaluate.py --run {run_id}")


if __name__ == "__main__":
    main()
#endregion

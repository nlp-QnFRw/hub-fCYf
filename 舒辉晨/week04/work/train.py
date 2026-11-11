"""
train.py -- 训练 decoder-only Transformer 语言模型

用法:
    python train.py                                    # 默认配置，从头训练
    python train.py --steps 3000 --batch-size 48
    python train.py --resume                            # 从 runs/last.pt 继续
    python train.py --device cpu --steps 200            # 快速验证能跑通

相比课件里的 LSTM 训练脚本，这里补齐了几件训练 RNN/Transformer 都必需的事：
    · 梯度裁剪   —— 防止个别 batch 把参数打飞
    · 学习率调度 —— 线性 warmup + 余弦退火
    · 断点续训   —— last.pt 存优化器/调度器/步数，中断了能接着跑
    · 参数分组   —— bias 和 LayerNorm 不做 weight decay
    · 混合精度   —— GPU 上自动开启 AMP（CPU 上自动跳过）
    · 评估口径   —— 固定窗口、固定顺序、按 token 加权
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch

import data as data_mod
from data import eval_windows, get_batch
from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))


class _NullBar:
    """没装 tqdm 时的空进度条，保证脚本在最小依赖下也能跑。"""

    def update(self, n=1): pass
    def set_postfix(self, **kw): pass
    def close(self): pass
    @staticmethod
    def write(msg): print(msg)


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it=None, **kw):
        return it if it is not None else _NullBar()
    tqdm.write = _NullBar.write


# ─────────────────────────── 学习率 ───────────────────────────

def lr_at(step: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """线性 warmup 到 base_lr，再余弦退火到 min_lr。"""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


# ─────────────────────────── 混合精度兼容 ───────────────────────────

def amp_ctx(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ─────────────────────────── 评估 ───────────────────────────

@torch.no_grad()
def evaluate(model: GPT, tokens: torch.Tensor, block_size: int, batch_size: int,
             device: torch.device, max_batches: int | None = None,
             amp: bool = False) -> tuple[float, int]:
    """返回 (平均每 token 交叉熵, 参与统计的 token 数)。

    按 token 加权而不是按 batch 平均 —— 最后一个 batch 可能不满，直接平均会偏。
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(eval_windows(tokens, block_size, batch_size)):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with amp_ctx(amp):
            _, loss = model(x, y)
        total_loss += float(loss.item()) * y.numel()
        total_tokens += y.numel()
    model.train()
    if total_tokens == 0:
        raise RuntimeError("验证集为空，无法评估")
    return total_loss / total_tokens, total_tokens


# ─────────────────────────── 存档 ───────────────────────────

def save_ckpt(path: str, model: GPT, optimizer, scaler, step: int, best_val: float,
              args: argparse.Namespace, val_tokens: int, with_optimizer: bool = True):
    """with_optimizer=False 用于 best.pt：推理只需要权重，省掉优化器状态（体积约 1/3）。"""
    payload = {
        "model": model.state_dict(),
        "config": model.cfg.to_dict(),
        "step": step,
        "best_val": best_val,
        "val_tokens": val_tokens,
        "train_args": {k: v for k, v in vars(args).items()},
    }
    if with_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scaler"] = scaler.state_dict() if scaler is not None else None
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # 原子替换，避免中途崩溃留下半截文件


def main():
    ap = argparse.ArgumentParser(description="训练字符级 Transformer 语言模型")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))

    # 规模
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    # 优化
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # 流程
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=40,
                    help="每次评估最多跑多少个 batch，0 表示全量")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--amp", type=int, default=1, help="1=GPU 上用混合精度")
    ap.add_argument("--resume", action="store_true", help="从 out-dir/last.pt 继续")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    # ── 环境
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    print(f"设备: {device}" + (f"  混合精度: {'开' if use_amp else '关'}"))

    # ── 数据
    meta = data_mod.load_meta(args.data_dir)
    train_tokens = data_mod.load_tokens(os.path.join(args.data_dir, "train.bin"))
    val_tokens = data_mod.load_tokens(os.path.join(args.data_dir, "val.bin"))
    print(f"语料: 训练 {len(train_tokens):,} token / 验证 {len(val_tokens):,} token "
          f"/ 字表 {meta['vocab_size']}")
    baseline = meta.get("baseline_ppl") or {}
    if baseline:
        print("N-gram 基线 PPL: " + "  ".join(f"{k}-gram={v}" for k, v in baseline.items()))

    # ── 模型
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"模型: {cfg.n_layer}层 {cfg.n_embd}维 {cfg.n_head}头 上下文{cfg.block_size}"
          f"  参数 {model.num_params:,}（其中非 embedding {model.num_params_non_embedding:,}）")
    if len(train_tokens) <= cfg.block_size + 1:
        raise SystemExit("训练语料太短，连一个 block 都切不出来")

    optimizer = model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    start_step, best_val = 0, float("inf")
    last_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best.pt")

    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"--resume 需要 {last_path}，但它不存在")
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"])
        best_val = float(ckpt["best_val"])
        print(f"从 {last_path} 续训，起始 step {start_step}，历史最佳 val loss {best_val:.4f}")

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": cfg.to_dict(), "train": vars(args)}, f,
                  ensure_ascii=False, indent=2, default=str)

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    csv_file = None
    if not args.no_csv:
        # 空文件也要补表头：中途被 Ctrl-C 打断会留下 0 字节的 csv，
        # 下次用 'a' 追加时如果只看 exists 就会永远缺表头。
        new_file = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if new_file:
            writer.writerow(["step", "lr", "train_loss", "val_loss", "val_ppl", "val_bpc", "sec"])

    # ── 训练循环
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    pbar = tqdm(total=args.steps, initial=start_step, desc="train", ncols=88,
                disable=None)      # disable=None: 输出被重定向到文件时自动静默
    t_start = time.time()
    running, running_tok = 0.0, 0
    step = start_step

    try:
        while step < args.steps:
            lr = lr_at(step, args.lr, args.min_lr, args.warmup, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_tokens, args.batch_size, cfg.block_size, gen)
            x, y = x.to(device), y.to(device)

            with amp_ctx(use_amp):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"step {step} 出现非有限 loss，训练发散；"
                                   f"尝试降低 --lr 或调小 --grad-clip")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * y.numel()
            running_tok += y.numel()
            step += 1
            pbar.update(1)

            if step % 20 == 0 or step == args.steps:
                avg = running / max(1, running_tok)
                pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(avg):.1f}",
                                 lr=f"{lr:.2e}", gnorm=f"{float(grad_norm):.2f}")
                running, running_tok = 0.0, 0

            if step % args.eval_interval == 0 or step == args.steps:
                val_loss, n_tok = evaluate(
                    model, val_tokens, cfg.block_size, args.batch_size, device,
                    args.eval_batches or None, use_amp)
                val_ppl = math.exp(val_loss)
                val_bpc = val_loss / math.log(2)
                flag = ""
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(best_path, model, optimizer, scaler, step, best_val, args, n_tok,
                              with_optimizer=False)
                    flag = "  <- best"
                save_ckpt(last_path, model, optimizer, scaler, step, best_val, args, n_tok)
                if csv_file:
                    writer.writerow([step, f"{lr:.6g}", f"{float(loss.item()):.4f}",
                                     f"{val_loss:.4f}", f"{val_ppl:.2f}", f"{val_bpc:.4f}",
                                     f"{time.time() - t_start:.1f}"])
                    csv_file.flush()
                pbar.set_postfix(loss=f"{float(loss.item()):.4f}", val=f"{val_loss:.4f}",
                                 ppl=f"{val_ppl:.1f}", lr=f"{lr:.2e}")
                tqdm.write(f"[step {step:>5}] val_loss {val_loss:.4f}  "
                           f"val_ppl {val_ppl:8.2f}  bpc {val_bpc:.4f}  "
                           f"{n_tok:,} token{flag}")
    except KeyboardInterrupt:
        print("\n中断，正在保存 last.pt ...")
        save_ckpt(last_path, model, optimizer, scaler, step, best_val, args, 0)
        raise
    finally:
        pbar.close()
        if csv_file:
            csv_file.close()

    # 收尾：再存一次 last（保证 step 是准确的），并做一次全量验证
    save_ckpt(last_path, model, optimizer, scaler, step, best_val, args, 0)
    val_loss, n_tok = evaluate(model, val_tokens, cfg.block_size, args.batch_size,
                               device, None, use_amp)
    print(f"\n完成 {step - start_step} 步，用时 {time.time() - t_start:.1f}s")
    print(f"全量验证: loss {val_loss:.4f}  PPL {math.exp(val_loss):.2f}  "
          f"bpc {val_loss / math.log(2):.4f}  ({n_tok:,} token)")
    if baseline:
        print(f"对比 1-gram 基线 PPL {baseline.get('1')}："
              f"{'低于' if math.exp(val_loss) < float(baseline.get('1', 1e9)) else '高于'}基线")
    print(f"检查点: {best_path}（最佳 val loss {best_val:.4f}） / {last_path}（可 --resume）")


if __name__ == "__main__":
    main()

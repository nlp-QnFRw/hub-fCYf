"""
bert_train.py -- 用同一份语料继续训练 bert-base-chinese（MLM 微调），并按阶段记录指标

用法:
    python bert_train.py                                  # 默认 600 步
    python bert_train.py --steps 1000 --eval-interval 200
    python bert_train.py --device cpu --steps 20 --max-len 64   # 只验证能跑

为什么要「继续训练」而不是从零训：
    bert-base-chinese 是在通用中文维基上预训练的，语料和这里完全不同。
    在本项目的问答语料上继续做 MLM，能直接观察到「预训练权重 + 领域数据」的收益，
    也才能和从零训练的 GPT 放在同一张表里看。

阶段指标（每个 eval-interval 记录一次）：
    val_loss / mask_acc   —— 遮字任务本身的损失和命中率（固定遮法，跨阶段可比）
    mlm_ppl               —— exp(val_loss)，只在被遮位置上统计
    pseudo_ppl            —— 伪困惑度：逐位置遮住、用双向上下文预测，和 GPT 的 PPL 同量纲
    tokens_seen           —— 到这一步为止模型看过多少 token，用来解释两边预算的差异
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

from bert_data import BertTok, BlockBank, build_blocks, mask_batch
from bert_model import BertConfig, load_pretrained, pseudo_perplexity, resolve_bert_dir
from train import amp_ctx, make_scaler, tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def linear_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    """BERT 微调标准调度：线性 warmup，然后线性衰减到 0。"""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * max(0.0, (total - step) / max(1, total - warmup))


def build_eval_set(bank: BlockBank, tok: BertTok, batch_size: int, limit: int,
                   prob: float, seed: int):
    """把验证集的遮字结果**一次性生成并固定下来**。

    如果每个阶段重新随机遮，loss 的波动会混进「训练有没有变好」的判断里，
    所以这里用固定种子遮一次，之后每个阶段都拿同一批样本评估。
    """
    gen = torch.Generator().manual_seed(seed)
    out = []
    for ids, am in bank.fixed_batches(batch_size, limit):
        inputs, labels = mask_batch(ids, am, tok, prob, gen)
        out.append((inputs, labels, am))
    return out


@torch.no_grad()
def evaluate(model, eval_set, device, amp) -> tuple[float, float, int]:
    model.eval()
    total_loss, n_masked, correct = 0.0, 0, 0
    for inputs, labels, am in eval_set:
        inputs, labels, am = inputs.to(device), labels.to(device), am.to(device)
        with amp_ctx(amp):
            logits, loss = model(inputs, am, labels=labels)
        n = int((labels != -100).sum())
        total_loss += float(loss.item()) * n
        n_masked += n
        pred = logits.argmax(dim=-1)
        correct += int((pred[labels != -100] == labels[labels != -100]).sum().item())
    if n_masked == 0:
        raise RuntimeError("评估集里没有被遮住的位置")
    return total_loss / n_masked, correct / n_masked, n_masked


def main():
    ap = argparse.ArgumentParser(description="bert-base-chinese MLM 继续训练")
    ap.add_argument("--bert-dir", default=None, help="默认自动查找 bert-base-chinese")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs_bert"))

    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--mask-prob", type=float, default=0.15)

    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-blocks", type=int, default=96, help="评估用多少个块")
    ap.add_argument("--pseudo-blocks", type=int, default=6,
                    help="算伪困惑度用几个块（逐位置遮，很费时间，别开太大）")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--save-model", type=int, default=1, help="1=保存 final.pt（约 410MB）")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    use_amp = bool(args.amp) and device.type == "cuda"
    if device.type == "cuda":
        print(f"设备: {device} ({torch.cuda.get_device_name(0)})  "
              f"混合精度: {'开' if use_amp else '关'}")
    else:
        print(f"设备: {device}  混合精度: 关")

    # ── 预训练模型
    bert_dir = resolve_bert_dir(args.bert_dir)
    print(f"预训练模型: {bert_dir}")
    t0 = time.time()
    model, cfg = load_pretrained(bert_dir, device=device)
    print(f"结构: {cfg.num_hidden_layers} 层 / {cfg.num_attention_heads} 头 / "
          f"{cfg.hidden_size} 维 / 词表 {cfg.vocab_size}  参数 {model.num_params:,}  "
          f"（加载 {time.time() - t0:.1f}s）")

    # ── 数据
    tok = BertTok(bert_dir)
    with open(os.path.join(args.data_dir, "train.txt"), encoding="utf-8") as f:
        train_lines = f.read().splitlines()
    with open(os.path.join(args.data_dir, "val.txt"), encoding="utf-8") as f:
        val_lines = f.read().splitlines()

    train_blocks = build_blocks(train_lines, tok, args.max_len)
    val_blocks = build_blocks(val_lines, tok, args.max_len)
    train_bank = BlockBank(train_blocks, tok, args.max_len)
    val_bank = BlockBank(val_blocks, tok, args.max_len)
    print(f"数据: 训练 {len(train_lines):,} 行 → {len(train_bank):,} 块 × {args.max_len} token"
          f" / 验证 {len(val_lines):,} 行 → {len(val_bank):,} 块")

    eval_set = build_eval_set(val_bank, tok, args.batch_size, args.eval_blocks,
                              args.mask_prob, args.seed)
    n_eval_masked = sum(int((lb != -100).sum()) for _, lb, _ in eval_set)
    pseudo_ids, pseudo_mask = val_bank.ids[: args.pseudo_blocks], val_bank.mask[: args.pseudo_blocks]
    print(f"评估集: {len(eval_set)} 个 batch，固定遮住 {n_eval_masked:,} 个位置"
          f" / 伪困惑度用 {pseudo_ids.size(0)} 个块")

    # ── 优化器
    optimizer = model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"bert_dir": bert_dir, "model": cfg.to_dict(), "train": vars(args)},
                  f, ensure_ascii=False, indent=2, default=str)
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    new_file = not os.path.exists(csv_path)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "lr", "train_loss", "val_loss", "mask_acc", "mlm_ppl",
                     "pseudo_ppl", "tokens_seen", "sec"])

    def run_stage(step: int, lr: float, train_loss: float, t_start: float):
        val_loss, acc, _ = evaluate(model, eval_set, device, use_amp)
        ppl = math.exp(val_loss)
        if pseudo_ids.size(0) > 0:
            p_ppl, n_pos = pseudo_perplexity(model, pseudo_ids.to(device),
                                             pseudo_mask.to(device), tok.mask_id)
        else:
            p_ppl, n_pos = float("nan"), 0
        tokens_seen = step * args.batch_size * args.max_len
        writer.writerow([step, f"{lr:.6g}", f"{train_loss:.4f}", f"{val_loss:.4f}",
                         f"{acc:.4f}", f"{ppl:.2f}", f"{p_ppl:.2f}", tokens_seen,
                         f"{time.time() - t_start:.1f}"])
        csv_file.flush()
        tqdm.write(f"[阶段 {step:>4}] MLM loss {val_loss:.4f}  命中率 {acc * 100:5.2f}%  "
                   f"mlm_ppl {ppl:8.2f}  伪PPL {p_ppl:8.2f}  "
                   f"（已看 {tokens_seen / 1e6:.2f}M token）")
        return val_loss, ppl, p_ppl

    # ── 训练
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    t_start = time.time()
    print("\n" + "=" * 78)
    print("阶段 0 = 未经任何训练的原版 bert-base-chinese，作为起点")
    print("=" * 78)
    run_stage(0, 0.0, float("nan"), t_start)

    pbar = tqdm(total=args.steps, desc="bert", ncols=88, disable=None)
    running, running_n = 0.0, 0
    for step in range(1, args.steps + 1):
        lr = linear_lr(step - 1, args.lr, args.warmup, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 遮字在 CPU 上做（随机数生成器也在 CPU），做完再搬到 GPU
        ids, am = train_bank.sample(args.batch_size, gen)
        inputs, labels = mask_batch(ids, am, tok, args.mask_prob, gen)
        inputs = inputs.to(device)
        labels = labels.to(device)
        am = am.to(device)

        with amp_ctx(use_amp):
            _, loss = model(inputs, am, labels=labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step} 出现非有限 loss，尝试降低 --lr")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.item())
        running_n += 1
        pbar.update(1)
        if step % 20 == 0:
            pbar.set_postfix(loss=f"{running / running_n:.4f}", lr=f"{lr:.2e}")
            running, running_n = 0.0, 0

        if step % args.eval_interval == 0 or step == args.steps:
            run_stage(step, lr, float(loss.item()), t_start)
    pbar.close()
    csv_file.close()

    print(f"\n训练完成 {args.steps} 步，用时 {time.time() - t_start:.1f}s")

    if args.save_model:
        path = os.path.join(args.out_dir, "final.pt")
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                    "bert_dir": bert_dir, "step": args.steps,
                    "note": "由 bert_train.py 在本项目语料上继续训练的 MLM 权重"}, path)
        print(f"已保存 {path}（{os.path.getsize(path) / 1024 / 1024:.0f} MB）")
    print(f"阶段指标: {csv_path}\n用 python compare.py 看两边的对比表")


if __name__ == "__main__":
    main()

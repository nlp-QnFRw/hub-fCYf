"""sft_train.py —— 在预训练字符级 Transformer 上做 SFT（指令微调）。

与 week04 的预训练 train.py 的区别：

    预训练 train.py                 SFT sft_train.py
    ------------------------------  ------------------------------------------
    从随机初始化开始                 从 checkpoints/best.pt 加载预训练权重
    整篇语料拼起来切块               一条样本 = 一条 instruction/output 问答
    每个位置都算 loss                只对「答：」之后的 token 算 loss（prompt mask）
    学习率 1e-4                      学习率 3e-4 + warmup + 余弦退火
    Adam                              AdamW（bias / LayerNorm 不做 weight decay）
    看 test loss / perplexity        看 val 答案 loss + Greedy 生成质量

用法::

    python code/sft_train.py                       # 使用 config/config.yaml
    python code/sft_train.py sft.epochs=5          # 命令行覆盖任意配置项
    python code/sft_train.py --limit-train-batches 10 --epochs 1   # 冒烟测试
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# 让 import 能找到 code/ 和 code/util/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "util"))

# Windows 下把输出重定向到文件/管道时默认 GBK 编码，"✓" 之类会抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

from my_module import MyModule  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from util.generate import generate_answer  # noqa: E402
from util.sft_dataset import (  # noqa: E402
    IGNORE_INDEX,
    SFTDataset,
    make_collate_fn,
    preview,
)


# ---------------------------------------------------------------- 基础工具

def set_seed(seed: int) -> None:
    """固定所有随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_param_groups(model: MyModule, weight_decay: float):
    """AdamW 惯例：bias 与 LayerNorm 参数不做 weight decay，其余权重衰减。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    """线性 warmup + 余弦退火：SFT 最常用的组合。"""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def display_path(path: Path) -> str:
    """尽量显示相对 week05/ 的短路径。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- 加载

def load_init_checkpoint(cfg):
    """加载预训练 checkpoint，并校验结构 / 词表与当前配置一致。"""
    init_path = Path(cfg.sft.init_ckpt)
    if not init_path.is_absolute():
        init_path = ROOT / init_path
    if not init_path.exists():
        raise FileNotFoundError(
            f"找不到预训练 checkpoint：{init_path}\n"
            f"请确认 checkpoints/ 下有 week04 训练好的 best.pt"
        )

    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    ckpt_cfg = OmegaConf.create(ckpt["cfg"])

    # 结构必须完全一致，否则 load_state_dict 会报 shape 不匹配；
    # 这里以 checkpoint 里的 cfg 为准，避免手改 config 后加载失败却看不出原因。
    for key in ("model_size", "num_heads", "d_ff", "act", "ln", "causal"):
        if cfg.transformer.get(key) != ckpt_cfg.transformer.get(key):
            print(
                f"  [警告] config.transformer.{key}={cfg.transformer.get(key)} "
                f"与 checkpoint 的 {ckpt_cfg.transformer.get(key)} 不一致，以 checkpoint 为准"
            )
    cfg.transformer = ckpt_cfg.transformer
    cfg.hparams.num_layers = ckpt_cfg.hparams.num_layers
    cfg.hparams.max_len = ckpt_cfg.hparams.max_len
    return ckpt, init_path


# ---------------------------------------------------------------- 评估

def evaluate_loss(model, loader, device, max_len: int) -> tuple[float, float]:
    """在 val 上计算「答案 token」平均交叉熵，返回 (loss, ppl)。"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += int((y != IGNORE_INDEX).sum().item())
    model.train()
    if total_tokens == 0:
        return float("inf"), float("inf")
    avg = total_loss / total_tokens
    return avg, math.exp(avg) if avg < 100 else float("inf")


@torch.no_grad()
def evaluate_generation(model, tok, records, device, max_len: int, max_new_tokens: int):
    """Greedy 生成前 N 条 val 样本，统计完全匹配率并返回样例。"""
    model.eval()
    hits, samples = 0, []
    for rec in records:
        pred = generate_answer(
            model,
            tok,
            rec["instruction"],
            device,
            max_len=max_len,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # 贪心解码，结果确定
        )
        ok = pred.strip() == rec["output"].strip()
        hits += int(ok)
        samples.append((rec["instruction"], rec["output"], pred, ok))
    model.train()
    return hits / max(1, len(records)), samples


# ---------------------------------------------------------------- 训练

def train(cfg, device, limit_batches: int | None = None) -> None:
    set_seed(cfg.hparams.seed)

    # 1) 预训练权重 + tokenizer -----------------------------------------
    print("=" * 72)
    print("SFT 训练开始")
    print("=" * 72)

    ckpt, init_path = load_init_checkpoint(cfg)
    tok = CharTokenizer.from_file(cfg)
    emb_rows = ckpt["model"]["embedding.weight"].shape[0]
    if len(tok) != emb_rows:
        raise RuntimeError(
            f"词表大小 {len(tok)} 与 checkpoint embedding 行数 {emb_rows} 不一致。\n"
            f"请确认 {cfg.hparams.vocab_path} 就是预训练时用的那份词表"
        )

    max_len = int(cfg.hparams.max_len)
    model = MyModule(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"权重加载异常：missing={missing}, unexpected={unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  预训练权重 : {init_path}")
    print(f"  epoch      : {ckpt.get('epoch', '?')} | val_loss : {ckpt.get('val_loss', float('nan')):.4f}")
    print(f"  模型参数   : {n_params:,} | vocab : {len(tok)} | max_len : {max_len}")
    print(f"  device     : {device}")

    # 2) 数据 -----------------------------------------------------------
    train_ds = SFTDataset(cfg, tok, "train")
    val_ds = SFTDataset(cfg, tok, "val")
    collate = make_collate_fn(tok)
    g = cfg.sft
    train_loader = DataLoader(
        train_ds,
        batch_size=g.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=g.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    print(
        f"  训练集     : {len(train_ds)} 条 | {train_ds.num_tokens:,} token "
        f"（其中监督 token {train_ds.num_answer_tokens:,}，"
        f"占比 {train_ds.num_answer_tokens / train_ds.num_tokens:.1%}）"
    )
    print(
        f"  验证集     : {len(val_ds)} 条 | {val_ds.num_tokens:,} token "
        f"（监督 {val_ds.num_answer_tokens:,}）"
    )
    if train_ds.truncated:
        print(f"  [提示] 训练集有 {train_ds.truncated} 条样本超出 max_len 被截断")

    print("\n  [mask 示例] [] 内的 token 不参与 loss：")
    example = train_ds.records[0]
    print("  " + preview(tok, example["instruction"], example["output"], max_len).replace("\n", "\n  "))
    print()

    # 3) 优化器 / 调度器 ------------------------------------------------
    grad_accum = max(1, int(g.grad_accum_steps))
    batches_per_epoch = len(train_loader)
    if limit_batches:
        batches_per_epoch = min(batches_per_epoch, limit_batches)
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    total_steps = steps_per_epoch * int(g.epochs)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, g.weight_decay),
        lr=g.lr,
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, total_steps, g.warmup_ratio, g.min_lr_ratio)
    print(
        f"  优化器     : AdamW lr={g.lr} wd={g.weight_decay} | "
        f"warmup {int(total_steps * g.warmup_ratio)} / 总 {total_steps} step"
    )

    # 4) 基线（预训练模型直接答题的效果，用于对比 SFT 收益） ------------
    base_loss, base_ppl = evaluate_loss(model, val_loader, device, max_len)
    print(f"  基线（SFT 前）val 答案 loss {base_loss:.4f} | ppl {base_ppl:.2f}")
    if g.eval_gen_samples and len(val_ds.records) >= g.eval_gen_samples:
        base_em, _ = evaluate_generation(
            model, tok, val_ds.records[: g.eval_gen_samples], device, max_len, g.max_new_tokens
        )
        print(f"  基线 Greedy 完全匹配率 {base_em:.1%}（{g.eval_gen_samples} 条）")
    print()

    # 5) 训练循环 -------------------------------------------------------
    save_path = Path(g.save_path)
    if not save_path.is_absolute():
        save_path = ROOT / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.resolve() == init_path.resolve():
        raise ValueError("sft.save_path 不能等于 sft.init_ckpt，否则会覆盖预训练权重")

    best_val_loss = float("inf")
    bad_epochs = 0
    patience = int(g.early_stop_patience)
    history = []

    for epoch in range(int(g.epochs)):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            # 关键：loss 只在 label != -100 的位置计算（即答案 + <eos>）
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
                label_smoothing=float(g.label_smoothing),
            )
            (loss / grad_accum).backward()

            running_loss += loss.item()

            if (i + 1) % grad_accum == 0 or (i + 1) == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(g.grad_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (i + 1) % int(g.log_every) == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch + 1} [{i + 1}/{batches_per_epoch}] | "
                    f"loss {loss.item():.4f} | lr {lr_now:.2e}"
                )

            if limit_batches and (i + 1) >= limit_batches:
                break  # 冒烟测试：只跑前 N 个 batch

        train_loss = running_loss / max(1, batches_per_epoch)
        val_loss, val_ppl = evaluate_loss(model, val_loader, device, max_len)
        elapsed = time.time() - t0

        em, samples = 0.0, []
        if g.eval_gen_samples:
            em, samples = evaluate_generation(
                model, tok, val_ds.records[: g.eval_gen_samples], device, max_len, g.max_new_tokens
            )

        print(
            f"Epoch {epoch + 1}/{g.epochs} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | ppl {val_ppl:.2f} | gen-EM {em:.1%} | {elapsed:.1f}s"
        )
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_ppl": val_ppl, "gen_em": em}
        )

        # 每轮展示一条生成样例，直观看到模型在学什么
        if samples:
            q, gold, pred, ok = samples[0]
            mark = "✓" if ok else "✗"
            print(f"  例 {mark} 问：{q[:60]}")
            print(f"       参考：{gold[:60]}")
            print(f"       生成：{pred[:60]}")

        # 保存 best（以 val 答案 loss 为准）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "gen_em": em,
                    "base_val_loss": base_loss,
                    "history": history,
                    "init_ckpt": str(init_path),
                },
                save_path,
            )
            print(f"  ✓ 新 best（val {val_loss:.4f}），已保存到 {display_path(save_path)}")
        else:
            bad_epochs += 1
            print(f"  val loss 未下降 ({bad_epochs}/{patience})")

        if patience > 0 and bad_epochs >= patience:
            print(f"\n早停：val loss 连续 {patience} 轮未下降，停止训练")
            break

    # 6) 总结 -----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"SFT 结束 | 基线 val loss {base_loss:.4f} (ppl {base_ppl:.2f}) → best {best_val_loss:.4f}")
    print(f"权重：{display_path(save_path)}")
    print("下一步：python code/demo.py          # 用 SFT 模型回答新问题")
    print("=" * 72)


# ---------------------------------------------------------------- 入口

def parse_args():
    parser = argparse.ArgumentParser(description="SFT 指令微调")
    parser.add_argument("--cfg", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 sft.epochs")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="只跑前 N 个 batch（冒烟测试）")
    parser.add_argument("--init", default=None, help="覆盖 sft.init_ckpt")
    parser.add_argument("--save", default=None, help="覆盖 sft.save_path")
    parser.add_argument("overrides", nargs="*", help="OmegaConf 风格覆盖，如 sft.lr=1e-4")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.epochs is not None:
        cfg.sft.epochs = args.epochs
    if args.init is not None:
        cfg.sft.init_ckpt = args.init
    if args.save is not None:
        cfg.sft.save_path = args.save

    limit_batches = args.limit_train_batches
    if limit_batches:
        # 冒烟测试：默认只跑 1 轮，不覆盖正式权重，落盘到 checkpoints/smoke.pt
        if args.epochs is None:
            cfg.sft.epochs = 1
        cfg.sft.save_path = "checkpoints/smoke.pt"
        print(f"[冒烟测试] 只跑前 {limit_batches} 个 batch，权重写到 checkpoints/smoke.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(cfg, device, limit_batches=limit_batches)


if __name__ == "__main__":
    main()

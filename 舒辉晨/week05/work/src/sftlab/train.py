"""通用训练循环：两个阶段共用同一套代码，差别只在数据集和初始化权重。

    阶段一 pretrain：PretrainDataset（全位置算 loss）+ 随机初始化
    阶段二 sft     ：SFTDataset（只算回答）+ 从阶段一的 best.pt 初始化

这种「同一个 trainer、换数据」的组织方式，正好对应 PPT 第 29 页那句：
SFT 没变架构、没变 attention mask、没变训练目标，变的是数据和 loss 的位置。
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch

from .data import (
    PretrainDataset,
    SFTDataset,
    make_loader,
    read_jsonl,
)
from .generate import chat, raw_continue
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .utils import (
    AverageMeter,
    Logger,
    describe_device,
    get_device,
    human_time,
    load_config,
    ppl,
    resolve,
    set_seed,
)


# ── 构建 ──────────────────────────────────────────────────────────────────────

def load_tokenizer(cfg: dict) -> CharTokenizer:
    path = resolve(cfg["data"]["vocab"])
    if not path.exists():
        raise FileNotFoundError(
            f"找不到词表 {path}，请先运行: python scripts/prepare_data.py"
        )
    return CharTokenizer.load(path)


def build_datasets(cfg: dict, tok: CharTokenizer, stage: str):
    train_rows = read_jsonl(resolve(cfg["data"]["train"]))
    val_rows = read_jsonl(resolve(cfg["data"]["val"]))
    block = int(cfg["model"]["block_size"])

    if stage == "pretrain":
        train_ds = PretrainDataset(train_rows, tok, block_size=block)
        val_ds = PretrainDataset(val_rows, tok, block_size=block)
    else:
        train_ds = SFTDataset(train_rows, tok, max_len=block)
        val_ds = SFTDataset(val_rows, tok, max_len=block)
    return train_ds, val_ds


def build_model(cfg: dict, vocab_size: int, device: torch.device) -> GPT:
    mcfg = GPTConfig.from_dict(cfg["model"], vocab_size=vocab_size)
    model = GPT(mcfg).to(device)
    return model


def load_init_weights(model: GPT, path: str | os.PathLike, tok: CharTokenizer) -> None:
    """SFT 从预训练检查点热启动。词表对不上就报错，而不是悄悄跳过。"""
    ckpt_path = resolve(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到预训练检查点 {ckpt_path}\n"
            f"请先跑阶段一: python scripts/train.py --stage pretrain"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["itos"] != tok.itos:
        raise ValueError("词表和检查点里的不一致：请确认两个阶段用的是同一份 vocab.json")
    model.load_state_dict(ckpt["model"])
    print(f"  已从 {ckpt_path.name} 加载权重（step={ckpt.get('step', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}）")


# ── 学习率 ────────────────────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup: int, total: int, min_lr_ratio: float):
    """线性 warmup + 余弦退火：起步太快容易炸，收尾太陡容易错过最优点。"""

    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, progress)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ── 评估 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: GPT, loader, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    total_loss, total_batches = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            _, loss = model(input_ids, labels)
        total_loss += float(loss)
        total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)


def preview(model: GPT, tok: CharTokenizer, stage: str, device: torch.device,
            sample: str, logger) -> None:
    """每个评估点生成一条样例，肉眼确认「模型现在会干什么」。"""
    if stage == "sft":
        out = chat(model, tok, sample, max_new_tokens=80, temperature=0.7, device=device)
    else:
        out = raw_continue(model, tok, sample, max_new_tokens=80, temperature=0.7, device=device)
    out = out.replace("\n", " ")
    logger(f"    ↳ 样例｜{sample[:24]}… → {out[:70]}{'…' if len(out) > 70 else ''}")
    model.train()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_train(
    config_path: str,
    stage: str,
    device_pref: str = "auto",
    epochs: int | None = None,
    max_steps: int | None = None,
    init_from: str | None = None,
    seed: int | None = None,
) -> dict:
    cfg = load_config(config_path)
    cfg["stage"] = stage
    if seed is not None:
        cfg["seed"] = seed

    tcfg = cfg["train"]
    if epochs is not None:
        tcfg["epochs"] = epochs
    set_seed(int(cfg["seed"]))

    device = get_device(device_pref)
    out_dir = resolve(tcfg.get("out_dir", f"outputs/{stage}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "train.log")

    logger("=" * 78)
    logger(f"阶段: {stage.upper()}   设备: {describe_device(device)}   种子: {cfg['seed']}")
    logger("=" * 78)

    tok = load_tokenizer(cfg)
    train_ds, val_ds = build_datasets(cfg, tok, stage)
    logger(f"词表 {len(tok)}  训练样本 {len(train_ds)}  验证样本 {len(val_ds)}")

    if stage == "sft":
        logger(f"loss mask: 平均只有 {train_ds.stats.supervised_ratio:.1%} 的 token 参与训练"
               f"（{train_ds.stats.supervised_tokens}/{train_ds.stats.total_tokens}）")

    model = build_model(cfg, len(tok), device)
    logger(f"模型 {model.num_params()/1e6:.2f}M 参数 · {cfg['model']['n_layer']} 层 · "
           f"d_model={cfg['model']['d_model']} · block={cfg['model']['block_size']}")

    init_path = init_from or (cfg.get("init_from") if stage == "sft" else None)
    if init_path:
        load_init_weights(model, init_path, tok)

    bs = int(tcfg["batch_size"])
    train_loader = make_loader(train_ds, bs, True, tok.pad_id, stage, int(tcfg.get("num_workers", 0)))
    val_loader = make_loader(val_ds, bs, False, tok.pad_id, stage, int(tcfg.get("num_workers", 0)))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(tcfg["epochs"])
    if max_steps:
        total_steps = min(total_steps, max_steps)

    optimizer = model.configure_optimizers(
        weight_decay=float(tcfg["weight_decay"]), lr=float(tcfg["lr"])
    )
    scheduler = make_scheduler(
        optimizer,
        warmup=int(tcfg.get("warmup_steps", 50)),
        total=total_steps,
        min_lr_ratio=float(tcfg.get("min_lr", 1e-4)) / float(tcfg["lr"]),
    )

    use_amp = device.type == "cuda"
    logger(f"计划 {total_steps} 步（{steps_per_epoch} 步/轮 × {tcfg['epochs']} 轮），"
           f"batch={bs}，AMP={'bf16' if use_amp else 'off'}")
    logger("-" * 78)

    preview_prompt = val_ds.examples[0].instruction if hasattr(val_ds, "examples") else "什么是人工智能？"

    global_step = 0
    best_val = float("inf")
    meter = AverageMeter()
    t_start = time.time()
    tokens_seen = 0
    history: list[dict] = []
    stop = False

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(input_ids, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            tokens_seen += input_ids.numel()
            meter.update(loss.detach().item())

            if global_step % int(tcfg.get("log_interval", 10)) == 0:
                elapsed = time.time() - t_start
                lr = scheduler.get_last_lr()[0]
                logger(
                    f"ep {epoch} step {global_step:5d}/{total_steps} "
                    f"loss {meter.avg:.4f} ppl {ppl(meter.avg):7.2f} "
                    f"lr {lr:.2e} {tokens_seen/max(elapsed,1e-9):7.0f} tok/s"
                )
                meter.reset()

            if global_step >= total_steps:
                stop = True
                break

        # 每轮结束评估一次：用验证集判断「是学会了还是背下来了」
        val_loss = evaluate(model, val_loader, device, int(tcfg.get("eval_batches", 9999)))
        logger(f"—— 第 {epoch} 轮结束 · val_loss {val_loss:.4f} · val_ppl {ppl(val_loss):.2f} · "
               f"用时 {human_time(time.time() - t_start)}")
        history.append({"epoch": epoch, "step": global_step, "val_loss": round(val_loss, 4)})

        ckpt = {
            "model": model.state_dict(),
            "model_cfg": vars(model.cfg),
            "itos": tok.itos,
            "stage": stage,
            "step": global_step,
            "val_loss": val_loss,
        }
        torch.save(ckpt, out_dir / "last.pt")
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
        logger(f"   检查点已保存（{'best + last' if improved else 'last'}，best_val {best_val:.4f}）")

        if not max_steps or global_step < max_steps:
            preview(model, tok, stage, device, preview_prompt, logger)

        if stop:
            break

    total_time = time.time() - t_start
    logger("=" * 78)
    logger(f"训练完成 · {global_step} 步 · {human_time(total_time)} · best_val {best_val:.4f}")
    logger(f"检查点目录: {out_dir}")
    logger("=" * 78)

    from .utils import save_json
    save_json({
        "stage": stage,
        "steps": global_step,
        "best_val_loss": round(best_val, 4),
        "history": history,
        "config": cfg,
    }, out_dir / "metrics.json")

    return {"best_val": best_val, "steps": global_step, "out_dir": str(out_dir)}


# ── 推理加载 ──────────────────────────────────────────────────────────────────

def load_model_for_inference(path: str | os.PathLike, device: torch.device):
    """从检查点还原 (模型, 分词器, 原始 ckpt)。词表跟着权重一起存，不会对不上。"""
    ckpt_path = resolve(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到检查点 {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer(ckpt["itos"])
    return model, tok, ckpt

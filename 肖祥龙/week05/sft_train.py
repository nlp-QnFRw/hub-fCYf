"""加载 week04 预训练权重，使用 tagged JSONL 完成回答部分监督的 SFT。"""

import argparse
import hashlib
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from sft_common import (
    IGNORE_INDEX, SCRIPT_DIR, SPECIAL_TOKENS, SFTDataset, collate_batch,
    extend_vocab, generate, initialize_sft_model, load_sft_checkpoint,
    load_tagged, perplexity, save_checkpoint, write_json,
)


def run_epoch(model, loader, device, optimizer=None, grad_clip=1.0):
    """执行训练或验证，返回按有效回答 token 加权的平均损失及困惑度。"""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            count = int(labels.ne(IGNORE_INDEX).sum().item())
            if count == 0:
                raise ValueError("批次中没有有效的回答标签")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            summed_loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if not torch.isfinite(summed_loss).item():
                raise FloatingPointError("损失出现 NaN 或无穷值")
            if training:
                (summed_loss / count).backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            total_loss += summed_loss.item()
            total_tokens += count
    if total_tokens == 0:
        raise ValueError("数据加载器没有有效的回答标签")
    loss = total_loss / total_tokens
    return {"loss": loss, "ppl": perplexity(loss), "tokens": total_tokens}


def file_sha256(path):
    """返回输入文件的 SHA-256，方便核对实际使用的数据和基础模型。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def example_outputs(model, vocab, samples, device):
    """在固定的前三条验证样本上生成，供训练前后定性比较，不参与优化。"""
    return [
        {"instruction": sample.instruction, "reference": sample.output,
         **generate(model, vocab, sample.instruction, device)}
        for sample in samples[:3]
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=Path, default=SCRIPT_DIR.parent / "week04" / "best_model_trans.pt")
    parser.add_argument("--train", type=Path, default=SCRIPT_DIR / "sft_train_tagged.jsonl")
    parser.add_argument("--val", type=Path, default=SCRIPT_DIR / "sft_val_tagged.jsonl")
    parser.add_argument("--save", type=Path, default=SCRIPT_DIR / "best_model_sft.pt")
    parser.add_argument("--log", type=Path, default=SCRIPT_DIR / "sft_training_log.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "seq_len", "lr", "grad_clip", "patience"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} 必须是有限正数")
    if not 0 <= args.dropout < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("dropout 必须在 [0,1) 内，weight_decay 必须为有限非负数")
    inputs = {p.resolve() for p in (args.base_model, args.train, args.val)}
    if args.save.resolve() in inputs or args.log.resolve() in inputs or args.save.resolve() == args.log.resolve():
        parser.error("模型与日志输出路径必须不同，且不能覆盖任何输入文件")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA，请使用 --device cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    print(f"设备: {device}；PyTorch: {torch.__version__}；CPU 线程: {torch.get_num_threads()}", flush=True)
    train_samples, val_samples = load_tagged(args.train), load_tagged(args.val)
    base = torch.load(args.base_model, map_location="cpu", weights_only=True)
    vocab, added_chars = extend_vocab(base["char2idx"], train_samples)
    model, cfg = initialize_sft_model(base, vocab, args.seq_len, args.dropout)
    model.to(device)
    train_data = SFTDataset(train_samples, vocab, args.seq_len)
    val_data = SFTDataset(val_samples, vocab, args.seq_len)
    collate = partial(collate_batch, pad_id=vocab[SPECIAL_TOKENS["pad"]])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    cfg.update(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, save=str(args.save.resolve()))
    config = {name: str(value.resolve()) if isinstance(value, Path) else value for name, value in vars(args).items()}
    report = {
        "config": config, "device": str(device), "torch_version": str(torch.__version__),
        "input_sha256": {name: file_sha256(path) for name, path in (("base_model", args.base_model), ("train", args.train), ("val", args.val))},
        "base_vocab_size": len(base["char2idx"]), "vocab_size": len(vocab), "added_chars": added_chars,
        "special_token_ids": {name: vocab[token] for name, token in SPECIAL_TOKENS.items()},
        "train_stats": train_data.stats(), "val_stats": val_data.stats(), "epochs": [],
    }
    print(f"训练: {args.train.name}，{len(train_data)} 条；验证: {args.val.name}，{len(val_data)} 条", flush=True)
    print(f"词表: {len(base['char2idx'])} -> {len(vocab)}，新增正文字符 {len(added_chars)} 个及特殊 token 5 个", flush=True)
    print(f"最大输入 token 数: 训练 {train_data.max_input_tokens}，验证 {val_data.max_input_tokens}", flush=True)
    print(f"验证集未知字符: 指令 {val_data.unknown_instruction}，回答 {val_data.unknown_output}", flush=True)
    report["baseline"] = run_epoch(model, val_loader, device)
    report["examples_before"] = example_outputs(model, vocab, val_samples, device)
    print(f"微调前验证 Loss: {report['baseline']['loss']:.4f}，PPL: {report['baseline']['ppl']:.2f}", flush=True)
    write_json(args.log, report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train = run_epoch(model, train_loader, device, optimizer, args.grad_clip)
        val = run_epoch(model, val_loader, device)
        improved = val["loss"] < best_loss
        if improved:
            best_loss, best_epoch, stale_epochs = val["loss"], epoch, 0
            save_checkpoint(args.save, model, vocab, cfg, config, {"epoch": epoch, "train": train, "val": val, "baseline": report["baseline"]})
        else:
            stale_epochs += 1
        record = {"epoch": epoch, "train": train, "val": val, "seconds": time.perf_counter() - epoch_started, "saved_best": improved}
        report["epochs"].append(record)
        report.update(best_epoch=best_epoch, best_val_loss=best_loss, best_val_ppl=perplexity(best_loss))
        write_json(args.log, report)
        print(f"Epoch {epoch:02d} | Train Loss {train['loss']:.4f} PPL {train['ppl']:.2f} | Val Loss {val['loss']:.4f} PPL {val['ppl']:.2f} | {record['seconds']:.1f}s{' *' if improved else ''}", flush=True)
        if stale_epochs >= args.patience:
            print(f"连续 {args.patience} 轮验证损失未改善，提前停止。", flush=True)
            break
    # 从磁盘重载最优轮次，验证实际交付的模型，而非最后一轮的内存参数。
    best_model, saved = load_sft_checkpoint(args.save, device)
    report["reloaded_validation"] = run_epoch(best_model, val_loader, device)
    if not math.isclose(report["reloaded_validation"]["loss"], best_loss, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("重载模型后的验证损失与最佳轮次不一致")
    report["examples_after"] = example_outputs(best_model, saved["char2idx"], val_samples, device)
    report["improved_over_baseline"] = best_loss < report["baseline"]["loss"]
    report["total_seconds"] = time.perf_counter() - started
    report["completed"] = True
    write_json(args.log, report)
    print(f"完成：最佳轮次 {best_epoch}，验证 PPL {report['best_val_ppl']:.2f}，总耗时 {report['total_seconds']:.1f}s", flush=True)
    print(f"模型: {args.save.resolve()}\n日志: {args.log.resolve()}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
A 组：判别式路线（中文 BERT vs 改进 BERT × 4层/12层 × Linear/CRF）

在课程 bert/train.py 基础上改三处：
  1. --encoder：bert-base-chinese / chinese-roberta-wwm-ext（全词掩码）/ chinese-macbert-base（纠错式 MLM）
  2. --init pretrained|random：random 只在 --bert_layers 4 时允许——
     专门复现课程坑 #30（随机初始化 4 层 = 把预训练变量混进层数消融），作为对照行；
  3. common/vram_guard.py：8GB 笔记本显存 85% 阈值保护，自动降 batch + 梯度累积，OOM 降档重试。

用法（12 个正式格子 + 1 个对照）：
  python disc/train.py --encoder bert-base-chinese      --decoder crf    # A1
  python disc/train.py --encoder bert-base-chinese      --decoder crf --bert_layers 4   # A2
  python disc/train.py --encoder chinese-roberta-wwm-ext --decoder crf ...               # A3/A4
  python disc/train.py --encoder chinese-macbert-base    --decoder crf ...               # A5/A6
  python disc/train.py --encoder bert-base-chinese --bert_layers 4 --init random ...     # 对照(坑#30)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from crf import CRFClassifier
from eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions
from log_utils import TrainLogger, peak_gpu_mb
from event_data import (IGNORE_INDEX, BioDataset, DATA_DIR, ROOT, build_bio_constraint_mask,
                        collate_bio, build_joint_schema, build_start_constraint_mask,
                        decode_bio_spans, illegal_bio_stats, load_enums, load_extract_split,
                        load_schema, make_pred_row, row_event, spans_to_event)
from model_paths import resolve_model_path
from vram_guard import VramGuard

ENCODERS = {
    "bert": "bert-base-chinese",
    "roberta": "chinese-roberta-wwm-ext",
    "macbert": "chinese-macbert-base",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def span_f1_from_char_tags(rows, char_tags):
    """模型选择用的轻量指标：span 级精确匹配 F1（课程同口径）。"""
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {(a["role"], a["text"]) for a in ge.get("args") or []}
        gold.add(("__trigger__", (ge.get("trigger") or {}).get("text", "")))
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        pred = {(a["role"], a["text"]) for a in (ev or {}).get("args", [])}
        pred.add(("__trigger__", (ev or {}).get("trigger", {}).get("text", "")))
        pred.discard(("__trigger__", ""))
        gold.discard(("__trigger__", ""))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def stage_dataset(dataset, device) -> dict:
    """把整个 split 的 token 特征一次性搬上 GPU（~50MB/千句）。

    为什么不再用 DataLoader + .to(device)：pageable 内存的 H2D 拷贝是**隐式同步**，
    一个 batch 5 个张量就是 5 次流同步。本机驱动状态异常时单次同步会被放大到秒级
    （py-spy 栈直接停在 run_epoch 的 .to() 那行），步速从 0.45s 掉到 5–10s/步，
    而 GPU 全程空闲、CPU 也只占 0.9 核——典型的"看起来像模型慢，其实在等驱动"。
    训练集只有 4969 句，整块常驻显存完全不心疼，换来循环里零 H2D、零同步。
    """
    items = [dataset[i] for i in range(len(dataset))]
    n = len(items)
    max_len = max((int(it["input_ids"].numel()) for it in items), default=8)
    out = {
        "input_ids": torch.zeros(n, max_len, dtype=torch.long),
        "attention_mask": torch.zeros(n, max_len, dtype=torch.long),
        "token_type_ids": torch.zeros(n, max_len, dtype=torch.long),
        "labels": torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long),
        "word_ids": torch.full((n, max_len), -1, dtype=torch.long),
        "row_idx": torch.zeros(n, dtype=torch.long),
    }
    for i, it in enumerate(items):
        L = int(it["input_ids"].numel())
        for k in ("input_ids", "attention_mask", "token_type_ids", "labels", "word_ids"):
            if k in it:
                out[k][i, :L] = it[k]
        out["row_idx"][i] = it["row_idx"]
    return {k: v.to(device) for k, v in out.items()}


class GpuBatchLoader:
    """已驻留显存的批加载器：只做 index_select，不产生任何 H2D 拷贝。

    接口对齐 DataLoader（__len__ / __iter__ / .dataset），所以训练与评测代码不用改。
    """

    def __init__(self, dataset, tensors: dict, batch_size: int, shuffle: bool = False,
                 seed: int = 42):
        self.dataset = dataset
        self.t = tensors
        self.bs = max(1, int(batch_size))
        self.shuffle = shuffle
        self.seed = seed
        self.n = tensors["input_ids"].size(0)

    def __len__(self) -> int:
        return max(1, (self.n + self.bs - 1) // self.bs)

    def __iter__(self):
        dev = self.t["input_ids"].device
        if self.shuffle:
            order = torch.randperm(self.n, device=dev, generator=None)
        else:
            order = torch.arange(self.n, device=dev)
        for i in range(0, self.n, self.bs):
            idx = order[i:i + self.bs]
            yield {k: v.index_select(0, idx) for k, v in self.t.items()}


class Trainee(nn.Module):
    """BERT + (Linear|CRF)，encoder/层数/初始化方式可配（BertEncoder 的本地扩展版）。"""

    def __init__(self, bert_path, num_hidden_layers, num_tags, dropout, use_crf,
                 constraint=None, start_constraint=None, init="pretrained"):
        super().__init__()
        if init == "random":
            # 故意复现课程坑 #30：随机初始化 N 层（预训练知识全丢），仅作对照行
            cfg = BertConfig.from_pretrained(bert_path)
            cfg.num_hidden_layers = num_hidden_layers or cfg.num_hidden_layers
            self.bert = BertModel(cfg)
        else:
            self.bert = BertModel.from_pretrained(bert_path, return_dict=True)
            if num_hidden_layers and num_hidden_layers > 0:
                self.bert.encoder.layer = nn.ModuleList(list(self.bert.encoder.layer)[:num_hidden_layers])
                self.bert.config.num_hidden_layers = num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.head = CRFClassifier(
            encoder=_EncoderAdapter(self.bert), hidden_size=self.hidden_size,
            num_tags=num_tags, dropout=dropout, use_crf=use_crf,
            constraint=constraint, start_constraint=start_constraint)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        return self.head(input_ids, attention_mask, labels, token_type_ids)

    def predict(self, input_ids, attention_mask, token_type_ids=None, **kw):
        return self.head.predict(input_ids, attention_mask, token_type_ids, **kw)


class _EncoderAdapter(nn.Module):
    """把 BertModel 适配成 CRFClassifier 需要的 encoder 接口（输入 ids → hidden）。
    注意必须包装 bert 模块本身而不是外层 Trainee —— 持有 owner 会让 nn.Module 的子模块
    图成环，.to(device) 时无限递归（本项目实测踩过）。"""

    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        return out.last_hidden_state


def evaluate_loss_and_f1(model, loader, id2label, device):
    model.eval()
    # 同步纪律：loss 用 GPU 张量累加，循环结束才 .item()（一次同步代替每 batch 一次）
    total_t, n = None, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            _, loss = model(batch["input_ids"], batch["attention_mask"],
                            batch["labels"], batch.get("token_type_ids"))
            total_t = loss.detach() if total_t is None else total_t + loss.detach()
            n += 1
    avg_loss = float(total_t.item()) / max(n, 1) if total_t is not None else 0.0
    tags, _ = predict_char_tags_local(model, loader, id2label, device)
    f1 = span_f1_from_char_tags(loader.dataset.rows, tags)
    return avg_loss, f1


def predict_char_tags_local(model, loader, id2label, device):
    """子词级预测 → 字符级标签（与课程 models.predict_char_tags 同逻辑）。

    同步纪律：预测标签和 word_ids 先在 GPU 上攒齐（pad 到 batch 内最长），
    循环结束后**一次性** .cpu()——整个评测集一次 DtoH 同步，而不是每 batch 一次。
    本机驱动异常时单次同步可被放大到分钟级，这一处改动把 2548 句 dev 评测的
    同步从 ~40 次压到 1 次。
    """
    model.eval()
    all_tags = []
    pred_chunks, wid_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred_chunks.append(model.predict(batch["input_ids"], batch["attention_mask"],
                                             batch.get("token_type_ids"), return_tensor=True))
            wid_chunks.append(batch["word_ids"])            # (B, L) GPU 张量
    # 动态 padding 下各 batch 序列长不同，先 pad 到全局最长再 cat（PAD 区 word_ids 填 -1，对齐时跳过）
    Lmax = max(p.shape[1] for p in pred_chunks)
    pred_cpu = torch.cat([torch.nn.functional.pad(p, (0, Lmax - p.shape[1]))
                          for p in pred_chunks], dim=0).cpu().tolist()
    Wmax = max(w.shape[1] for w in wid_chunks)
    wid_cpu = torch.cat([torch.nn.functional.pad(w, (0, Wmax - w.shape[1]), value=-1)
                         for w in wid_chunks], dim=0).cpu().tolist()   # 全评测集唯一一次 DtoH
    for b in range(len(pred_cpu)):
        max_wid = max(wid_cpu[b]) if wid_cpu[b] else -1
        n_chars = max_wid + 1 if max_wid >= 0 else 0
        tags = ["O"] * n_chars
        prev_wid = None
        for j, wid in enumerate(wid_cpu[b]):
            if wid < 0 or wid >= n_chars:
                prev_wid = wid
                continue
            if wid != prev_wid and j < len(pred_cpu[b]):
                tags[wid] = id2label.get(pred_cpu[b][j], "O")
            prev_wid = wid
        all_tags.append(tags)
    return all_tags, {}


def build_optimizer(model, args):
    enc_params, head_params = [], []
    for name, p in model.head.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if "encoder." in name else head_params).append(p)
    groups = [{"params": enc_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr * args.head_lr_mult})
    return AdamW(groups, weight_decay=args.weight_decay)


def run_epoch(model, loader, optimizer, scheduler, device, args, epoch, logger=None):
    model.train()
    # 同步纪律（本机驱动状态异常，一次 .item() 可能被放大到秒级~分钟级）：
    # loss 用 GPU 张量累加，只在 log_every 点和 epoch 末尾才 .item()，把每 epoch
    # 百余次同步压到两三次。正常机器上这样也快（省百余次 DtoH）。
    total_t, n = None, 0
    loss_acc = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        # non_blocking：pinned 内存 + 异步拷贝，避免每步 5 次隐式同步（见 make_loaders 注释）
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        _, loss = model(batch["input_ids"], batch["attention_mask"],
                        batch["labels"], batch.get("token_type_ids"))
        (loss / args.grad_accum).backward()
        total_t = loss.detach() if total_t is None else total_t + loss.detach()
        n += 1
        if (step + 1) % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if logger is not None:
            logger.log_step(epoch=epoch, step=step, loss=loss_acc / max(n - 1, 1),
                            lr=scheduler.get_last_lr()[0])
        if (step + 1) % 100 == 0:
            loss_acc = float(total_t.item())
            print(f"  epoch {epoch} step {step + 1}/{len(loader)} loss={loss_acc / n:.4f}", flush=True)
    if len(loader) % args.grad_accum != 0:
        nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return float(total_t.item()) / max(n, 1) if total_t is not None else 0.0


def schema_stats_str(schema):
    roles = {r for v in schema.values() for r in v}
    return f"{len(schema)} 类 / {sum(len(v) for v in schema.values())} 槽位 / {len(roles)} 去重角色"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备：{device}")
    guard = VramGuard(device=device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"显存：可用 {free / 1024**2:.0f} / 总 {total / 1024**2:.0f} MiB，阈值 {guard.budget_mb():.0f} MiB")

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")
    label_scheme = build_joint_schema(schema)
    labels = label_scheme["labels"]
    id2label = label_scheme["id2label"]
    print(f"标签体系：joint，{len(labels)} 个标签（{schema_stats_str(schema)}）")

    model_dir = ENCODERS[args.encoder]
    bert_path = resolve_model_path(model_dir)
    print(f"编码器：{args.encoder} → {bert_path}")
    tokenizer = BertTokenizer.from_pretrained(str(bert_path))

    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]

    rng = random.Random(args.seed)
    idxs = list(range(len(train_rows)))
    rng.shuffle(idxs)
    n_inner = max(1, int(len(train_rows) * args.inner_val))
    inner_idx = set(idxs[:n_inner])
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]
    print(f"训练 {len(fit_rows)} 行 / 内层验证 {len(val_rows)} 行 / 评测 dev {len(dev_rows)} 行")

    # 显存保护：训练前按阈值定 batch，OOM 再降档
    batch_size, grad_accum = guard.adjust(args.batch_size, args.grad_accum)
    args.grad_accum = grad_accum

    # 三个 split 的特征一次常驻显存（合计约 95MB），训练/评测循环里零 H2D（见 stage_dataset）
    datasets = {
        "fit": BioDataset(fit_rows, tokenizer, label_scheme, args.max_len),
        "val": BioDataset(val_rows, tokenizer, label_scheme, args.max_len),
        "dev": BioDataset(dev_rows, tokenizer, label_scheme, args.max_len),
    }
    staged = {k: stage_dataset(ds, device) for k, ds in datasets.items()}

    def make_loaders(bs, eval_bs=None):
        eval_bs = eval_bs or max(bs, 64)   # 评测用更大 batch：FewFC dev 2548 句，监控成本主要在推理
        return (GpuBatchLoader(datasets["fit"], staged["fit"], bs, shuffle=True, seed=args.seed),
                GpuBatchLoader(datasets["val"], staged["val"], eval_bs, shuffle=False),
                GpuBatchLoader(datasets["dev"], staged["dev"], eval_bs, shuffle=False))

    constraint = start_c = None
    if args.hard_constraint:
        constraint = build_bio_constraint_mask(label_scheme)
        start_c = build_start_constraint_mask(label_scheme)
    model = Trainee(str(bert_path), args.bert_layers, len(labels), args.dropout,
                    use_crf=(args.decoder == "crf"), constraint=constraint,
                    start_constraint=start_c, init=args.init).to(device)
    n_params = sum(p.numel() for p in model.head.parameters())
    print(f"模型参数：{n_params / 1e6:.1f}M，解码器={args.decoder}，"
          f"层数={args.bert_layers or 12}，初始化={args.init}")

    optimizer = build_optimizer(model, args)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"

    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=n_params,
                         notes=f"FewFC 判别式（{args.encoder}/{args.bert_layers or 12}L/{args.init}/{args.decoder}）")

    best_f1, best_epoch = -1.0, 0
    train_start = time.perf_counter()
    total_steps = max(1, (len(fit_rows) // batch_size) * args.epochs // max(grad_accum, 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    for epoch in range(1, args.epochs + 1):
        while True:  # OOM 降档重试本 epoch
            try:
                fit_loader, val_loader, dev_loader = make_loaders(batch_size)
                t0 = time.perf_counter()
                train_loss = run_epoch(model, fit_loader, optimizer, scheduler, device,
                                       args, epoch, logger=logger)
                break
            except torch.cuda.OutOfMemoryError:
                batch_size, grad_accum = guard.on_oom(batch_size, grad_accum)
                args.grad_accum = grad_accum
                if batch_size == 1 and guard.history[-1].get("oom_at_batch") == 1:
                    raise
        val_loss, val_f1 = evaluate_loss_and_f1(model, val_loader, id2label, device)
        # dev 监控只在第 1 轮跑（看初始水平）；之后靠内层验证选点，最终 dev 评测在训后一次性做。
        # --dev_monitor 0 可关掉（本机慢窗口下 2548 句监控评测会放大成十几分钟，纯诊断无决策价值）
        if epoch == 1 and args.dev_monitor:
            _, dev_f1_monitor = evaluate_loss_and_f1(model, dev_loader, id2label, device)
        else:
            dev_f1_monitor = float("nan")
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"inner_loss={val_loss:.4f} inner_spanF1={val_f1:.4f} | "
              f"dev_spanF1(监控,仅ep1)={dev_f1_monitor:.4f} | {elapsed:.0f}s", flush=True)
        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                         val_metrics={"inner_loss": val_loss, "inner_span_f1": val_f1,
                                      "dev_span_f1_monitor": dev_f1_monitor},
                         extra={"best_so_far": round(best_f1, 6)})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.head.state_dict(),
                        "label_scheme": {k: label_scheme[k] for k in
                                         ("labels", "label2id", "id2label", "name")},
                        "args": vars(args), "epoch": epoch,
                        "inner_val_span_f1": val_f1}, ckpt_path)
            print(f"  ★ 内层验证最优 → 保存 {ckpt_path.name}", flush=True)

    train_s = time.perf_counter() - train_start

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.head.load_state_dict(ckpt["state_dict"])
    print(f"加载最优 checkpoint（epoch {ckpt['epoch']}，内层验证 spanF1={best_f1:.4f}）")

    _, _, dev_loader = make_loaders(batch_size)
    with Timer() as timer:
        dev_tags, _ = predict_char_tags_local(model, dev_loader, id2label, device)
    dev_illegal = illegal_bio_stats(dev_tags)

    preds = []
    n_empty = 0
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        if ev is None:
            n_empty += 1
        # 多事件句：一个 id 可能对应多个 gold 行——预测按句聚合（make_pred_row 按行会重复 id，
        # 统一交给 evaluate 的 merge_rows_by_sample 聚合，因此这里每句只写一行）
        preds.append({"uid": str(row["id"]), "id": row["id"], "events": [ev] if ev else []})
    print(f"dev 解码：{len(preds)} 句，没抽到触发词 {n_empty} 句")

    lat = timer.summary(len(dev_rows))
    meta = {
        "display_name": f"{args.encoder}-{args.bert_layers or 12}L({args.init})+{args.decoder.upper()}",
        "group": "A",
        "encoder": args.encoder, "bert_layers": args.bert_layers or 12,
        "init": args.init, "decoder": args.decoder,
        "train_s": round(train_s, 1),
        "params_m": round(n_params / 1e6, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "max_len": args.max_len, "epochs": args.epochs, "lr": args.lr,
        "batch_size": batch_size, "grad_accum": grad_accum,
        "best_epoch": best_epoch, "inner_val_span_f1": round(best_f1, 4),
        "num_labels": len(labels), "illegal_dev": dev_illegal,
        "n_truncated": dev_loader.dataset.n_truncated,
        "hard_constraint": bool(args.hard_constraint), "seed": args.seed,
        "vram_guard": guard.meta(),
        "selection_note": "按 train 切出的 10% 内层验证集选 epoch；dev 未参与训练与选点",
    }
    res = evaluate_predictions(args.run_name, preds, meta=meta,
                               gold_path=data_dir / "dev_extract.jsonl")
    logger.finalize(
        total_s=train_s, best_epoch=best_epoch,
        best_inner_val_span_f1=round(best_f1, 6),
        dev_metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        inference_latency_ms=lat["ms_per_sentence"],
        peak_gpu_mb=peak_gpu_mb(),
        illegal_dev=dev_illegal,
        vram_guard=guard.meta(),
    )


def parse_args():
    ap = argparse.ArgumentParser(description="FewFC 判别式：编码器×层数×解码器 网格")
    ap.add_argument("--encoder", choices=list(ENCODERS.keys()), default="bert")
    ap.add_argument("--decoder", choices=["linear", "crf"], default="crf")
    ap.add_argument("--bert_layers", type=int, default=0, help="0=完整 12 层；4=截断前 4 层（保留预训练权重）")
    ap.add_argument("--init", choices=["pretrained", "random"], default="pretrained",
                    help="random 仅用于对照：随机初始化 --bert_layers 层（复现课程坑 #30）")
    ap.add_argument("--hard_constraint", action="store_true")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inner_val", type=float, default=0.1)
    ap.add_argument("--train_limit", type=int, default=0, help="只取前 N 行训练（低资源消融用）")
    ap.add_argument("--device", default="")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--dev_monitor", type=int, default=1,
                    help="epoch1 是否跑 dev 全量监控评测（0=跳过，最终 dev 评测不受影响）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--data_dir", default="")
    args = ap.parse_args()
    if args.init == "random" and not args.bert_layers:
        ap.error("--init random 必须配合 --bert_layers N（随机初始化完整 12 层没有意义）")
    if not args.run_name:
        tag = f"{args.encoder}{args.bert_layers or 12}L"
        if args.init == "random":
            tag += "rand"
        args.run_name = f"{tag}_{args.decoder}_{args.max_len}"
    return args


if __name__ == "__main__":
    main()

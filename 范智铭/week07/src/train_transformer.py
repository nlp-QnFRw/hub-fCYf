"""微调 BERT / 中文 RoBERTa（同一个脚本，靠 --model 切换）。

    python -m src.train_transformer --dataset resume --model bert --epochs 6 --max_len 512
    python -m src.train_transformer --dataset cluener --model roberta --epochs 3 --max_len 128

产物（outputs/<dataset>/<bert|roberta>/）：
    best/（HF 格式） / metrics.json / pred.jsonl / train_log.json / train.log

关键实现点（都是踩过的坑）：
- 子词对齐用 offset_mapping + word_ids，同一 word 首子词 B-、后续 I-；
- 动态 padding（batch 内最长），不按全局 max_len 浪费显存；
- 推理时从 offset 还原字符区间，而不是拿 token 文本拼；
- 按 dev 实体级 F1 存 best，不看 loss。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .common import OUTPUTS, ROOT, Timer, get_logger, human_time, set_seed, write_json, write_jsonl
from .data import (HFDataset, PadCollator, bio_to_entities, build_label_maps, load_labels, load_splits)
from .metrics import evaluate, format_table

MODEL_DIRS = {"bert": ROOT / "models" / "bert", "roberta": ROOT / "models" / "roberta"}
MODEL_REPOS = {"bert": "google-bert/bert-base-chinese", "roberta": "hfl/chinese-roberta-wwm-ext"}


def load_tokenizer_model(alias: str, num_labels: int, device):
    path = MODEL_DIRS[alias]
    src = str(path) if path.exists() and (path / "config.json").exists() else MODEL_REPOS[alias]
    tok = AutoTokenizer.from_pretrained(src, local_files_only=Path(src).exists(), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        src, num_labels=num_labels, local_files_only=Path(src).exists())
    return tok, model.to(device)


def decode_offsets(pred_ids: list[int], offsets: list[tuple[int, int]], id2tag: list[str]) -> list[dict]:
    """token 预测 → 字符区间。

    跳过特殊 token（offset 为 (0,0)）；同一个实体跨多个子词时首尾合并。
    """
    ents: list[dict] = []
    cur_lab, cur_start, cur_end = None, None, None
    for tid, (s, e) in zip(pred_ids, offsets):
        if s == 0 and e == 0:
            continue
        tag = id2tag[tid]
        if tag == "O":
            if cur_lab is not None:
                ents.append({"start": cur_start, "end": cur_end, "label": cur_lab})
                cur_lab = None
            continue
        prefix, _, lab = tag.partition("-")
        if prefix == "B" or cur_lab != lab:
            if cur_lab is not None:
                ents.append({"start": cur_start, "end": cur_end, "label": cur_lab})
            cur_lab, cur_start, cur_end = lab, s, e
        else:
            cur_end = e
    if cur_lab is not None:
        ents.append({"start": cur_start, "end": cur_end, "label": cur_lab})
    return ents


@torch.no_grad()
def predict(model, tok, rows, id2tag, max_len, device, batch_size=64) -> list[dict]:
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len,
                  return_offsets_mapping=True, padding=True, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        preds = logits.argmax(-1).cpu().tolist()
        offs = enc["offset_mapping"].tolist()
        for row, ids, off in zip(chunk, preds, offs):
            out.append({"id": row["id"], "text": row["text"],
                        "entities": decode_offsets(ids, off, id2tag)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["resume", "cluener", "msra"])
    ap.add_argument("--model", required=True, choices=["bert", "roberta"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    out_dir = OUTPUTS / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"{args.model}.{args.dataset}", out_dir / "train.log")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = args.fp16 and device.type == "cuda"

    splits = load_splits(args.dataset)
    labels = load_labels(args.dataset)
    tag2id, id2tag = build_label_maps(labels)
    tok, model = load_tokenizer_model(args.model, len(id2tag), device)
    model.config.id2label = dict(enumerate(id2tag))
    model.config.label2id = tag2id
    n_params = sum(p.numel() for p in model.parameters())
    log.info("device=%s model=%s params=%.1fM fp16=%s", device, args.model, n_params / 1e6, fp16)

    collator = PadCollator(tok.pad_token_id or 0)
    ds = {s: HFDataset(splits[s], tok, tag2id, args.max_len) for s in splits}
    dl = {
        "train": DataLoader(ds["train"], batch_size=args.batch, shuffle=True, collate_fn=collator),
        "dev": DataLoader(ds["dev"], batch_size=args.batch * 2, shuffle=False, collate_fn=collator),
    }
    total_steps = len(dl["train"]) * args.epochs
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    best_f1, best_epoch, history = -1.0, -1, []
    with Timer() as tt:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, total, nb = time.perf_counter(), 0.0, 0
            for batch in dl["train"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=fp16):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += float(loss.item())
                nb += 1
            dev_pred = predict(model, tok, splits["dev"], id2tag, args.max_len, device)
            dev_f1 = evaluate(splits["dev"], dev_pred, labels)["overall"]["f1"]
            rec = {"epoch": epoch, "train_loss": total / max(1, nb), "dev_f1": dev_f1,
                   "sec": round(time.perf_counter() - t0, 1)}
            history.append(rec)
            log.info("epoch %d | loss %.4f | dev F1 %.4f | %.0fs", epoch, rec["train_loss"], dev_f1, rec["sec"])
            if dev_f1 > best_f1:
                best_f1, best_epoch = dev_f1, epoch
                model.save_pretrained(out_dir / "best")
                tok.save_pretrained(out_dir / "best")
    log.info("训练结束 best epoch=%d dev F1=%.4f 用时 %s", best_epoch, best_f1, human_time(tt.seconds))

    # 用 best 权重推理测试集
    model = AutoModelForTokenClassification.from_pretrained(out_dir / "best", local_files_only=True).to(device)
    t0 = time.perf_counter()
    test_pred = predict(model, tok, splits["test"], id2tag, args.max_len, device)
    infer_sec = time.perf_counter() - t0
    write_jsonl(out_dir / "pred.jsonl", test_pred)

    result = evaluate(splits["test"], test_pred, labels)
    hard = [r for r in splits["test"] if r.get("meta", {}).get("hard")]
    if hard:
        result["hard_subset"] = evaluate(splits["test"], test_pred, labels, subset_ids={r["id"] for r in hard})

    metrics = {
        "dataset": args.dataset, "method": args.model, "test": result,
        "dev_f1_best": best_f1, "best_epoch": best_epoch, "n_params": n_params,
        "train_seconds": round(tt.seconds, 1), "infer_seconds": round(infer_sec, 2),
        "ms_per_sentence": round(1000 * infer_sec / max(1, len(splits["test"])), 2),
        "epochs_run": len(history), "args": vars(args),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "train_log.json", history)
    log.info("测试集：\n%s", format_table(result))
    if "hard_subset" in result:
        log.info("hard 子集 F1 %.4f", result["hard_subset"]["overall"]["f1"])
    log.info("ms/句 %.2f", metrics["ms_per_sentence"])


if __name__ == "__main__":
    main()

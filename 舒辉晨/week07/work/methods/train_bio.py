"""判别式序列标注：换两个预训练 backbone（rbt3 38M / MacBERT 102M），同一套 BIO 解码。

复用 week7/common 的 BioDataset / CRFClassifier / predict_char_tags / evaluate_events，
只把训练编排和产物路径搬进 work/，保证「只是换了模型」而不是换了评测口径。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (CKPT_DIR, DATA_ROOT, LOG_DIR, MODEL_DIRS, PRETRAIN_ROOTS,  # noqa: E402
                    SEED, WEEK7_DATA)
from common.bridge import (  # noqa: E402
    BioDataset, Timer, batch_to_device, build_joint_schema, build_model,
    collate_bio, decode_bio_spans, load_enums, load_jsonl, load_schema,
    make_pred_row, predict_char_tags, resolve_model_path, spans_to_event,
)
from common.eval_local import run_eval  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

IGNORE = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def span_f1(rows, tags) -> float:
    """内层验证用的轻量 span F1（与 week7 训练脚本同口径）。"""
    tp = fp = fn = 0
    for row, tagseq in zip(rows, tags):
        ev = row.get("event")
        gold = set()
        if ev:
            gold = {(a["role"], a["text"]) for a in (ev.get("args") or [])}
            t = (ev.get("trigger") or {}).get("text", "")
            if t:
                gold.add(("__trigger__", t))
        predicted = spans_to_event(decode_bio_spans(tagseq), row["text"]) or {}
        pred = {(a["role"], a["text"]) for a in (predicted.get("args") or [])}
        t = (predicted.get("trigger") or {}).get("text", "")
        if t:
            pred.add(("__trigger__", t))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


@torch.no_grad()
def eval_loss_f1(model, loader, id2label, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        _, loss = model(batch["input_ids"], batch["attention_mask"],
                        batch["labels"], batch.get("token_type_ids"))
        losses.append(float(loss.item()))
    tags, _ = predict_char_tags(model, loader, id2label, device)
    return sum(losses) / max(len(losses), 1), span_f1(loader.dataset.rows, tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rbt3", "macbert"], required=True)
    ap.add_argument("--variant", choices=["sent", "heldout", "doc"], default="sent")
    ap.add_argument("--decoder", choices=["crf", "linear"], default="crf")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--inner_val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    ap.add_argument("--predict_only", action="store_true",
                    help="跳过训练，直接拿 --src_tag 的 checkpoint 预测（用于换数据粒度）")
    ap.add_argument("--src_tag", default="", help="checkpoint 来源方法名")
    args = ap.parse_args()
    set_seed(args.seed)

    ddir = WEEK7_DATA if args.variant == "sent" else DATA_ROOT / args.variant
    train_path = ddir / "train_extract.jsonl"
    if not train_path.exists():
        train_path = WEEK7_DATA / "train_extract.jsonl"   # 篇章变体只改 dev，训练沿用原句级数据
    train_rows = load_jsonl(str(train_path))
    dev_rows = load_jsonl(str(ddir / "dev_extract.jsonl"))
    schema = load_schema(str(ddir / "event_schema.json"))
    enums = load_enums(str(ddir / "event_schema.json"))
    labels = build_joint_schema(schema)
    id2label = labels["id2label"]

    rng = random.Random(args.seed)
    idx = list(range(len(train_rows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(train_rows) * args.inner_val))
    val_idx = set(idx[:n_val])
    fit = [r for i, r in enumerate(train_rows) if i not in val_idx]
    val = [r for i, r in enumerate(train_rows) if i in val_idx]
    print(f"模型={args.model} 变体={args.variant} | 训练 {len(fit)} / 内层验证 {len(val)} "
          f"/ dev {len(dev_rows)}", flush=True)

    model_path = resolve_model_path(MODEL_DIRS[args.model], extra_roots=PRETRAIN_ROOTS)
    print(f"权重：{model_path}", flush=True)
    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    ds = lambda rows: BioDataset(rows, tokenizer, labels, args.max_len)
    loader = lambda rows, shuffle: DataLoader(
        ds(rows), batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_bio)
    fit_loader, val_loader, dev_loader = loader(fit, True), loader(val, False), loader(dev_rows, False)

    cfg = argparse.Namespace(bert_path=str(model_path), bert_layers=0,
                             decoder=args.decoder, dropout=0.1, hard_constraint=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, labels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数 {n_params / 1e6:.1f}M | device={device}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = args.tag or f"{args.model}_{args.decoder}_{args.variant}"
    best_f1, best_epoch, log, train_s = -1.0, 0, [], 0.0

    if args.predict_only:
        src = args.src_tag or f"{args.model}_{args.decoder}_sent"
        ckpt_path = CKPT_DIR / f"{src}.pt"
        print(f"[predict_only] 直接加载 {ckpt_path.name}（不训练）", flush=True)
    else:
        enc, head = [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (enc if pname.startswith("encoder.") else head).append(p)
        optimizer = AdamW([{"params": enc, "lr": args.lr},
                           {"params": head, "lr": args.lr * args.head_lr_mult}], weight_decay=0.01)
        steps = len(fit_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
        print(f"steps={steps}", flush=True)
        ckpt_path = CKPT_DIR / f"{name}.pt"
        t0 = time.perf_counter()
        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for batch in fit_loader:
                batch = batch_to_device(batch, device)
                _, loss = model(batch["input_ids"], batch["attention_mask"],
                                batch["labels"], batch.get("token_type_ids"))
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total += float(loss.item())
            _, val_f1 = eval_loss_f1(model, val_loader, id2label, device)
            log.append({"epoch": ep, "loss": round(total / len(fit_loader), 5),
                        "inner_span_f1": round(val_f1, 5)})
            print(f"  epoch {ep}/{args.epochs} loss={total / len(fit_loader):.4f} "
                  f"inner_span_f1={val_f1:.4f}", flush=True)
            if val_f1 > best_f1:
                best_f1, best_epoch = val_f1, ep
                torch.save({"state_dict": model.state_dict(), "epoch": ep}, ckpt_path)
        train_s = time.perf_counter() - t0
        with open(LOG_DIR / f"train_{name}.json", "w", encoding="utf-8") as f:
            json.dump({"epochs": log, "best_epoch": best_epoch, "best_inner_span_f1": best_f1,
                       "train_s": round(train_s, 1), "args": vars(args),
                       "model_path": str(model_path)}, f, ensure_ascii=False, indent=2)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if args.predict_only:
        best_epoch = int(ckpt.get("epoch", 0))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with Timer() as timer:
        tags, _ = predict_char_tags(model, dev_loader, id2label, device)
    lat = timer.summary(len(dev_rows))
    preds = []
    for row, tagseq in zip(dev_rows, tags):
        ev = spans_to_event(decode_bio_spans(tagseq), row["text"], enums=enums)
        preds.append(make_pred_row(row, [ev] if ev else []))
    meta = {
        "model": args.model, "model_path": str(model_path), "backbone_params_m": 38.5 if args.model == "rbt3" else 102.3,
        "total_params_m": round(n_params / 1e6, 1), "decoder": args.decoder,
        "train_s": round(train_s, 1), "epochs": args.epochs, "best_epoch": best_epoch,
        "best_inner_span_f1": round(best_f1, 4) if best_f1 >= 0 else None,
        "src_tag": (args.src_tag or f"{args.model}_{args.decoder}_sent") if args.predict_only else None,
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"], "max_len": args.max_len,
        "batch_size": args.batch_size, "seed": args.seed,
        "selection_note": "按 train 内切 10% 选 epoch；dev 未参与训练与选点",
    }
    res = run_eval(name, preds, ddir / "dev_extract.jsonl", meta=meta,
                   doc_index=(ddir / "doc_index.json") if args.variant == "doc" else None)
    print(f"{name}: argument F1={res['argument']['f1']:.4f} "
          f"event F1={res['event_exact']['f1']:.4f} | {train_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()

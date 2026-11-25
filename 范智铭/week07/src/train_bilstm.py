"""训练 BiLSTM(-CRF)，并在测试集上产出统一格式的预测。

一条命令跑一个数据集：
    python -m src.train_bilstm --dataset resume --epochs 20
    python -m src.train_bilstm --dataset cluener --epochs 10
    python -m src.train_bilstm --dataset msra --epochs 10

产物（outputs/<dataset>/bilstm/）：
    best.pt / vocab.json / labels.json / metrics.json / pred.jsonl / train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .bilstm_crf import BiLSTMCRF
from .common import OUTPUTS, Timer, human_time, get_logger, set_seed, write_json, write_jsonl
from .data import (CharDataset, bio_to_entities, build_label_maps, build_vocab, collate_chars,
                   load_labels, load_splits)
from .metrics import evaluate, format_table


def predict(model, loader, id2tag, rows, device) -> list[dict]:
    model.eval()
    out, k = [], 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            paths = model.decode(ids, mask)
            for path in paths:
                tags = [id2tag[i] for i in path]
                out.append({"id": rows[k]["id"], "text": rows[k]["text"],
                            "entities": bio_to_entities(tags)})
                k += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["resume", "cluener", "msra"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no_crf", action="store_true", help="消融：去掉 CRF 只留 BiLSTM")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    name = "bilstm" if not args.no_crf else "bilstm_nocrf"
    out_dir = Path(args.out) if args.out else OUTPUTS / args.dataset / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"bilstm.{args.dataset}", out_dir / "train.log")
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    log.info("device=%s dataset=%s", device, args.dataset)

    splits = load_splits(args.dataset)
    labels = load_labels(args.dataset)
    tag2id, id2tag = build_label_maps(labels)
    vocab = build_vocab(splits["train"])
    write_json(out_dir / "vocab.json", vocab)
    write_json(out_dir / "labels.json", {"labels": labels, "tags": id2tag})

    ds = {s: CharDataset(splits[s], vocab, tag2id, args.max_len) for s in splits}
    dl = {
        "train": DataLoader(ds["train"], batch_size=args.batch, shuffle=True, collate_fn=collate_chars),
        "dev": DataLoader(ds["dev"], batch_size=args.batch, shuffle=False, collate_fn=collate_chars),
        "test": DataLoader(ds["test"], batch_size=args.batch, shuffle=False, collate_fn=collate_chars),
    }

    model = BiLSTMCRF(len(vocab), len(id2tag), emb_dim=args.emb, hidden=args.hidden,
                      dropout=args.dropout, use_crf=not args.no_crf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log.info("参数量 %.2f M | train=%d dev=%d test=%d", model.num_params() / 1e6,
             len(ds["train"]), len(ds["dev"]), len(ds["test"]))

    best_f1, best_epoch, bad = -1.0, -1, 0
    history = []
    with Timer() as train_timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, total = time.perf_counter(), 0.0
            for batch in dl["train"]:
                opt.zero_grad()
                loss = model.loss(batch["input_ids"].to(device),
                                  batch["attention_mask"].to(device),
                                  batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += float(loss.item()) * len(batch["labels"])
            train_sec = time.perf_counter() - t0

            dev_pred = predict(model, dl["dev"], id2tag, splits["dev"], device)
            dev_f1 = evaluate(splits["dev"], dev_pred, labels)["overall"]["f1"]
            rec = {"epoch": epoch, "train_loss": total / len(ds["train"]), "dev_f1": dev_f1,
                   "sec": round(train_sec, 1)}
            history.append(rec)
            log.info("epoch %2d | loss %.4f | dev F1 %.4f | %.1fs", epoch, rec["train_loss"], dev_f1, train_sec)
            if dev_f1 > best_f1:
                best_f1, best_epoch, bad = dev_f1, epoch, 0
                torch.save(model.state_dict(), out_dir / "best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    log.info("dev F1 连续 %d 轮没提升，早停", bad)
                    break
    log.info("训练结束：best epoch=%d dev F1=%.4f 用时 %s", best_epoch, best_f1, human_time(train_timer.seconds))

    # ---- 用 best 权重跑测试集
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    t0 = time.perf_counter()
    test_pred = predict(model, dl["test"], id2tag, splits["test"], device)
    infer_sec = time.perf_counter() - t0
    write_jsonl(out_dir / "pred.jsonl", test_pred)

    result = evaluate(splits["test"], test_pred, labels)
    # resume 的 hard 子集单独统计（meta.hard）
    hard_rows = [r for r in splits["test"] if r.get("meta", {}).get("hard")]
    if hard_rows:
        ids = {r["id"] for r in hard_rows}
        result["hard_subset"] = evaluate(splits["test"], test_pred, labels, subset_ids=ids)

    metrics = {
        "dataset": args.dataset,
        "method": name,
        "test": result,
        "dev_f1_best": best_f1,
        "best_epoch": best_epoch,
        "n_params": model.num_params(),
        "train_seconds": round(train_timer.seconds, 1),
        "infer_seconds": round(infer_sec, 2),
        "ms_per_sentence": round(1000 * infer_sec / max(1, len(splits["test"])), 2),
        "epochs_run": len(history),
        "args": vars(args),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "train_log.json", history)

    log.info("测试集：\n%s", format_table(result))
    if "hard_subset" in result:
        log.info("hard 子集 F1 %.4f", result["hard_subset"]["overall"]["f1"])
    log.info("ms/句 %.2f | 参数量 %.2fM", metrics["ms_per_sentence"], metrics["n_params"] / 1e6)


if __name__ == "__main__":
    main()

"""
文档级事件抽取：切句 → 逐句推理 → 文档级 offset 还原 → 事件汇总。

输入 JSONL（每行一篇文档，events 字段即使存在也会被忽略）：
  {"id": "...", "title": "...", "text": "..."}

输出 JSONL：
  {"id", "title", "n_sentences", "events", "needs_review"}
  每个 event 带 type / trigger / args / confidence / sent_ids；
  trigger 和 args 的 start/end 都是相对原始 text 的左闭右开字符偏移。

confidence 是 CRF 路径（或 Linear token 概率）的平均置信度，未做概率校准，
不能直接解释为“抽取正确的概率”。低置信度事件仍保留，只把文档标成 needs_review。

使用方式：
  python bert/predict_doc.py --ckpt outputs/checkpoints/bert_crf_256.pt
  python bert/predict_doc.py --ckpt ... --input docs.jsonl --output predictions.jsonl
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from eval_utils import CKPT_DIR, PRED_DIR
from event_data import (BioDataset, DATA_DIR, collate_bio, decode_bio_spans,
                        load_enums, load_jsonl, save_jsonl, spans_to_event)
from model_paths import BERT_MODEL_DIR, resolve_model_path
from models import build_model, predict_char_tags

SENTENCE_END = frozenset("。！？!?；;\r\n")


def split_sentences(text: str) -> list[dict]:
    """按句末标点/换行切句，保留标点，并记录每句在原文中的字符起点。"""
    sentences = []
    start = 0
    for i, char in enumerate(text):
        if char not in SENTENCE_END:
            continue
        piece = text[start:i + 1]
        if piece.strip():
            sentences.append({"sent_id": len(sentences), "text": piece, "start": start})
        start = i + 1
    if start < len(text) and text[start:].strip():
        sentences.append({"sent_id": len(sentences), "text": text[start:], "start": start})
    return sentences


def format_event(event: dict, doc_offset: int, sent_id: int,
                 confidence: float, review_threshold: float) -> tuple[dict, bool]:
    """句内事件 → 文档级事件；返回事件和该事件是否需要人工复核。"""
    args = []
    enum_invalid = False
    for arg in event["args"]:
        enum_invalid = enum_invalid or not arg.get("enum_ok", True)
        args.append({
            "role": arg["role"],
            "text": arg["text"],
            "start": arg["start"] + doc_offset,
            "end": arg["end"] + doc_offset,
            "source": "model",
        })
    formatted = {
        "type": event["type"],
        "trigger": {
            "text": event["trigger"]["text"],
            "start": event["trigger"]["start"] + doc_offset,
            "end": event["trigger"]["end"] + doc_offset,
        },
        "args": args,
        "confidence": round(float(confidence), 4),
        "sent_ids": [sent_id],
    }
    return formatted, confidence < review_threshold or enum_invalid


def main():
    ap = argparse.ArgumentParser(description="文档级事件抽取（输出文档级 offset 与置信度）")
    ap.add_argument("--ckpt", default="", help="训练 checkpoint；默认 outputs/checkpoints/bert_crf_256.pt")
    ap.add_argument("--input", default="", help="文档 JSONL；默认 data/processed/dev_docs.jsonl")
    ap.add_argument("--output", default="", help="输出 JSONL；默认 outputs/disc/pred_docs_<ckpt>.jsonl")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--review_threshold", type=float, default=0.7,
                    help="低于此置信度的事件保留，但文档 needs_review=true")
    ap.add_argument("--device", default="", help="cuda / cpu；留空自动选择")
    args = ap.parse_args()
    if not 0.0 <= args.review_threshold <= 1.0:
        ap.error("--review_threshold 必须在 0~1 之间")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / "bert_crf_256.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}；请先运行 bert/train.py 训练")
    input_path = Path(args.input) if args.input else DATA_DIR / "dev_docs.jsonl"
    output_path = (Path(args.output) if args.output
                   else PRED_DIR / f"pred_docs_{ckpt_path.stem}.jsonl")

    docs = load_jsonl(input_path)
    for i, doc in enumerate(docs):
        if not isinstance(doc.get("text"), str):
            raise ValueError(f"{input_path} 第 {i + 1} 行缺少字符串字段 text")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_scheme = ckpt["label_scheme"]
    id2label = ckpt.get("id2label") or label_scheme["id2label"]
    train_args = ckpt["args"]
    model_args = argparse.Namespace(**train_args)
    model_args.bert_path = train_args.get("bert_path") or str(resolve_model_path(BERT_MODEL_DIR))
    model_args.bert_layers = train_args.get("bert_layers", 0)
    model_args.decoder = train_args.get("decoder", "crf")
    model_args.hard_constraint = train_args.get("hard_constraint", False)
    model_args.dropout = train_args.get("dropout", 0.1)

    model = build_model(model_args, label_scheme).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # Fast 分词器才有 word_ids()，用于把子词预测还原到字符位置。
    tokenizer = BertTokenizerFast.from_pretrained(model_args.bert_path)
    schema_path = input_path.parent / "event_schema.json"
    if not schema_path.exists():
        schema_path = DATA_DIR / "event_schema.json"
    enums = load_enums(schema_path) if schema_path.exists() else {}

    outputs = []
    flat_rows, flat_meta = [], []
    for doc_idx, doc in enumerate(docs):
        sentences = split_sentences(doc["text"])
        outputs.append({
            "id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "n_sentences": len(sentences),
            "events": [],
            "needs_review": False,
        })
        for sent in sentences:
            flat_rows.append({"text": sent["text"]})
            flat_meta.append((doc_idx, sent["sent_id"], sent["start"]))

    n_truncated = 0
    if flat_rows:
        ds = BioDataset(flat_rows, tokenizer, label_scheme,
                        train_args.get("max_len", 256))
        n_truncated = ds.n_truncated
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)
        tags_list, info = predict_char_tags(model, loader, id2label, device,
                                            return_scores=True)
        scores = info.get("scores", [])
        if not (len(flat_meta) == len(tags_list) == len(scores)):
            raise RuntimeError("句子、标签和置信度数量不一致，无法安全还原文档 offset")

        for (doc_idx, sent_id, doc_offset), row, tags, score in zip(
                flat_meta, flat_rows, tags_list, scores):
            event = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
            if event is None:
                continue
            formatted, needs_review = format_event(
                event, doc_offset, sent_id, score, args.review_threshold)
            outputs[doc_idx]["events"].append(formatted)
            outputs[doc_idx]["needs_review"] = (
                outputs[doc_idx]["needs_review"] or needs_review)

    save_jsonl(outputs, output_path)
    n_events = sum(len(doc["events"]) for doc in outputs)
    n_review = sum(doc["needs_review"] for doc in outputs)
    print(f"文档 {len(docs)} 篇 / 句子 {len(flat_rows)} 句 / 事件 {n_events} 个")
    print(f"需人工复核 {n_review} 篇（阈值 {args.review_threshold:.2f}）；截断 {n_truncated} 句")
    print(f"预测已写入 → {output_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
单句探针：判断某个已训练模型对「输入一句话」能不能抽出结构化事件。

它不负责判断“抽得对不对”（那需要金标），而是先回答三个更基础的问题：
  1. 有没有抽到事件（还是空输出）？
  2. 事件类型、触发词 span、论元 span 是否**结构合法**（能在原文还原、角色属于该类型）？
  3. 模型给出的置信度/诊断信息是什么？

用法：
  cd week7/src
  python eval/probe_sentence.py --text "漳州发展中标马洲污水处理厂PPP项目，中标金额3.2亿元" --methods rule,bilstm
  python eval/probe_sentence.py --text "..." --methods bert \
      --ckpt /path/to/bert_crf_256.pt --bert_path /path/to/bert-base-chinese

  # 批量看 dev 上 300 句的“空输出率 / 结构合法率 / 金标 F1”，用：
  python eval/score_subset.py

产物：只在终端打印；加 --json 输出机器可读 JSON。

判读约定（很重要）：
  - `ok`      = 结构合法：类型在 schema、触发词/论元 span 都能在原文还原、角色属于该类型。
                它**不保证内容正确**，只保证“能安全地交给下游/人工复核”。
  - `invalid` = 抽到了事件，但结构不合法（典型：角色串类型、span 对不上原文），不能直接入库。
  - `empty`   = 没有输出事件；可能是句子本来无事件，也可能是触发词没命中/模型漏抽。
  - `error`   = 运行/加载失败（路径、checkpoint、依赖），与模型能力无关。

为什么不在这里直接算 F1：
  单句的“对/错”是二值的，一句抽对不代表模型能力；F1 必须在 dev 300 句上按统一口径算。
  本脚本只负责“能不能出结构”，批量指标请用 `eval/score_subset.py` 或训练结束时的 dev 评测。
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]      # week7/
SRC = Path(__file__).resolve().parents[1]       # week7/src/
sys.path.insert(0, str(SRC / "common"))

from event_data import (BioDataset, collate_bio, decode_bio_spans, load_enums,
                        load_extract_split, load_schema, spans_to_event)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────── 校验：结构是否合法 ───────────────────────────

def validate(text: str, event: dict | None, schema: dict) -> tuple[list[dict], str]:
    """对单条预测做“结构合法性”校验，返回 (逐项检查, 总结论)。

    检查项与含义：
      1. 事件类型在 schema 内：类型名必须是 13 类之一，否则下游无法路由；
      2. 触发词 span 可在原文还原：要求 `text[start:end] == trigger.text`，
         这是最容易被忽略的坑——模型可能返回原文没有的词（幻觉触发词）；
      3. 论元角色属于该事件类型：例如“股份回购”不能出现“质押物占总股比”；
      4. 论元 span 可在原文还原：同上，防止模型改写/拼接原文；
      5. 至少一个论元：本数据集 dev 的事件最少也有 1 个论元，空论元通常意味着解码失败。

    注意：**同一角色多值是合法的**（如两个「中标公司」），所以只提示、不算错；
    event_exact 要求“集合完全一致”，多值会让 exact 更难，但不是结构错误。
    """
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if not event:
        add("抽到事件", False, "模型没有输出任何事件")
        return checks, "empty"

    etype = event.get("type")
    add("事件类型在 schema 内", etype in schema, str(etype))
    trig = event.get("trigger") or {}
    ts, te, tt = trig.get("start"), trig.get("end"), trig.get("text")
    trig_ok = isinstance(ts, int) and isinstance(te, int) and text[ts:te] == tt
    add("触发词 span 可在原文还原", trig_ok, f"{tt} [{ts}:{te}]")

    args = event.get("args") or []
    roles = schema.get(etype, [])
    bad_role = [a.get("role") for a in args if a.get("role") not in roles]
    bad_span = [a.get("role") for a in args
                if not (isinstance(a.get("start"), int) and isinstance(a.get("end"), int)
                        and text[a["start"]:a["end"]] == a.get("text"))]
    add("论元角色属于该事件类型", not bad_role, f"非法角色：{bad_role}" if bad_role else f"{len(args)} 个论元")
    add("论元 span 可在原文还原", not bad_span, f"非法 span：{bad_span}" if bad_span else "全部可还原")
    add("至少有一个论元", len(args) > 0, f"论元数 {len(args)}")

    # 重复角色是合法的（同一角色多值），只提示不算错
    role_count = {}
    for a in args:
        role_count[a.get("role")] = role_count.get(a.get("role"), 0) + 1
    multi = {k: v for k, v in role_count.items() if v > 1}
    if multi:
        checks.append({"check": "同角色多值", "ok": True, "detail": str(multi)})

    struct_ok = all(c["ok"] for c in checks)
    return checks, "ok" if struct_ok else "invalid"


# ─────────────────────────── 各方法单句推理 ───────────────────────────

def infer_rule(text: str, data_dir: Path):
    """规则基线：从 train 统计触发词/公司词典，再对单句做最长优先匹配。

    为什么每次都重建词典：
      规则没有 checkpoint，词典就是它的“模型参数”；单句探针只跑一两次，
      重建 800 句的词典不到 0.1 秒，比额外维护一份词典文件更不容易过期。
      触发词按长度降序匹配，避免「质押」抢走「解除质押」。
    """
    rule = load_module("week7_rule", SRC / "rule" / "run_rule.py")
    schema = load_schema(data_dir / "event_schema.json")
    train_rows = load_extract_split("train", data_dir)
    trig_pairs = rule.merge_trigger_dict(
        rule.build_trigger_dict(train_rows, min_count=2), rule.MANUAL_TRIGGERS)
    company_dict = rule.build_company_dict(train_rows)
    return rule.extract_one(text, schema, trig_pairs, company_dict), {}


def infer_bilstm(text: str, data_dir: Path, ckpt_path: Path):
    """BiLSTM-CRF 单句推理。

    checkpoint 里同时存了 `vocab`、`labels`、`args`，所以推理不依赖训练时的数据集：
      - vocab：字 → id，未登录字用 <UNK>=1；
      - labels：149 个 BIO 标签的 id→name 映射；
      - args：emb_dim/hidden/num_layers/decoder 等结构参数，必须和训练时一致。
    这里不补 PAD：单句本身就是 batch=1 的最长序列，mask 全 1。
    """
    train_mod = load_module("week7_bilstm_train", SRC / "bilstm_crf" / "train.py")
    from crf import CRFClassifier

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab, labels, cfg = ckpt["vocab"], ckpt["labels"], ckpt["args"]
    label_scheme = {"labels": labels,
                    "label2id": {l: i for i, l in enumerate(labels)},
                    "id2label": {i: l for i, l in enumerate(labels)}}
    max_len = int(cfg.get("max_len", 256))
    chars = text[:max_len]
    ids = [vocab.get(ch, vocab.get("<UNK>", 1)) for ch in chars]
    input_ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(input_ids)

    encoder = train_mod.BiLSTMEncoder(
        len(vocab), emb_dim=int(cfg.get("emb_dim", 128)),
        hidden=int(cfg.get("hidden", 256)), num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.3)))
    model = CRFClassifier(encoder, hidden_size=encoder.hidden_size,
                          num_tags=len(labels), dropout=float(cfg.get("dropout", 0.3)),
                          use_crf=cfg.get("decoder", "crf") == "crf")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    paths, scores = model.predict(input_ids, mask, return_scores=True)
    tags = [labels[i] for i in paths[0]]
    event = spans_to_event(decode_bio_spans(tags), text, enums=load_enums(data_dir / "event_schema.json"))
    return event, {"confidence": round(float(scores[0]), 4), "n_truncated": len(text) > max_len}


def infer_bert(text: str, data_dir: Path, ckpt_path: Path, bert_path: str):
    """BERT+Linear/CRF 单句推理。

    两个容易踩的点：
      1. 这里用 `BioDataset(..., with_labels=False)`：纯推理不需要金标，
         但 dataset/collate 仍要生成 word_ids，才能把子词预测还原成字符级 BIO；
      2. checkpoint 里的 `args.bert_path` 可能是训练机器的绝对路径，换机器会失效，
         所以优先用命令行传入的 `--bert_path`。
    """
    from transformers import BertTokenizerFast
    from models import build_model, predict_char_tags

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label_scheme = ckpt["label_scheme"]
    id2label = ckpt["id2label"]
    cfg = ckpt["args"]

    class A:
        pass

    a = A()
    for k, v in cfg.items():
        setattr(a, k, v)
    a.bert_path = bert_path or cfg.get("bert_path")
    a.hard_constraint = False
    model = build_model(a, label_scheme)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(a.bert_path)
    row = {"id": "probe", "sent_id": "probe", "text": text, "event": None}
    ds = BioDataset([row], tokenizer, label_scheme,
                    max_length=int(cfg.get("max_len", 256)),
                    with_labels=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_bio)
    tags, _ = predict_char_tags(model, loader, id2label, torch.device("cpu"))
    event = spans_to_event(decode_bio_spans(tags[0]), text, enums=load_enums(data_dir / "event_schema.json"))
    return event, {"best_epoch": ckpt.get("epoch"), "inner_val_span_f1": ckpt.get("inner_val_span_f1")}


def main():
    ap = argparse.ArgumentParser(description="单句抽取探针")
    ap.add_argument("--text", required=True)
    ap.add_argument("--methods", default="rule,bilstm", help="rule,bilstm,bert 逗号分隔")
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--bilstm_ckpt", default=str(ROOT / "outputs" / "local_run" / "checkpoints" / "bilstm_crf_sample.pt"))
    ap.add_argument("--ckpt", default="", help="BERT checkpoint（--methods bert 时必填）")
    ap.add_argument("--bert_path", default="", help="bert-base-chinese 目录")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    # schema 只加载一次；每个方法单独 try/except，保证一个模型加载失败不影响其它方法
    schema = load_schema(data_dir / "event_schema.json")
    results = []

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        try:
            if method == "rule":
                event, meta = infer_rule(args.text, data_dir)
            elif method == "bilstm":
                event, meta = infer_bilstm(args.text, data_dir, Path(args.bilstm_ckpt))
            elif method == "bert":
                if not args.ckpt:
                    raise ValueError("--methods bert 需要 --ckpt 指向训练好的 BERT checkpoint")
                event, meta = infer_bert(args.text, data_dir, Path(args.ckpt), args.bert_path)
            else:
                raise ValueError(f"未知方法 {method}")
        except Exception as exc:
            results.append({"method": method, "verdict": "error", "error": str(exc)})
            continue
        checks, verdict = validate(args.text, event, schema)
        results.append({"method": method, "verdict": verdict, "event": event,
                        "checks": checks, **meta})

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print("输入：" + args.text)
    for r in results:
        print("\n" + "=" * 70)
        print(f"方法：{r['method']}  →  {r['verdict']}")
        if r.get("error"):
            print("  错误：" + r["error"])
            continue
        ev = r.get("event")
        if not ev:
            print("  ❌ 没有抽到事件（模型认为这句没有事件，或触发词未命中）")
        else:
            print(f"  事件类型：{ev.get('type')}")
            trig = ev.get("trigger") or {}
            print(f"  触发词：{trig.get('text')} [{trig.get('start')}:{trig.get('end')}]")
            for a in ev.get("args") or []:
                print(f"    - {a.get('role')}：{a.get('text')}")
        for c in r.get("checks", []):
            print(f"  {'✓' if c['ok'] else '✗'} {c['check']}：{c['detail']}")
        if r.get("confidence") is not None:
            print(f"  置信度：{r['confidence']}")
        print("  结论：" + {
            "ok": "结构合法，可抽取（内容是否正确需人工/金标核对）",
            "invalid": "抽到了事件，但结构不合法，不能直接入库",
            "empty": "未抽到事件；换一句有明确触发词的公告再试",
            "error": "运行失败，见上方错误",
        }.get(r["verdict"], r["verdict"]))


if __name__ == "__main__":
    main()

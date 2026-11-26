"""从 week7 的 DuEE-Fin 派生数据构建三个「换数据集」变体。

三种变体各自回答一个不同的问题：
  sent    : 原句级单事件（week7 口径）——对照组，不复制数据，直接引用
  heldout : 跨类型泛化——train 删掉 3 个类型的全部样本，dev 只看这 3 类。
            判别式的标签头见过全部 149 个标签名，但对应类型**没有任何训练样本**；
            LLM 只靠 schema 描述，理论上仍能抽。用来看「封闭标签模型的边界」。
  doc     : 篇章级——把 dev_docs 的 985 篇公告切句，offset 用 find 在句内重算，
            评测除了句级指标，额外算「整篇公告的事件集合 F1」。

所有构建统计落盘到 data/build_stats.json，不把「放弃的样本」藏在脚本里。
"""
import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_ROOT, WEEK7_DATA  # noqa: E402
from common.bridge import load_jsonl, save_jsonl  # noqa: E402

HELDOUT_TYPES = ["被约谈", "企业破产", "股东增持"]
SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])\s*")


def build_heldout() -> dict:
    out = DATA_ROOT / "heldout"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(WEEK7_DATA / "event_schema.json", out / "event_schema.json")

    train = load_jsonl(WEEK7_DATA / "train_extract.jsonl")
    dev = load_jsonl(WEEK7_DATA / "dev_extract.jsonl")
    tr = [r for r in train if r["event"]["type"] not in HELDOUT_TYPES]
    dv = [r for r in dev if r["event"]["type"] in HELDOUT_TYPES]
    save_jsonl(tr, out / "train_extract.jsonl")
    save_jsonl(dv, out / "dev_extract.jsonl")
    return {
        "heldout_types": HELDOUT_TYPES,
        "train_rows": [len(train), len(tr)],
        "dev_rows": [len(dev), len(dv)],
        "train_removed_by_type": dict(Counter(
            r["event"]["type"] for r in train if r["event"]["type"] in HELDOUT_TYPES)),
        "dev_by_type": dict(Counter(r["event"]["type"] for r in dv)),
    }


def _events_of(doc: dict) -> list[dict]:
    evs = doc.get("events") or []
    if evs and isinstance(evs[0], dict) and "type" in evs[0]:
        return evs
    if doc.get("event"):
        return [doc["event"]]
    return []


def build_doc() -> dict:
    """篇章切句：offset 在**句内**用 text.find 重算，不以原 doc offset 为准。"""
    out = DATA_ROOT / "doc"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(WEEK7_DATA / "event_schema.json", out / "event_schema.json")

    docs = load_jsonl(WEEK7_DATA / "dev_docs.jsonl")
    rows, index = [], []
    stat = Counter()
    for doc in docs:
        doc_id = str(doc.get("id") or doc.get("doc_id"))
        text = doc.get("text") or ""
        base = len(rows)   # rows 是跨文档的全局列表，句内序号必须加这个偏移
        sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]
        stat["docs"] += 1
        stat["sentences"] += len(sentences)

        row_ids, doc_gold = [], []
        for idx, sent in enumerate(sentences):
            rid = f"{doc_id}__s{idx}"
            row_ids.append(rid)
            rows.append({
                "id": rid, "sent_id": rid, "doc_id": doc_id, "sent_idx": idx,
                "text": sent, "event": None,
            })

        # 每个 gold 事件：触发词所在句 = 第一处 find 命中的句子；论元优先同句，其次全篇任一命中句
        for ev in _events_of(doc):
            trig = (ev.get("trigger") or {}).get("text", "")
            gargs = [(a.get("role", ""), a.get("text", "")) for a in (ev.get("args") or [])]
            doc_gold.append({"type": ev.get("type", ""), "trigger": trig,
                             "args": [{"role": r, "text": t} for r, t in gargs]})
            hits = [i for i, s in enumerate(sentences) if trig and trig in s]
            if not hits:
                stat["events_trigger_not_found"] += 1
                continue
            stat["events_trigger_found"] += 1
            # 选「触发词所在且论元最多」的一句作为句级 gold：
            # 篇章里同一事件的金额/时间常写在另一句，直接取第一处命中会把论元几乎全丢掉。
            si = max(hits, key=lambda i: sum(a[1] in sentences[i] for a in gargs))
            if rows[base + si]["event"] is not None:
                stat["events_same_sentence_overwritten"] += 1
            new_args, dropped = [], 0
            for role, arg_text in gargs:
                cand = None
                # 只保留触发词所在句内的论元：句级评测的 gold 不能带别的句子的 offset。
                # 跨句论元仍完整保留在 doc_index.json 的 doc_gold 里，篇章级评测照算。
                cand = sentences[si].find(arg_text) if arg_text in sentences[si] else -1
                if cand < 0:
                    dropped += 1
                    stat["args_cross_sentence_or_missing"] += 1
                    continue
                new_args.append({"role": role, "text": arg_text,
                                 "start": cand, "end": cand + len(arg_text)})
            tpos = sentences[si].find(trig)
            event = {
                "type": ev.get("type", ""),
                "trigger": {"text": trig, "start": tpos, "end": tpos + len(trig)},
                "args": new_args,
            }
            rows[base + si]["event"] = event   # 一句只放一个事件，与 week7 口径一致
            stat["events_placed"] += 1
            stat["args_dropped"] += dropped
            stat["events_trigger_multi_hit"] += int(len(hits) > 1)

        index.append({"doc_id": doc_id, "row_ids": row_ids, "gold_events": doc_gold})

    save_jsonl(rows, out / "dev_extract.jsonl")
    with open(out / "doc_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    return {**dict(stat), "sentence_rows": len(rows), "docs_indexed": len(index),
            "n_gold_events": sum(len(d["gold_events"]) for d in index)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["heldout", "doc"], default="")
    args = ap.parse_args()
    stats = {}
    if args.only in ("", "heldout"):
        stats["heldout"] = build_heldout()
    if args.only in ("", "doc"):
        stats["doc"] = build_doc()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(DATA_ROOT / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

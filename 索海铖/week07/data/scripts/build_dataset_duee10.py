# -*- coding: utf-8 -*-
"""按 build_dataset.py 的既有契约，把 DuEE 1.0（百度千言，65 类通用事件）
处理成 week7 事件抽取教学用的数据文件 + 事件 schema，**独立输出到 processed_duee10/**
（不覆盖 data/processed/ 下的 DuEE-Fin 产物，两套数据可并存对比）。

输入（由 convert_from_hf.py 还原的官方原始格式，见该脚本头注释的来源说明）：
  E:\\...\\datatasets\\DuEE1.0\\duee_train.json      官方原始 JSONL
  E:\\...\\datasets\\DuEE1.0\\duee_dev.json        官方原始 JSONL
  E:\\...\\datasets\\DuEE1.0\\duee_event_schema.json

与 DuEE-Fin 版（build_dataset.py）的四处差异，也是本脚本存在的意义：
  1) **span 定位策略相反**：DuEE-Fin 镜像的 start_index 不可靠，只能 text.find()；
     DuEE 1.0 的坐标是准的（已全量校验 text[start:end]==span.text），
     所以优先采信官方 start_index，find 只做兜底；
  2) **没有 sent_id**：DuEE 1.0 每行就是独立句子，用 id 兼任 sent_id
     （下游 make_pred_row 只要求字段存在，按 id|sent_id 对齐仍唯一）；
  3) **没有篇章级文件**：dev_docs 退化为「单句即一篇」（title 置空），
     保证 service/demo 的 dev_docs 输入不断档；
  4) **schema 无枚举角色**：DuEE-Fin 的「公司上市.环节」枚举在 DuEE 1.0 不存在，
     __enums__ 输出空 dict（load_enums 本来就容许为空）。

产出契约（与 build_dataset.py 完全一致，写入 data/processed_duee10/）：
  event_schema.json / train_extract.jsonl / dev_extract.jsonl /
  train_detect.jsonl / dev_detect.jsonl / dev_docs.jsonl
统计审计写入 data/stats/build_stats_duee10.json。

使用（训练侧入口无需改代码，指 --data_dir 即可）：
  python bert/train.py --data_dir data/processed_duee10 --max_len 128
"""
import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent          # code/data
RAW =  DATA / "raw"    
PROC = DATA / "processed_duee10"
STATS = DATA / "stats"
SENT = {"train": RAW / "duee_train.json", "dev": RAW / "duee_dev.json"}
SCHEMA_PATH = RAW / "duee_event_schema.json"


def iter_lines(path):
    """duee_train/dev.json 是 JSONL（每行一个 JSON 对象），逐行读。"""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for row in json.load(f):
                yield row
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def locate_span(text, span_text, start_index):
    """DuEE 1.0 版定位：优先采信官方 start_index（坐标已校验可靠），
    校验不过再退回 text.find()（多次出现取第一次）。
    与 build_dataset.py 的 find_span 策略相反，原因见模块 docstring 第 1 条。
    返回 (start, end, from_official_index) 或 None。
    """
    if start_index is not None and 0 <= start_index <= len(text) - len(span_text):
        if text[start_index:start_index + len(span_text)] == span_text:
            return start_index, start_index + len(span_text), True
    pos = text.find(span_text)
    if pos < 0:
        return None
    return pos, pos + len(span_text), False


def pct(values):
    """分位数摘要（与 build_dataset.py 同口径）。"""
    if not values:
        return {}
    s = sorted(values)
    return {"n": len(s), "mean": round(sum(s) / len(s), 2), "p50": s[len(s) // 2],
            "p90": s[int(len(s) * 0.9)], "p99": s[int(len(s) * 0.99)], "max": s[-1]}


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 1) schema
def build_schema():
    """65 行 jsonl → 事件类型→角色列表；DuEE 1.0 没有 enum_items，__enums__ 为空。"""
    type_roles = {}
    for row in iter_lines(SCHEMA_PATH):
        type_roles[row["event_type"]] = [item["role"] for item in row["role_list"]]
    type_roles["__enums__"] = {}
    (PROC / "event_schema.json").write_text(
        json.dumps(type_roles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return type_roles


# ---------------------------------------------------------------- 2/3) extract
def build_extract(sentence_rows, split, stats):
    """单事件句子 → 带 span 的训练样本；统计官方坐标采信率与丢弃量。"""
    rows_out = []
    n_all = len(sentence_rows)
    n_one = 0
    n_sample_dropped = 0
    n_args_total = n_args_kept = 0
    drop_by_role = Counter()
    n_index_used = n_find_fallback = 0   # span 定位策略统计：官方坐标 vs find 兜底
    ambig_spans = n_spans = 0
    lens_extract = []
    args_per_ev = []
    trig_lens = []
    n_trigger_eq_arg = 0
    n_overlap_events = 0
    for row in sentence_rows:
        evs = row.get("event_list") or []
        if len(evs) != 1:
            continue
        n_one += 1
        text = row["text"]
        ev = evs[0]
        trig = locate_span(text, ev["trigger"], ev.get("trigger_start_index"))
        if trig is None:  # 理论上不会发生（官方坐标已全量校验）
            n_sample_dropped += 1
            continue
        n_spans += 1
        if trig[2]:
            n_index_used += 1
        else:
            n_find_fallback += 1
        if text.count(ev["trigger"]) > 1:
            ambig_spans += 1
        args = []
        for a in ev.get("arguments") or []:
            n_args_total += 1
            role, arg = a["role"], a["argument"]
            sp = locate_span(text, arg, a.get("argument_start_index"))
            if sp is None:  # 论元是归一化取值（原文没有这串字）
                drop_by_role[role] += 1
                continue
            n_spans += 1
            if sp[2]:
                n_index_used += 1
            else:
                n_find_fallback += 1
            if text.count(arg) > 1:
                ambig_spans += 1
            args.append({"role": role, "text": arg, "start": sp[0], "end": sp[1]})
            n_args_kept += 1
        # 教学坑（与 DuEE-Fin 同款）：trigger 与论元可能落在同一/重叠区间
        spans_all = [(trig[0], trig[1])] + [(a["start"], a["end"]) for a in args]
        if any((a["start"], a["end"]) == (trig[0], trig[1]) for a in args):
            n_trigger_eq_arg += 1
        overlap = False
        for i in range(len(spans_all)):
            for j in range(i + 1, len(spans_all)):
                x, y = spans_all[i], spans_all[j]
                if x[0] < y[1] and y[0] < x[1]:
                    overlap = True
        if overlap:
            n_overlap_events += 1
        rows_out.append({
            # DuEE 1.0 没有 sent_id（句子独立成行），用 id 兼任，保证 id|sent_id 仍唯一
            "id": row.get("id"),
            "sent_id": row.get("id"),
            "text": text,
            "event": {
                "type": ev["event_type"],
                "trigger": {"text": ev["trigger"], "start": trig[0], "end": trig[1]},
                "args": args,
            },
        })
        lens_extract.append(len(text))
        args_per_ev.append(len(args))
        trig_lens.append(len(ev["trigger"]))
    stats["extract_" + split] = {
        "n_sentence_rows": n_all,
        "n_rows_with_exactly_1_event": n_one,
        "n_output_rows": len(rows_out),
        "n_samples_dropped_trigger_not_found": n_sample_dropped,
        "n_args_total": n_args_total,
        "n_args_kept": n_args_kept,
        "n_args_dropped": n_args_total - n_args_kept,
        "arg_drop_rate": round((n_args_total - n_args_kept) / max(1, n_args_total), 6),
        "arg_drop_by_role": dict(drop_by_role.most_common()),
        "n_spans": n_spans,
        "n_spans_from_official_index": n_index_used,
        "n_spans_from_find_fallback": n_find_fallback,
        "n_spans_string_ambiguous": ambig_spans,
        "span_ambiguity_rate": round(ambig_spans / max(1, n_spans), 6),
        "n_events_trigger_equals_arg_span": n_trigger_eq_arg,
        "n_events_with_any_span_overlap": n_overlap_events,
        "sentence_len": pct(lens_extract),
        "trigger_len": pct(trig_lens),
        "args_per_event": pct(args_per_ev),
    }
    dump_jsonl(PROC / ("%s_extract.jsonl" % split), rows_out)
    return rows_out


# ---------------------------------------------------------------- 4/5) detect
def build_detect(sentence_rows, split, stats):
    """全部句子 → 句子级事件检测样本（DuEE 1.0 全量句子都带事件，负样本=0）。"""
    rows_out = []
    n_ev_hist = Counter()
    type_dist = Counter()
    lens = []
    for row in sentence_rows:
        evs = row.get("event_list") or []
        types = sorted({ev["event_type"] for ev in evs})
        rows_out.append({
            "id": row.get("id"),
            "sent_id": row.get("id"),
            "text": row["text"],
            "has_event": bool(evs),
            "types": types,
            "n_events": len(evs),
        })
        lens.append(len(row["text"]))
        n_ev_hist[len(evs)] += 1
        for t in types:
            type_dist[t] += 1
    stats["detect_" + split] = {
        "n_rows": len(rows_out),
        "n_positive": sum(1 for r in rows_out if r["has_event"]),
        "n_negative": sum(1 for r in rows_out if not r["has_event"]),
        "positive_rate": round(sum(1 for r in rows_out if r["has_event"]) / max(1, len(rows_out)), 4),
        "n_events_hist": dict(sorted(n_ev_hist.items())),
        "text_len": pct(lens),
        "type_distribution": dict(type_dist.most_common()),
    }
    dump_jsonl(PROC / ("%s_detect.jsonl" % split), rows_out)
    return rows_out


# ---------------------------------------------------------------- 6) dev docs
def build_dev_docs(dev_rows, stats):
    """DuEE 1.0 没有篇章级数据：把「有事件的 dev 单句」当一篇文档（单句即篇）。
    契约字段与 DuEE-Fin 版一致（id/title/text/events），title 置空；
    保证 service/demo 与 load_docs("dev", data_dir=...) 不断档。
    """
    rows_out = []
    for row in dev_rows:
        evs = row.get("event_list") or []
        if not evs:
            continue
        text = row["text"]
        events = []
        for ev in evs:
            trig = locate_span(text, ev["trigger"], ev.get("trigger_start_index"))
            if trig is None:
                continue
            args = []
            for a in ev.get("arguments") or []:
                sp = locate_span(text, a["argument"], a.get("argument_start_index"))
                if sp is None:
                    continue
                args.append({"role": a["role"], "text": a["argument"], "start": sp[0], "end": sp[1]})
            events.append({
                "type": ev["event_type"],
                "trigger": {"text": ev["trigger"], "start": trig[0], "end": trig[1]},
                "args": args,
            })
        if not events:
            continue
        rows_out.append({"id": row.get("id"), "title": "", "text": text, "events": events})
    stats["dev_docs"] = {
        "n_output_docs": len(rows_out),
        "n_events_in_output": sum(len(r["events"]) for r in rows_out),
        "note": "DuEE 1.0 无篇章级数据，单句即一篇（title 为空串）",
        "doc_len": pct([len(r["text"]) for r in rows_out]),
    }
    dump_jsonl(PROC / "dev_docs.jsonl", rows_out)
    return rows_out


# ---------------------------------------------------------------- 主流程
def main():
    PROC.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    stats = {}
    schema = build_schema()
    stats["schema"] = {
        "n_event_types": len([k for k in schema if k != "__enums__"]),
        "n_roles_per_type": {k: len(v) for k, v in schema.items() if k != "__enums__"},
        "n_union_roles": len({r for v in schema.values() if isinstance(v, list) for r in v}),
    }

    sent = {split: list(iter_lines(SENT[split])) for split in ("train", "dev")}
    extract = {split: build_extract(sent[split], split, stats) for split in ("train", "dev")}
    build_detect(sent["train"], "train", stats)
    build_detect(sent["dev"], "dev", stats)

    # 单/多事件比例（DuEE 1.0 有不少多事件句，extract 只取单事件子集，必须报占比）
    for split in ("train", "dev"):
        hist = Counter(len(r.get("event_list") or []) for r in sent[split])
        stats["sentence_events_hist_" + split] = dict(sorted(hist.items()))
        stats["sentence_events_hist_" + split]["_n_rows"] = len(sent[split])

    # 类型 / 角色分布（extract 子集）
    def dist(rows):
        types, roles = Counter(), Counter()
        for r in rows:
            types[r["event"]["type"]] += 1
            for a in r["event"]["args"]:
                roles[a["role"]] += 1
        return dict(types.most_common()), dict(roles.most_common())

    stats["extract_train_type_distribution"], stats["extract_train_role_distribution"] = dist(extract["train"])
    stats["extract_dev_type_distribution"], _ = dist(extract["dev"])

    build_dev_docs(sent["dev"], stats)

    # 重复 / 泄漏检查（id 是 md5(文本+序号)，文本级查重更本质）
    tr = [r["text"].strip() for r in sent["train"]]
    dv = [r["text"].strip() for r in sent["dev"]]
    stats["dedup_leakage"] = {
        "train_dup_texts": len(tr) - len(set(tr)),
        "dev_dup_texts": len(dv) - len(set(dv)),
        "train_dev_overlap": len(set(tr) & set(dv)),
    }

    # 文件体积
    sizes = {}
    for p in sorted(PROC.iterdir()):
        sizes[p.name] = round(p.stat().st_size / 1024 / 1024, 2)
    stats["processed_file_size_mb"] = sizes
    stats["processed_total_mb"] = round(sum(sizes.values()), 2)

    (STATS / "build_stats_duee10.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("schema types:", stats["schema"]["n_event_types"],
          "| union roles:", stats["schema"]["n_union_roles"])
    for split in ("train", "dev"):
        e = stats["extract_" + split]
        d = stats["detect_" + split]
        print("%s: extract rows=%d (单事件句=%d, trigger丢弃=%d, 论元 %d/%d 保留, 丢弃率 %.2f%%) | detect rows=%d (pos=%d)" % (
            split, e["n_output_rows"], e["n_rows_with_exactly_1_event"], e["n_samples_dropped_trigger_not_found"],
            e["n_args_kept"], e["n_args_total"], 100 * e["arg_drop_rate"], d["n_rows"], d["n_positive"]))
    print("dev_docs:", json.dumps({k: v for k, v in stats["dev_docs"].items() if not isinstance(v, dict)},
                                  ensure_ascii=False))
    print("processed total MB:", stats["processed_total_mb"])
    print("leakage:", json.dumps(stats["dedup_leakage"], ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:                      # 允许覆盖原始数据目录
        RAW = Path(sys.argv[1])
        SENT = {"train": RAW / "duee_train.json", "dev": RAW / "duee_dev.json"}
        SCHEMA_PATH = RAW / "duee_event_schema.json"
    main()

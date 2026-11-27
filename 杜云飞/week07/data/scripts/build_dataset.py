# -*- coding: utf-8 -*-
"""
FewFC original -> 课程派生格式（build_dataset）

输入：data/raw/FewFC/original/{train_base.json, dev_base.json}（JSONL）
输出：data/processed/
  - event_schema.json   {事件类型: [角色...]}（角色按数据中出现频次排序）
  - {train,dev}_extract.jsonl  一行 = 一句原文 + 恰好 1 个事件
        {id, sent_id, text, event:{type, trigger:{text,start,end}, args:[{role,text,start,end}]}}
  - build_stats.json    构建口径与审计统计（不填估算值，全部实测）

转换纪律（与课程 build_dataset.py 同口径）：
  1. offset 一律用 text.find() 重算、首次出现为准（不信任原始 offset）；
  2. 逐行校验 text[start:end] == span.text，对不上的 span 丢弃并记账；
  3. 一句多事件 -> 拆成多行（每行 1 个事件），sent_id = md5(id|事件序号)；
  4. 同一句内「类型+触发词+论元集合」完全相同的重复事件去重；
  5. FewFC 没有"无事件"句（全部句子都含事件），detect 任务不适用，不设 detect 文件；
  6. 多事件句中"同一角色多个不同取值"（如两个减持方）按课程口径：BIO 一行表达不了，
     拆行后每个事件各自成行即可表达——FewFC 的标注本身就是按事件分开的，天然兼容。
"""
import json
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "FewFC" / "original"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# 三个源文件 → 三个口径（FewFC 发布即"base 5 类全量 + trans 5 类迁移"两段，拼成统一 10 类）：
#   train = train_base + train_trans_5_type   训练集（全量标注路线）
#   dev   = dev_base + dev_trans              开发集（选点/调参/中期对比唯一入口）
#   test  = test_trans                        最终测试集（纪律：所有方法定型前不碰，
#                                              报告里只跑一次，对应课程"dev 不做模型选择"的加强版）
SPLIT_SOURCES = {
    "train": ["train_base", "train_trans_5_type"],
    "dev": ["dev_base", "dev_trans"],
    "test": ["test_trans"],
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def relocate(text, word, orig_span):
    """用 find() 重算 offset，首次出现为准；找不到返回 None。"""
    i = text.find(word)
    if i < 0:
        return None
    return i, i + len(word)


def convert(rows, split):
    out_rows = []
    stats = Counter()
    type_counter = Counter()
    role_counter = Counter()
    for row in rows:
        text = row["content"]
        stats["n_sents"] += 1
        seen = set()  # 同句去重：(type, trigger, 排序后的 args)
        for ev in row.get("events", []):
            stats["n_events_raw"] += 1
            trigger = None
            args = []
            for m in ev["mentions"]:
                role = m["role"]
                word = m["word"]
                sp = relocate(text, word, m["span"])
                if sp is None:
                    stats["span_not_found"] += 1
                    continue
                s, e = sp
                # 二次校验：原文切片必须等于标注词
                if text[s:e] != word:
                    stats["span_mismatch"] += 1
                    continue
                if role == "trigger":
                    if trigger is None:
                        trigger = {"text": word, "start": s, "end": e}
                    else:
                        stats["dup_trigger_in_event"] += 1
                else:
                    args.append({"role": role, "text": word, "start": s, "end": e})
            if trigger is None:
                stats["event_no_trigger_dropped"] += 1
                continue
            key = (ev["type"], trigger["text"],
                   tuple(sorted((a["role"], a["text"]) for a in args)))
            if key in seen:
                stats["event_dupe_dropped"] += 1
                continue
            seen.add(key)
            ev_id = hashlib.md5(f"{row['id']}|{len(seen)}".encode()).hexdigest()
            out_rows.append({
                "id": row["id"],
                "sent_id": ev_id,
                "text": text,
                "event": {"type": ev["type"], "trigger": trigger, "args": args},
            })
            type_counter[ev["type"]] += 1
            for a in args:
                role_counter[a["role"]] += 1
    return out_rows, stats, type_counter, role_counter


def main():
    schema = {}
    all_stats = {"splits": {}}
    split_texts = {}  # split -> set(content)，用于训练/开发/测试泄漏审计
    for split in ["train", "dev", "test"]:
        rows = []
        for src in SPLIT_SOURCES[split]:
            rows.extend(load_jsonl(RAW / f"{src}.json"))
        out_rows, stats, type_counter, role_counter = convert(rows, split)
        split_texts[split] = set(r["content"] for r in rows)
        with open(OUT / f"{split}_extract.jsonl", "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for t, c in type_counter.items():
            schema.setdefault(t, Counter())[""]  # placeholder
        # schema 的角色集合：用 train+dev 全部数据的角色（课程口径：schema 来自全量标注）
        all_stats["splits"][split] = {
            "n_sents": stats["n_sents"],
            "n_rows": len(out_rows),
            "n_events_raw": stats["n_events_raw"],
            "span_not_found": stats["span_not_found"],
            "span_mismatch": stats["span_mismatch"],
            "event_no_trigger_dropped": stats["event_no_trigger_dropped"],
            "event_dupe_dropped": stats["event_dupe_dropped"],
            "dup_trigger_in_event": stats["dup_trigger_in_event"],
        }
        print(f"[{split}] sents={stats['n_sents']} rows={len(out_rows)} "
              f"raw_events={stats['n_events_raw']} not_found={stats['span_not_found']} "
              f"dropped_no_trigger={stats['event_no_trigger_dropped']} "
              f"dupe={stats['event_dupe_dropped']}")
        # 保存每个 split 的 role counter 供 schema 汇总
        all_stats.setdefault("_roles", Counter())
        all_stats["_roles"].update(role_counter)
        all_stats.setdefault("_types", Counter())
        all_stats["_types"].update(type_counter)

    # 泄漏审计：train/dev、train/test 的文本重叠（课程 DuEE-Fin 有 9.2% 泄漏的先例，必须查）
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        ov = len(split_texts[a] & split_texts[b])
        all_stats.setdefault("text_overlap", {})[f"{a}_vs_{b}"] = {
            "overlap_sents": ov,
            "ratio_in_{}".format(b): round(ov / max(1, len(split_texts[b])), 4),
        }
    print("text_overlap:", json.dumps(all_stats["text_overlap"], ensure_ascii=False))

    # schema：{type: [roles]}，角色按频次降序（train+dev 全量；test 不偷看 schema 之外的类型）
    type_roles = {}
    rows_all = []
    for split in ["train", "dev"]:
        with open(OUT / f"{split}_extract.jsonl", encoding="utf-8") as f:
            rows_all.extend((json.loads(l) for l in f))
    for r in rows_all:
        t = r["event"]["type"]
        type_roles.setdefault(t, Counter())
        for a in r["event"]["args"]:
            type_roles[t][a["role"]] += 1
    schema = {t: [role for role, _ in c.most_common()] for t, c in sorted(type_roles.items())}
    with open(OUT / "event_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=1)

    all_stats.pop("_roles", None)
    all_stats["types"] = dict(all_stats.pop("_types"))
    all_stats["schema"] = {"n_types": len(schema),
                           "roles_per_type": {t: len(rs) for t, rs in schema.items()}}
    # 句子长度统计（按 (id,text) 去重）
    import statistics
    sents = {}
    for r in rows_all:
        sents.setdefault((r["id"], r["text"]), r["text"])
    lens = [len(t) for t in sents.values()]
    all_stats["sentence_length"] = {
        "n_unique_sents": len(lens), "mean": round(statistics.mean(lens), 1),
        "median": statistics.median(lens), "max": max(lens),
        "over256": sum(1 for x in lens if x > 256),
        "over512": sum(1 for x in lens if x > 512),
    }
    # 同句多事件比例：按 (id, text) 聚合（sent_id 是事件级唯一键，不能直接数）
    per_sent = Counter()
    for r in rows_all:
        per_sent[(r["id"], r["text"])] += 1
    multi = sum(1 for v in per_sent.values() if v > 1)
    all_stats["multi_event_sents"] = {
        "n": multi, "n_unique_sents": len(per_sent),
        "ratio": round(multi / len(per_sent), 4),
        "max_events_per_sent": max(per_sent.values()),
        "rows_per_sent_mean": round(len(rows_all) / len(per_sent), 2),
    }
    # span 在句内多次出现比例（find 取首次的精度上限损失）
    amb = tot = 0
    for r in rows_all:
        t = r["event"]["trigger"]["text"]
        tot += 1
        if t and t in r["text"]:
            if r["text"].count(t) > 1:
                amb += 1
    all_stats["trigger_ambiguous_ratio"] = round(amb / tot, 4)

    # dev_clean：剔除与 train 文本重叠的句子（课程口径：clean 版才代表真实泛化，双报）
    train_texts = split_texts["train"]
    clean_rows, full_rows = [], []
    with open(OUT / "dev_extract.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            full_rows.append(r)
            if r["text"] not in train_texts:
                clean_rows.append(r)
    with open(OUT / "dev_extract_clean.jsonl", "w", encoding="utf-8") as f:
        for r in clean_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    all_stats["dev_clean"] = {"n_rows_full": len(full_rows), "n_rows_clean": len(clean_rows),
                              "dropped": len(full_rows) - len(clean_rows)}

    # 固定 n=200 句子级评测子集（LLM 侧与判别式侧共用的唯一可比口径）
    # canonical_order：按 (id, text) 排序后 shuffle(seed=42) 取前 200 句——
    # 抽样规则只此一份，n=100/50 是 n=200 的前缀（课程坑 #21：两个脚本各自采样=样本不重合）
    import random
    dev_sents = {}
    for r in full_rows:
        dev_sents.setdefault((r["id"], r["text"]), None)
    keys = sorted(dev_sents.keys())
    rng = random.Random(42)
    rng.shuffle(keys)
    chosen = set(keys[:200])
    n200_rows = [r for r in full_rows if (r["id"], r["text"]) in chosen]
    with open(OUT / "dev_n200.jsonl", "w", encoding="utf-8") as f:
        for r in n200_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "eval_ids.json", "w", encoding="utf-8") as f:
        json.dump([{"id": k[0], "text_md5": hashlib.md5(k[1].encode()).hexdigest()} for k in keys[:200]],
                  f, ensure_ascii=False, indent=1)
    all_stats["eval_subset"] = {"n_sents": 200, "n_rows": len(n200_rows)}

    with open(OUT / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=1)
    print("schema types:", len(schema))
    print(json.dumps(all_stats["schema"], ensure_ascii=False))


if __name__ == "__main__":
    main()

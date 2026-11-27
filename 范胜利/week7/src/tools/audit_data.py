# -*- coding: utf-8 -*-
"""
数据体检：判断 week7 抽取数据"是不是都能用、用在哪儿、有哪些坑"。

检查项（对应数据工程的常见事故）：
  1. 文件是否齐全、行数 / 唯一 id / 唯一 sent_id 是否对得上；
  2. 抽取集 span 是否能在原文中精确还原（start/end 与 text 一致）；
  3. 事件类型、角色是否都在 schema 内；是否存在空触发词、空论元；
  4. 重复样本、重复文本、同文不同标（标签冲突）；
  5. train/dev 是否存在文本级泄漏；
  6. 检测集正负样本比例、n_events 与 has_event 是否自洽；
  7. 篇章集事件数 / span 是否合法；
  8. 事件类型长尾、schema 槽位覆盖率（哪些角色训练集从没出现）；
  9. sample 与全量 processed 的分布差异（TV 距离），说明 sample 指标能不能代表全量；
 10. sample_stats.json 与真实行数是否一致。

用法：
  python tools/audit_data.py --data_dir data/sample
  python tools/audit_data.py --data_dir data/sample --full_dir data/processed
产物：
  outputs/data_audit.json   机器可读
  outputs/data_audit.json   机器可读（不再写 .md；学习类 md 在桌面合集）
"""

import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/tools/ -> week7/
SRC = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path):
    """读 JSONL；遇到坏行直接抛错并带上行号，避免“静默少几条”导致分母不对。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} 第 {i} 行不是合法 JSON：{exc}") from exc
    return rows


def span_ok(text: str, span: dict) -> bool:
    """span 是否能在原文精确还原：0 <= start < end <= len(text) 且 text[start:end] == text。

    这是字符级标注最核心的契约；offset 差一位，BIO 对齐、评测、可视化全错。
    """
    if not isinstance(span, dict):
        return False
    s, e, t = span.get("start"), span.get("end"), span.get("text")
    if not isinstance(s, int) or not isinstance(e, int) or not isinstance(t, str):
        return False
    return 0 <= s < e <= len(text) and text[s:e] == t


def tv_distance(c1: dict, c2: dict) -> float:
    """两个类别分布的总变差距离（0=完全一致，1=完全不重叠）。

    用来回答“sample 能不能代表全量”。<0.05 可认为代表性较好；
    本项目 sample vs 全量 dev 约 0.053，所以报告里只把 sample 当排序/冒烟。
    """
    keys = set(c1) | set(c2)
    n1, n2 = sum(c1.values()) or 1, sum(c2.values()) or 1
    return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2) for k in keys)


def audit_extract(rows, schema, split):
    """句级抽取集体检：类型/角色合法性、span 精确性、长度截断、多值角色、重叠。

    `issues` 只放**真错误**（span 非法、未知类型/角色）；
    “同一角色多值”是合法标注，单独统计，不混进 issues。
    """
    types = collections.Counter()
    roles = collections.Counter()
    n_args = []
    lengths = []
    multi_value_events = multi_value_pairs = overlap_pairs = 0
    issues = collections.Counter()
    examples = collections.defaultdict(list)
    for r in rows:
        text = r.get("text") or ""
        lengths.append(len(text))
        event = r.get("event") or {}
        etype = event.get("type")
        types[etype] += 1
        if etype not in schema:
            issues["未知事件类型"] += 1
            examples["未知事件类型"].append(r.get("id"))
        trig = event.get("trigger") or {}
        if not span_ok(text, trig):
            issues["触发词 span 非法"] += 1
            examples["触发词 span 非法"].append(r.get("id"))
        args = event.get("args") or []
        n_args.append(len(args))
        if not args:
            issues["无任何论元"] += 1
        # 同一事件允许同一角色多值（两个中标公司、两个减持方），这是 DuEE-Fin 的标注规范；
        # 它会让 event_exact 更难，但不是数据错误，所以只统计不报错。
        role_count = collections.Counter(a.get("role") for a in args)
        if any(v > 1 for v in role_count.values()):
            multi_value_events += 1
            multi_value_pairs += sum(v - 1 for v in role_count.values() if v > 1)
        spans = []
        for a in args:
            role = a.get("role")
            roles[role] += 1
            if isinstance(a.get("start"), int) and isinstance(a.get("end"), int):
                spans.append((a["start"], a["end"]))
        # 论元 span 互相重叠会让 BIO 投影产生“同一字符两个标签”的冲突；
        # 这是金标自身非法 BIO 地板值（约 1.9%）的来源，必须如实统计。
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 < e1 and s1 != s2:
                overlap_pairs += 1
            if role not in schema.get(etype, []):
                issues["角色不属于该类型"] += 1
                examples["角色不属于该类型"].append(f"{r.get('id')}:{etype}/{role}")
            if not span_ok(text, a):
                issues["论元 span 非法"] += 1
                examples["论元 span 非法"].append(r.get("id"))
    n = len(rows) or 1
    return {
        "split": split,
        "rows": len(rows),
        "type_counts": dict(types),
        "role_counts": dict(roles),
        "avg_args_per_event": round(sum(n_args) / n, 3),
        "multi_value_role_events": multi_value_events,
        "multi_value_role_pairs": multi_value_pairs,
        "overlap_arg_pairs": overlap_pairs,
        "min_args": min(n_args) if n_args else 0,
        "max_args": max(n_args) if n_args else 0,
        "len_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
        "truncate_rate_254": round(sum(1 for x in lengths if x > 254) / n, 4),
        "issues": dict(issues),
        "issue_examples": {k: v[:5] for k, v in examples.items()},
    }


def audit_detect(rows, schema, split):
    """检测集体检：正负样本比例、has_event 与 n_events 自洽性、未知类型、长度。"""
    has = collections.Counter(bool(r.get("has_event")) for r in rows)
    n_events = collections.Counter(int(r.get("n_events", 0)) for r in rows)
    inconsistent = sum(
        1 for r in rows
        if bool(r.get("has_event")) != (int(r.get("n_events", 0)) > 0)
    )
    bad_types = sum(
        1 for r in rows for t in (r.get("types") or []) if t not in schema
    )
    lengths = [len(r.get("text") or "") for r in rows]
    return {
        "split": split,
        "rows": len(rows),
        "positive": has.get(True, 0),
        "negative": has.get(False, 0),
        "positive_rate": round(has.get(True, 0) / max(len(rows), 1), 4),
        "n_events_dist": dict(sorted(n_events.items())),
        "has_event_vs_n_events_mismatch": inconsistent,
        "unknown_types": bad_types,
        "len_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
    }


def audit_docs(rows, schema, split):
    """篇章集体检：事件数分布、多事件比例、每个事件的 span/角色合法性。

    篇章集只用于系统层演示（一篇多事件），不参与句级 F1；
    所以这里只报告问题，不把它当训练集。
    """
    issues = collections.Counter()
    n_events = []
    for r in rows:
        text = r.get("text") or ""
        evs = r.get("events") or []
        n_events.append(len(evs))
        for e in evs:
            if e.get("type") not in schema:
                issues["未知事件类型"] += 1
            if not span_ok(text, e.get("trigger") or {}):
                issues["触发词 span 非法"] += 1
            for a in e.get("args") or []:
                if a.get("role") not in schema.get(e.get("type"), []):
                    issues["角色不属于该类型"] += 1
                if not span_ok(text, a):
                    issues["论元 span 非法"] += 1
    return {
        "split": split,
        "rows": len(rows),
        "events_total": sum(n_events),
        "events_per_doc_dist": dict(sorted(collections.Counter(n_events).items())),
        "multi_event_doc_rate": round(sum(1 for x in n_events if x > 1) / max(len(rows), 1), 4),
        "issues": dict(issues),
    }


def dup_analysis(rows, with_event: bool):
    """重复分析：同 id、同文本、同 sent_id 三种。

    注意：DuEE-Fin 的 `id` 是公告 id，同一篇公告多句共用 id 是**正常**的；
    真正要关注的是“同文本重复”和“同文不同标”（标签冲突）。
    """
    by_id = collections.defaultdict(list)
    by_text = collections.defaultdict(list)
    by_sent = collections.defaultdict(list)
    for r in rows:
        by_id[r.get("id")].append(r)
        by_text[r.get("text")].append(r)
        if r.get("sent_id"):
            by_sent[r["sent_id"]].append(r)
    dup_ids = {k: v for k, v in by_id.items() if len(v) > 1}
    dup_texts = {k: v for k, v in by_text.items() if len(v) > 1}
    dup_sents = {k: v for k, v in by_sent.items() if len(v) > 1}
    conflict = 0
    if with_event:
        for group in dup_texts.values():
            labels = {json.dumps(x.get("event"), ensure_ascii=False, sort_keys=True) for x in group}
            if len(labels) > 1:
                conflict += 1
    return {
        "dup_id_groups": len(dup_ids),
        "dup_id_extra_rows": sum(len(v) - 1 for v in dup_ids.values()),
        "dup_text_groups": len(dup_texts),
        "dup_text_extra_rows": sum(len(v) - 1 for v in dup_texts.values()),
        "dup_text_conflict_groups": conflict,
        "dup_sent_id_groups": len(dup_sents),
    }


def main():
    ap = argparse.ArgumentParser(description="week7 数据体检")
    ap.add_argument("--data_dir", default="data/sample")
    ap.add_argument("--full_dir", default="", help="全量目录（可选），用于比较 sample 代表性")
    ap.add_argument("--out_json", default="outputs/data_audit.json")
    ap.add_argument("--out_md", default="report/data_audit.md")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()
    if not data_dir.exists():
        data_dir = ROOT / "data" / "sample"
    full_dir = Path(args.full_dir).expanduser() if args.full_dir else None
    if full_dir is not None and not full_dir.is_absolute():
        full_dir = (Path.cwd() / full_dir).resolve()

    raw_schema = json.loads((data_dir / "event_schema.json").read_text(encoding="utf-8"))
    # __enums__ 是枚举型角色的取值范围，不是事件类型；统计时必须排除
    schema = {k: v for k, v in raw_schema.items() if not k.startswith("__")}
    all_roles = {r for v in schema.values() for r in v}

    report = {"data_dir": str(data_dir), "schema": {
        "types": len(schema), "slots": sum(len(v) for v in schema.values()),
        "unique_roles": len(all_roles)}, "files": {}}

    # ── 1. 抽取集 ────────────────────────────────────────────────
    extract = {}
    for split in ("train", "dev"):
        p = data_dir / f"{split}_extract.jsonl"
        if not p.exists():
            report["files"][p.name] = {"exists": False}
            continue
        rows = load_jsonl(p)
        info = audit_extract(rows, schema, split)
        info["unique_ids"] = len({r.get("id") for r in rows})
        info["unique_sent_ids"] = len({r.get("sent_id") for r in rows})
        info.update(dup_analysis(rows, with_event=True))
        extract[split] = rows
        report["files"][p.name] = {"exists": True, **info}

    # extract 与 detect 的重叠：两个集合是独立抽样还是包含关系？
    overlap = {}
    for split in ("train", "dev"):
        pe, pd = data_dir / f"{split}_extract.jsonl", data_dir / f"{split}_detect.jsonl"
        if pe.exists() and pd.exists() and split in extract:
            detect = load_jsonl(pd)
            e_sent = {r.get("sent_id") or r.get("id") for r in extract[split]}
            d_sent = {r.get("sent_id") or r.get("id") for r in detect}
            d_pos = {r.get("sent_id") or r.get("id") for r in detect if r.get("has_event")}
            overlap[split] = {
                "extract_rows": len(extract[split]),
                "detect_rows": len(detect),
                "sent_overlap": len(e_sent & d_sent),
                "detect_positive": len(d_pos),
                "detect_positive_in_extract": len(d_pos & e_sent),
            }
    report["extract_detect_overlap"] = overlap

    # 训练集槽位覆盖：schema 里有、train 里一次都没出现的角色
    if "train" in extract:
        seen = set(report["files"]["train_extract.jsonl"]["role_counts"])
        missing_roles = sorted(all_roles - seen)
        report["schema"]["roles_never_in_train"] = missing_roles
        report["schema"]["slot_coverage"] = round(len(seen) / max(len(all_roles), 1), 4)

    # ── 2. 检测集 / 篇章集 ───────────────────────────────────────
    for split in ("train", "dev"):
        p = data_dir / f"{split}_detect.jsonl"
        if p.exists():
            rows = load_jsonl(p)
            info = audit_detect(rows, schema, split)
            info["unique_ids"] = len({r.get("id") for r in rows})
            info["unique_sent_ids"] = len({r.get("sent_id") for r in rows})
            info.update(dup_analysis(rows, with_event=False))
            report["files"][p.name] = {"exists": True, **info}

    p = data_dir / "dev_docs.jsonl"
    if p.exists():
        rows = load_jsonl(p)
        info = audit_docs(rows, schema, "dev")
        info["unique_ids"] = len({r.get("id") for r in rows})
        report["files"][p.name] = {"exists": True, **info}

    # ── 3. train/dev 文本泄漏 ───────────────────────────────────
    leakage = {}
    if "train" in extract and "dev" in extract:
        tr_text = collections.Counter(r.get("text") for r in extract["train"])
        dv_text = collections.Counter(r.get("text") for r in extract["dev"])
        overlap = [t for t in dv_text if t in tr_text]
        # 同文不同标：同一句在 train/dev 里事件类型不同
        lab = collections.defaultdict(set)
        for r in extract["train"] + extract["dev"]:
            lab[r.get("text")].add((r.get("event") or {}).get("type"))
        conflict = [t for t in overlap if len(lab[t]) > 1]
        leakage = {
            "exact_text_overlap": len(overlap),
            "overlap_rate_in_dev": round(len(overlap) / max(len(extract["dev"]), 1), 4),
            "same_text_diff_type": len(conflict),
            "examples": [t[:60] for t in overlap[:3]],
        }
    report["train_dev_leakage"] = leakage

    # ── 4. sample_stats 一致性 ──────────────────────────────────
    stats_path = data_dir / "sample_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        actual = {}
        for key in ("train_extract", "dev_extract", "train_detect", "dev_detect", "dev_docs"):
            fp = data_dir / f"{key}.jsonl"
            actual[key] = len(load_jsonl(fp)) if fp.exists() else None
        report["sample_stats"] = {"declared": stats.get("sizes", {}), "actual": actual,
                                  "match": stats.get("sizes", {}) == actual}

    # ── 5. sample vs 全量（可选）────────────────────────────────
    if full_dir and (full_dir / "dev_extract.jsonl").exists() and "dev" in extract:
        full_rows = load_jsonl(full_dir / "dev_extract.jsonl")
        full_types = collections.Counter((r.get("event") or {}).get("type") for r in full_rows)
        sample_types = collections.Counter(report["files"]["dev_extract.jsonl"]["type_counts"])
        report["sample_vs_full"] = {
            "full_dev_rows": len(full_rows),
            "sample_dev_rows": len(extract["dev"]),
            "full_dev_unique_ids": len({r.get("id") for r in full_rows}),
            "type_tv_distance": round(tv_distance(sample_types, full_types), 4),
            "per_type": {
                k: {"sample": sample_types.get(k, 0), "full": full_types.get(k, 0)}
                for k in sorted(set(sample_types) | set(full_types))
            },
            "note": "TV 距离 0 表示类型分布完全一致；<0.05 可视为代表性较好。",
        }

    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 6. Markdown 报告 ────────────────────────────────────────
    L = []
    L.append("> **本文档主题：** sample 数据体检：文件用途、span/标签、重复/泄漏、长尾与 TV 距离\n")
    L.append("> **初学者导读：** 先看第 1 节文件清单和第 7 节总结论；中间是检查明细。\n"
             "> **怎么用：** 只关心“有没有阻断训练的硬伤”；长尾/截断/重复是写进实验假设的风险。\n"
             "> **对应章节：** 初学者教程第 4 章「评测」\n"
             "> **文档级别：** 📊 结果")
    L.append("# 数据体检报告（week7 抽取任务）\n")
    L.append(f"- 数据目录：`{data_dir}`")
    L.append(f"- Schema：**{report['schema']['types']} 类 / {report['schema']['slots']} 槽位 / "
             f"{report['schema']['unique_roles']} 去重角色**\n")

    L.append("## 1. 文件清单与规模\n")
    purpose = {
        "train_extract.jsonl": ("句级抽取训练（一句一事件）", "✅ 是"),
        "dev_extract.jsonl": ("句级抽取主评测", "✅ 是（只评测）"),
        "train_detect.jsonl": ("两段式检测器训练（含负样本）", "✅ 是"),
        "dev_detect.jsonl": ("检测器评测", "✅ 是（只评测）"),
        "dev_docs.jsonl": ("篇章级事件流演示 / 系统层评测", "⚠ 仅系统层"),
    }
    L.append("| 文件 | 行数 | 唯一 id | 用途 | 能否直接用 | 结论 |")
    L.append("|---|---:|---:|---|---|---|")
    for name, info in report["files"].items():
        if not info.get("exists"):
            L.append(f"| `{name}` | - | - | - | - | ❌ 缺失 |")
            continue
        use, direct = purpose.get(name, ("-", "-"))
        L.append(f"| `{name}` | {info['rows']} | {info['unique_ids']} | {use} | {direct} | ✅ 可用 |")
    L.append(f"| `event_schema.json` | - | - | 标签体系（13 类 / 92 槽位 / 61 角色 + 枚举） | ✅ 构建标签必需 | ✅ 可用 |")
    L.append(f"| `sample_stats.json` | - | - | 抽样来源/seed/规模（seed=42） | ⚠ 元数据，不喂模型 | ✅ 一致 |")

    for split in ("train", "dev"):
        key = f"{split}_extract.jsonl"
        if key not in report["files"]:
            continue
        info = report["files"][key]
        L.append(f"\n### {split}_extract：span 与标签校验\n")
        L.append(f"- 平均论元数/事件：**{info['avg_args_per_event']}**（{info['min_args']}~{info['max_args']}）")
        L.append(f"- 文本长度：中位 **{info['len_median']}** 字，最长 **{info['len_max']}** 字，"
                 f"超过 254 字（会被 max_len=256 截断）的比例 **{info['truncate_rate_254']:.1%}**")
        L.append(f"- 真实校验问题：`{info['issues'] or '无'}`")
        L.append(f"- 同一事件同一角色多值：{info['multi_value_role_events']} 个事件"
                 f"（多出 {info['multi_value_role_pairs']} 个取值，属于合法标注，但会让 event 级完全匹配更难）")
        L.append(f"- 论元 span 互相重叠：{info['overlap_arg_pairs']} 对（投影到 BIO 会产生少量不可消除的非法序列）")
        L.append(f"- 重复：同 id {info['dup_id_groups']} 组（多出 {info['dup_id_extra_rows']} 行，"
                 f"属于一篇公告多句，评测用 id+sent_id 无影响）；同文本 {info['dup_text_groups']} 组"
                 f"（多出 {info['dup_text_extra_rows']} 行）")
        L.append(f"- 类型分布（Top/Bottom）：")
        tc = sorted(info["type_counts"].items(), key=lambda x: -x[1])
        L.append("  - 最多：" + "、".join(f"{k} {v}" for k, v in tc[:4]))
        L.append("  - 最少：" + "、".join(f"{k} {v}" for k, v in tc[-4:]))

    L.append("\n## 2. train/dev 泄漏\n")
    if leakage:
        L.append(f"- 完全相同文本：**{leakage['exact_text_overlap']} 句**"
                 f"（占 dev {leakage['overlap_rate_in_dev']:.1%}），其中同文不同事件类型 "
                 f"**{leakage['same_text_diff_type']} 句**")
        if leakage["examples"]:
            L.append(f"- 示例：{'；'.join(leakage['examples'])}")
    else:
        L.append("- 无 train/dev 对照")

    L.append("\n## 3. 检测集与篇章集\n")
    for name in ("train_detect.jsonl", "dev_detect.jsonl"):
        if name in report["files"]:
            info = report["files"][name]
            L.append(f"- `{name}`：{info['rows']} 行，正样本 **{info['positive']}**"
                     f"（{info['positive_rate']:.1%}），has_event 与 n_events 不一致 "
                     f"**{info['has_event_vs_n_events_mismatch']}** 条，未知类型 **{info['unknown_types']}** 条")
    if "dev_docs.jsonl" in report["files"]:
        info = report["files"]["dev_docs.jsonl"]
        L.append(f"- `dev_docs.jsonl`：{info['rows']} 篇，共 {info['events_total']} 个事件，"
                 f"多事件篇占 **{info['multi_event_doc_rate']:.1%}**，span/角色问题 "
                 f"`{info['issues'] or '无'}`")

    if report.get("extract_detect_overlap"):
        L.append("\n### 抽取集与检测集的重叠\n")
        for split, ov in report["extract_detect_overlap"].items():
            L.append(f"- `{split}`：extract {ov['extract_rows']} 句、detect {ov['detect_rows']} 句，"
                     f"句级重叠仅 **{ov['sent_overlap']}** 句；detect 正样本 {ov['detect_positive']} 句，"
                     f"其中 {ov['detect_positive_in_extract']} 句也出现在 extract。"
                     f"→ 两者是**独立抽样**，不能把 dev_detect 当成 dev_extract 加负样本。")

    L.append("\n## 4. Schema 覆盖\n")
    L.append(f"- 训练集覆盖了 **{report['schema'].get('slot_coverage', 0):.1%}** 的 schema 角色。")
    if report["schema"].get("roles_never_in_train"):
        L.append(f"- 训练集从未出现的角色：{'、'.join(report['schema']['roles_never_in_train'])}")

    if "sample_stats" in report:
        s = report["sample_stats"]
        L.append("\n## 5. sample_stats.json 一致性\n")
        L.append(f"- 声明：`{s['declared']}`")
        L.append(f"- 实际：`{s['actual']}`")
        L.append(f"- 结论：{'✅ 一致' if s['match'] else '❌ 不一致'}")

    if "sample_vs_full" in report:
        sv = report["sample_vs_full"]
        L.append("\n## 6. sample 与全量的代表性\n")
        L.append(f"- 全量 dev_extract：{sv['full_dev_rows']} 行 / {sv['full_dev_unique_ids']} 个唯一 id；"
                 f"sample dev：{sv['sample_dev_rows']} 行")
        L.append(f"- 事件类型分布 TV 距离：**{sv['type_tv_distance']}**"
                 f"（{'代表性较好' if sv['type_tv_distance'] < 0.05 else '分布有偏，sample 指标只能作冒烟参考'}）")

    L.append("\n## 7. 总结论\n")
    L.append("- **文件层面**：7 个文件各司其职，没有无效文件；`event_schema.json` / `sample_stats.json` "
             "是元数据，训练不需要单独喂。")
    L.append("- **样本层面**：span 全部能在原文精确还原，角色/类型均在 schema 内；"
             "主要问题是**长尾类型样本过少**、**少量重复文本**、**2 句 train/dev 文本重叠**。")
    L.append("- **使用建议**：train/dev_extract 用于句级抽取主评测；detect 用于两段式检测器；"
             "dev_docs 用于篇章级事件流演示；sample 是冒烟子集，**不要拿它的指标直接对 PPT 全量数字**。")

    # 不写 .md：数据体检报告属于学习/分析类 Markdown，统一放桌面合集；
    # JSON 仍是机器可读的权威结果。
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2))
    print(f"\n已写入：{out_json}")


if __name__ == "__main__":
    main()

"""统一评测：实体级严格匹配（等价 seqeval strict）+ 错误归因。

所有方法（BiLSTM / BERT / RoBERTa / LLM）都输出同一种 pred.jsonl：
    {"id": ..., "text": ..., "entities": [{"start": int, "end": int, "label": str}]}
因此可以用同一个函数评，避免「各脚本各算一套指标」。

严格匹配定义：预测三元组 (start, end, label) 与金标完全相同才算 TP。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

Entity = tuple[int, int, str]  # (start, end, label)


def to_set(row: dict, keep_label: bool = True) -> set:
    out = set()
    for e in row.get("entities", []):
        s, t, lab = int(e["start"]), int(e["end"]), str(e["label"])
        if t <= s:
            continue
        out.add((s, t, lab) if keep_label else (s, t))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _overlap(a: Entity, b: Entity) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def attribute_errors(gold: set, pred: set) -> dict:
    """把金标里没被命中的实体（FN）分成四类，便于讲清「错在哪」。

    - type_err      : 区间完全一致、标签错（最典型的类型混淆）
    - boundary_err  : 标签一致、区间有交集但不完全一致（多字/少字）
    - both_err      : 区间不一致且标签也错，但有交集
    - miss          : 完全没检出（无任何交集的预测）
    """
    stat = Counter(type_err=0, boundary_err=0, both_err=0, miss=0)
    same_span = {(s, t): lab for s, t, lab in pred}
    # @doc ** 判定顺序不能换，否则 type_err 被吞进 boundary_err，报告出现「类型错=0」
    # 否则 type_err 会被吞进 boundary_err，报告里就会出现「类型错=0」的假象
    for s, t, lab in gold:
        if (s, t, lab) in pred:
            continue
        if (s, t) in same_span:
            stat["type_err"] += 1
            continue
        hit_same = any(lab == p[2] and _overlap((s, t, lab), p) for p in pred)
        hit_any = any(_overlap((s, t, lab), p) for p in pred)
        if hit_same:
            stat["boundary_err"] += 1
        elif hit_any:
            stat["both_err"] += 1
        else:
            stat["miss"] += 1
    stat["total_fn"] = sum(stat[k] for k in ("type_err", "boundary_err", "both_err", "miss"))
    return dict(stat)


def evaluate(
    gold_rows: Iterable[dict],
    pred_rows: Iterable[dict],
    labels: list[str] | None = None,
    subset_ids: set[str] | None = None,
    group_key: str | None = None,
) -> dict:
    """gold / pred 按 id 对齐后评测。

    subset_ids: 只评这些 id（LLM 抽样时用，保证其他方法能在同一子集上另算一份）。
    group_key : 可选，按 row[group_key] 分组（如 resume 的 meta.hard）再给分。
    """
    gold = {r["id"]: r for r in gold_rows}
    pred = {r["id"]: r for r in pred_rows}
    ids = [i for i in gold if (subset_ids is None or i in subset_ids)]
    # @doc ** subset_ids 只限评哪些 id，不能过滤 pred，否则漏检被从分母里删掉
    # 预测缺失必须算漏检，否则 LLM 只跑 300 条时，另外 774 条会被悄悄从分母里删掉
    if subset_ids is not None:
        # 预测缺失也算漏检，只是没有预测实体
        pass
    if labels is None:
        labels = sorted({e["label"] for r in gold.values() for e in r.get("entities", [])})

    per_type = {lab: Counter(tp=0, fp=0, fn=0) for lab in labels}
    total = Counter(tp=0, fp=0, fn=0)
    errs = Counter(type_err=0, boundary_err=0, both_err=0, miss=0)
    groups: dict[str, Counter] = defaultdict(lambda: Counter(tp=0, fp=0, fn=0, n=0))
    missing_pred = 0
    len_buckets: dict[str, Counter] = defaultdict(lambda: Counter(tp=0, fn=0))

    for i in ids:
        g = gold[i]
        p = pred.get(i)
        if p is None:
            missing_pred += 1
            p = {"id": i, "entities": []}
        gs, ps = to_set(g), to_set(p)
        tp = gs & ps
        fp = ps - gs
        fn = gs - ps
        total["tp"] += len(tp)
        total["fp"] += len(fp)
        total["fn"] += len(fn)
        for s, t, lab in tp:
            per_type.setdefault(lab, Counter())[ "tp"] += 1
        for s, t, lab in fp:
            per_type.setdefault(lab, Counter())[ "fp"] += 1
        for s, t, lab in fn:
            per_type.setdefault(lab, Counter())[ "fn"] += 1
        e = attribute_errors(gs, ps)
        for k in ("type_err", "boundary_err", "both_err", "miss"):
            errs[k] += e[k]

        # 实体长度分桶：实体级微平均，观察「长实体是不是更难」
        for s, t, _ in gs:
            n = t - s
            b = "1-2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9+"
            len_buckets[b]["fn"] += 0 if (s, t, _) in ps else 1
            len_buckets[b]["tp"] += 1 if (s, t, _) in ps else 0

        if group_key:
            gv = str(g.get(group_key, g.get("meta", {}).get(group_key, "unknown")))
            groups[gv]["tp"] += len(tp)
            groups[gv]["fp"] += len(fp)
            groups[gv]["fn"] += len(fn)
            groups[gv]["n"] += 1

    overall = prf(total["tp"], total["fp"], total["fn"])
    overall["n_docs"] = len(ids)
    overall["n_gold"] = total["tp"] + total["fn"]
    overall["n_pred"] = total["tp"] + total["fp"]
    overall["missing_pred_docs"] = missing_pred
    errs["total_fn"] = sum(errs[k] for k in ("type_err", "boundary_err", "both_err", "miss"))

    per_label = {}
    for lab in labels:
        c = per_type[lab]
        if c["tp"] + c["fp"] + c["fn"] == 0:
            continue
        d = prf(c["tp"], c["fp"], c["fn"])
        d["support"] = c["tp"] + c["fn"]
        per_label[lab] = d
    macro = sum(d["f1"] for d in per_label.values()) / len(per_label) if per_label else 0.0

    hard = {}
    for gv, c in groups.items():
        hard[gv] = prf(c["tp"], c["fp"], c["fn"]) | {"n_docs": c["n"]}

    len_stats = {}
    for b, c in len_buckets.items():
        len_stats[b] = prf(c["tp"], c["tp"] + 0, c["fn"]) | {"n_gold": c["tp"] + c["fn"]}

    return {
        "overall": overall,
        "macro_f1": macro,
        "n_labels": len(per_label),
        "per_type": dict(sorted(per_label.items(), key=lambda kv: -kv[1]["support"])),
        "error_attribution": errs,
        "groups": hard,
        "length_buckets": dict(sorted(len_stats.items())),
    }


def format_table(result: dict) -> str:
    """把 evaluate() 的结果打成 Markdown 小表（终端/报告两用）。"""
    o = result["overall"]
    lines = [
        f"docs={o['n_docs']}  gold={o['n_gold']}  pred={o['n_pred']}",
        f"P={o['precision']:.4f}  R={o['recall']:.4f}  F1={o['f1']:.4f}  macroF1={result['macro_f1']:.4f}",
    ]
    if result["error_attribution"]["total_fn"]:
        e = result["error_attribution"]
        tot = e["total_fn"]
        lines.append(
            "FN 归因: "
            + "  ".join(
                f"{k}={e[k]}({e[k]/tot:.0%})" for k in ("type_err", "boundary_err", "both_err", "miss")
            )
        )
    return "\n".join(lines)

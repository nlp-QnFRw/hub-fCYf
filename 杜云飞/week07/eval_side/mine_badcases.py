# -*- coding: utf-8 -*-
"""mine_badcases.py — 通用 bad case 挖掘（A 组判别式 / B 组本地 LLM / C 组 API 同一套口径）

用法：
  python eval_siide/mine_badcases.py --pred outputs/disc/pred_bert12L_crf.jsonl

口径与 evaluate() 完全一致：按「句」聚合（键 = uid or id），trigger 按**文本**相等判定
（不是字符偏移——评测器本身就是文本口径），论元按 (role, text)。
错误分类：类型没覆盖 / 触发词边界错（互为子串）/ 触发词完全没命中 / 论元缺失 / 论元多余。
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from evaluate_events import load_jsonl, merge_rows_by_sample  # noqa: E402


def key_of(r):
    return str(r.get("uid") or r.get("id") or "")


def trig_text(e):
    t = e.get("trigger")
    if isinstance(t, dict):
        return str(t.get("text", ""))
    return str(t or "")


def argpairs(e):
    out = []
    for a in e.get("args") or []:
        if isinstance(a, dict):
            out.append((a.get("role"), a.get("text")))
        elif isinstance(a, (list, tuple)) and len(a) >= 2:
            out.append((a[0], a[1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default=str(ROOT / "data" / "processed" / "dev_n200.jsonl"))
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    text_by_key = {key_of(r): r["text"] for r in gold_rows}
    gold = {}
    for r in gold_rows:
        ev = r["event"]
        gold.setdefault(key_of(r), []).append(
            {"type": ev["type"], "trigger": ev["trigger"]["text"],
             "args": [(a["role"], a["text"]) for a in ev.get("args") or []]})

    pred = {key_of(p): (p.get("events") or []) for p in merge_rows_by_sample(load_jsonl(args.pred))}
    pred = {k: v for k, v in pred.items() if k in gold}

    n_gold_ev = sum(len(v) for v in gold.values())
    n_pred_ev = sum(len(v) for v in pred.values())
    hit_type = hit_exact = 0
    miss_trig, boundary, cross_type, arg_miss, arg_extra = Counter(), [], [], [], []

    for k, gevs in gold.items():
        text = text_by_key.get(k, "")
        pevs = pred.get(k, [])
        for g in gevs:
            same_type = [p for p in pevs if p.get("type") == g["type"]]
            exact = [p for p in same_type if trig_text(p) == g["trigger"]]
            if same_type:
                hit_type += 1
            if exact:
                hit_exact += 1
                pa = argpairs(exact[0])
                gt = set(g["args"])
                miss = [a for a in g["args"] if a not in set(pa)]
                extra = [a for a in pa if a not in gt]
                if miss:
                    arg_miss.append((text, g, exact[0], miss))
                if extra:
                    arg_extra.append((text, g, exact[0], extra))
            else:
                miss_trig[g["trigger"]] += 1
                near = [p for p in same_type
                        if trig_text(p) and (trig_text(p) in g["trigger"] or g["trigger"] in trig_text(p))]
                if near:
                    boundary.append((text, g, near[0]))
                else:
                    cross_type.append((text, g, (same_type or pevs or [None])[0]))

    total = max(n_gold_ev, 1)
    print(f"# {Path(args.pred).name}")
    print(f"gold 事件 {n_gold_ev} / pred 事件 {n_pred_ev}（句级对齐，{len(gold)} 句）")
    print(f"类型覆盖 {hit_type} ({hit_type/total:.1%}) | 触发词文本精确命中 {hit_exact} ({hit_exact/total:.1%})")
    print(f"漏检 {n_gold_ev - hit_exact}：边界错 {len(boundary)} / 类型错-漏 {len(cross_type)}")
    print(f"论元有缺失的事件 {len(arg_miss)} | 论元有多余的事件 {len(arg_extra)}")
    print("漏检最多的触发词:", "、".join(f"{w}x{c}" for w, c in miss_trig.most_common(8)))
    print()

    for tag, bucket in (("边界错", boundary), ("类型错", cross_type), ("论元缺失", arg_miss)):
        for item in bucket[:args.examples]:
            text, g, p = item[0], item[1], item[2]
            print(f"[{tag}] {text[:60]}")
            print(f"  gold type={g['type']} trigger={g['trigger']} args={g['args']}")
            if p is None:
                print("  pred （整句无预测事件）")
            else:
                print(f"  pred type={p.get('type')} trigger={trig_text(p)} args={argpairs(p)}")
            if len(item) > 3:
                print(f"  缺失: {item[3]}")
            print()


if __name__ == "__main__":
    main()

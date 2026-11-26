"""L0 规则基线：从 train 自动挖触发词词典 + 通用数值正则，零训练、零 GPU。

不是手抄词典：词典完全由各变体的 train 生成，所以 heldout 变体天然学不到被拿掉的类型。
论点元 regex 只认「时间 / 金额 / 比例 / 数量」四类强模式，角色名从 schema 里按关键字挑，
刻意不碰公司名这类需要语义的对象——规则法的上限正是这里。
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_ROOT, WEEK7_DATA  # noqa: E402
from common.bridge import load_jsonl, load_schema, make_pred_row  # noqa: E402
from common.eval_local import run_eval  # noqa: E402

NUM = r"[0-9０-９][0-9０-９,，]*\.?[0-9０-９]*"
PATTERNS = [
    ("date", re.compile(r"\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d{1,2}日|\d{1,2}:\d{2}"),
     ("时间", "日期", "期限", "日")),
    ("money", re.compile(rf"{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|港币)|{NUM}\s*亿"),
     ("金额", "价格", "资金", "数额", "报酬", "融资额")),
    ("percent", re.compile(rf"{NUM}\s*(?:%|％|百分点)"), ("比例", "占比", "幅度", "涨幅", "跌幅")),
    ("share", re.compile(rf"{NUM}\s*(?:万股|亿股|股)"), ("数量", "股数", "股份数")),
]


def data_dir(variant: str) -> Path:
    return WEEK7_DATA if variant == "sent" else DATA_ROOT / variant


def train_dict(rows: list[dict]) -> dict:
    counts = defaultdict(Counter)
    for r in rows:
        ev = r.get("event")
        if not ev:
            continue
        trig = (ev.get("trigger") or {}).get("text", "")
        if trig:
            counts[trig][ev.get("type", "")] += 1
    return {t: c.most_common(1)[0][0] for t, c in counts.items()}


def pick_role(roles: list[str], keys: tuple) -> str:
    for k in keys:
        for role in roles:
            if k in role:
                return role
    return ""


def predict(text: str, trig2type: dict, schema: dict) -> list[dict]:
    # 长触发词优先，避免「质押」盖住「解除质押」
    hit = None
    for t in sorted(trig2type, key=len, reverse=True):
        pos = text.find(t)
        if pos >= 0:
            hit = (t, pos, trig2type[t])
            break
    if not hit:
        return []
    trig, pos, etype = hit
    roles = schema.get(etype, [])
    args = []
    for _, pattern, keys in PATTERNS:
        role = pick_role(roles, keys)
        if not role:
            continue
        m = pattern.search(text)
        if m:
            args.append({"role": role, "text": m.group(), "start": m.start(), "end": m.end()})
    # 触发词若恰好等于某个 span，丢掉这个论元（week7 的 span 冲突口径）
    args = [a for a in args if not (a["start"] == pos and a["end"] == pos + len(trig))]
    return [{"type": etype, "trigger": {"text": trig, "start": pos, "end": pos + len(trig)},
             "args": args}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sent", "heldout", "doc"], default="sent")
    args = ap.parse_args()
    ddir = data_dir(args.variant)
    schema = load_schema(ddir / "event_schema.json")
    train = load_jsonl(ddir / "train_extract.jsonl") if (ddir / "train_extract.jsonl").exists() \
        else load_jsonl(WEEK7_DATA / "train_extract.jsonl")
    dev = load_jsonl(ddir / "dev_extract.jsonl")
    trig2type = train_dict(train)
    preds = []
    for r in dev:
        preds.append(make_pred_row(r, predict(r["text"], trig2type, schema)))
    name = f"rule_{args.variant}"
    res = run_eval(name, preds, ddir / "dev_extract.jsonl",
                   meta={"trigger_dict_size": len(trig2type), "train_rows": len(train),
                         "decoder": "dictionary+regex", "train_s": 0},
                   doc_index=(ddir / "doc_index.json") if args.variant == "doc" else None)
    print(f"{name}: argument F1={res['argument']['f1']:.4f} event F1={res['event_exact']['f1']:.4f}")


if __name__ == "__main__":
    main()

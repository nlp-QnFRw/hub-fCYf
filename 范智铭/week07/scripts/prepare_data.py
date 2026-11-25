"""把三套原始数据统一成 JSONL（字符 offset + 标签），并做完整性校验。

统一格式（每个数据集目录下 train/dev/test.jsonl）：
    {"id": "...", "text": "...", "entities": [{"start": 0, "end": 2, "label": "name"}]}

- resume : src.gen_resume 生成（已就绪，这里只做校验与统计）
- cluener: CLUENER2020 官方 jsonl，label 是 {类型: {实体串: [[s, e], ...]}} 的嵌套 dict
- msra   : BIO 文本，每行「字\\t标签」，空行分句

用法：
    python scripts/prepare_data.py --datasets resume cluener msra
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DATA_PROCESSED, DATA_RAW, get_logger, read_jsonl, write_json, write_jsonl  # noqa: E402

log = get_logger("prepare")


# ---------------------------------------------------------------- CLUENER
def convert_cluener(n_train: int = 8600, n_dev: int = 1074, seed: int = 42) -> None:
    """CLUENER 官方 test.json 是盲测集（没有 label 字段），所以从官方 train 里重新切分。

    train 10748 条 → train 8600 / dev 1074 / test 1074（seed=42 固定，可复现）。
    官方 dev 1343 条不使用，避免与我们的 dev 口径混淆。
    """
    out = DATA_PROCESSED / "cluener"
    label_counter: Counter[str] = Counter()
    dropped = 0
    pool = []
    with open(DATA_RAW / "cluener" / "train.json", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = __import__("json").loads(line)
            text = raw["text"]
            ents = []
            for lab, mentions in (raw.get("label") or {}).items():
                for mention, spans in mentions.items():
                    for s, e in spans:
                        # @doc *** CLUENER 的 end 是闭区间（[9,11]=叶老桂），必须 +1
                        # 直接当左闭右开会得到「叶老」；+1 之后才能通过 text[s:e]==mention 断言
                        s, e = int(s), int(e) + 1
                        if text[s:e] != mention:
                            raise ValueError(f"cluener offset 不一致: {i} {mention} -> {text[s:e]!r}")
                        ents.append({"start": s, "end": e, "label": lab})
            # 嵌套实体（如《…》书名里套公司名）BIO 表达不了：保留外层、丢弃被包含的
            ents.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
            kept = []
            for e in ents:
                if any(e["start"] < k["end"] and k["start"] < e["end"] for k in kept):
                    dropped += 1
                    continue
                kept.append(e)
            ents = sorted(kept, key=lambda x: x["start"])
            pool.append({"text": text, "entities": ents})
    random.Random(seed).shuffle(pool)
    splits = {
        "train": pool[:n_train],
        "dev": pool[n_train:n_train + n_dev],
        "test": pool[n_train + n_dev:n_train + 2 * n_dev],
    }
    stats = {}
    for split, data in splits.items():
        rows, n_ent = [], 0
        for i, r in enumerate(data):
            for e in r["entities"]:
                label_counter[e["label"]] += 1
            rows.append({"id": f"cluener-{split}-{i:06d}", "text": r["text"], "entities": r["entities"]})
            n_ent += len(r["entities"])
        write_jsonl(out / f"{split}.jsonl", rows)
        stats[split] = {"n": len(rows), "entities": n_ent, "chars": sum(len(r["text"]) for r in rows)}
        log.info("cluener %-5s %5d 条  %6d 实体  平均 %.1f 字", split, len(rows), n_ent,
                 stats[split]["chars"] / max(1, len(rows)))
    labels = sorted(label_counter)
    write_json(out / "labels.json", {"labels": labels, "counts": dict(label_counter),
                                     "dropped_nested": dropped})
    write_json(out / "stats.json", stats)


# ---------------------------------------------------------------- MSRA
MSRA_MAP = {"PER": "PER", "LOC": "LOC", "ORG": "ORG"}


def read_msra_bio(path: Path) -> list[tuple[str, list[dict]]]:
    """解析 BIO 文本 → [(sentence, entities)]，顺手修非法 I- 开头。"""
    sents: list[tuple[str, list[dict]]] = []
    chars: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        if not chars:
            return
        text = "".join(chars)
        ents, cur = [], None
        for i, t in enumerate(tags):
            if t == "O" or not t:
                if cur:
                    ents.append(cur)
                    cur = None
                continue
            prefix, _, typ = t.partition("-")
            typ = MSRA_MAP.get(typ, typ)
            if prefix == "B" or cur is None or cur["label"] != typ:
                if cur:
                    ents.append(cur)
                cur = {"start": i, "end": i + 1, "label": typ}
            else:
                cur["end"] = i + 1
        if cur:
            ents.append(cur)
        if ents:
            sents.append((text, ents))
        chars.clear()
        tags.clear()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0]:
                continue
            chars.append(parts[0])
            tags.append(parts[1].strip())
    flush()
    return sents


def convert_msra(n_train: int, n_dev: int, n_test: int, seed: int = 42) -> None:
    out = DATA_PROCESSED / "msra"
    train_all = read_msra_bio(DATA_RAW / "msra_train_bio.txt")
    test_all = read_msra_bio(DATA_RAW / "msra_test_bio.txt")
    rng = random.Random(seed)
    rng.shuffle(train_all)
    train, dev = train_all[:n_train], train_all[n_train:n_train + n_dev]
    rng.shuffle(test_all)
    test = test_all[:n_test]

    label_counter: Counter[str] = Counter()
    stats = {}
    for split, data in (("train", train), ("dev", dev), ("test", test)):
        rows = []
        for i, (text, ents) in enumerate(data):
            for e in ents:
                label_counter[e["label"]] += 1
            rows.append({"id": f"msra-{split}-{i:06d}", "text": text, "entities": ents})
        write_jsonl(out / f"{split}.jsonl", rows)
        stats[split] = {"n": len(rows), "entities": sum(len(r["entities"]) for r in rows),
                        "chars": sum(len(r["text"]) for r in rows)}
        log.info("msra %-5s %5d 条  %6d 实体  平均 %.1f 字", split, len(rows), stats[split]["entities"],
                 stats[split]["chars"] / max(1, len(rows)))
    write_json(out / "labels.json", {"labels": sorted(label_counter), "counts": dict(label_counter)})
    write_json(out / "stats.json", stats)


# ---------------------------------------------------------------- 校验
def validate(name: str) -> dict:
    """四道检查：offset 可切片、区间合法且不重叠、无空实体、split 统计。"""
    out = DATA_PROCESSED / name
    report = {"dataset": name, "splits": {}, "problems": []}
    entity_strings = {}
    for split in ("train", "dev", "test"):
        rows = read_jsonl(out / f"{split}.jsonl")
        n_ent, texts = 0, set()
        for r in rows:
            text = r["text"]
            if text in texts:
                report["problems"].append(f"{split}: 重复文本 {r['id']}")
            texts.add(text)
            last_end = -1
            for e in r["entities"]:
                s, t, lab = e["start"], e["end"], e["label"]
                if not (0 <= s < t <= len(text)):
                    report["problems"].append(f"{split}/{r['id']}: 越界 {e}")
                    continue
                if s < last_end:
                    report["problems"].append(f"{split}/{r['id']}: 实体重叠 {e}")
                last_end = t
                n_ent += 1
            entity_strings.setdefault(split, set()).update(text[e["start"]:e["end"]] for e in r["entities"])
        report["splits"][split] = {"n": len(rows), "entities": n_ent,
                                   "chars": sum(len(r["text"]) for r in rows),
                                   "unique_entity_strings": len(entity_strings[split])}
    # 跨 split 的实体串泄漏（只报告，不阻断；这是数据质量指标）
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        inter = entity_strings[a] & entity_strings[b]
        report["splits"][f"{a}_{b}_overlap"] = len(inter)
    write_json(out / "validation.json", report)
    status = "OK" if not report["problems"] else f"{len(report['problems'])} 个问题"
    log.info("%-8s 校验 %s", name, status)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["resume", "cluener", "msra"])
    ap.add_argument("--msra_train", type=int, default=10000)
    ap.add_argument("--msra_dev", type=int, default=1000)
    ap.add_argument("--msra_test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "cluener" in args.datasets:
        convert_cluener()
    if "msra" in args.datasets:
        convert_msra(args.msra_train, args.msra_dev, args.msra_test, args.seed)
    for name in args.datasets:
        validate(name)


if __name__ == "__main__":
    main()

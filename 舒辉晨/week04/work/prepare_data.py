"""
prepare_data.py -- 语料清洗 → 去重 → 划分 → 建词表 → 编码成 .bin

用法:
    python prepare_data.py                          # 用 data/raw/ 下的所有 .txt
    python prepare_data.py --input ../循环神经网络语言模型/corpus_general.txt
    python prepare_data.py --max-vocab 6000 --val-ratio 0.05 --near-dup 0

这一步解决旧版本里最影响结论的几个问题：
    1. 语料没清洗：混进「欢迎发表评论我要评论」这类爬虫模板行；
    2. 重复行占 37.7%，打乱后切分 → 验证集 60% 的句子训练集里原样见过，PPL 虚低；
    3. 字表外的字直接丢弃，制造了语料里不存在的伪相邻二元组。

所以顺序是：先清洗、先去重，再划分，最后只用训练集建词表
（如果拿全量语料建词表，验证集的信息会从词表泄露进训练）。
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import time

import numpy as np

UNK = "[UNK]"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# 爬虫模板 / 广告 / 占位符：出现在语料里只会教模型背废话
JUNK_MARKERS = (
    "我要评论", "发表评论", "COUNT", "本测试仅供", "http://", "https://", "www.",
    "点击查看", "更多精彩", "转载请注明", "扫描二维码", "关注公众号",
)

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ─────────────────────────── 读取与清洗 ───────────────────────────

def read_lines(paths: list[str]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到输入文件: {path}")
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines.extend(f.read().splitlines())
    return lines


def normalize(line: str) -> str:
    line = line.replace("\u3000", " ").replace("\ufeff", "")
    line = _CTRL.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def is_junk(line: str, min_len: int, max_len: int, min_unique_ratio: float) -> str | None:
    """返回判定为垃圾的原因，正常行返回 None。"""
    if len(line) < min_len:
        return "too_short"
    if len(line) > max_len:
        return "too_long"
    if any(m in line for m in JUNK_MARKERS):
        return "template"
    if len(set(line)) / len(line) < min_unique_ratio:
        return "repetitive"      # 「哈哈哈哈哈哈哈」这类
    if not any("\u4e00" <= c <= "\u9fa5" for c in line):
        return "no_chinese"
    return None


def dedupe_key(line: str) -> str:
    """去重用的归一化键：抹掉标点、空格、大小写差异。

    「谁有范冰冰的苹果这部电影」和「谁有范冰冰的苹果这部电影？」应当算同一行，
    但它们对语言模型贡献的是同一句话，留着只会让验证集泄漏更严重。
    """
    return "".join(c for c in line.lower() if c.isalnum() or "\u4e00" <= c <= "\u9fa5")


# ─────────────────────────── 近重复（MinHash + LSH） ───────────────────────────

_P = (1 << 61) - 1


def _shingle_hashes(text: str, n: int) -> list[int]:
    grams = {text} if len(text) <= n else {text[i: i + n] for i in range(len(text) - n + 1)}
    return [int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big") % _P
            for g in grams]


def _shingle_set(text: str, n: int) -> set[str]:
    return {text} if len(text) <= n else {text[i: i + n] for i in range(len(text) - n + 1)}


def _perm_params(k: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(1, _P), rng.randrange(_P)) for _ in range(k)]


def _signature(hs: list[int], perms: list[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(min((a * h + b) % _P for h in hs) for a, b in perms)


def dedupe_near(lines: list[str], threshold: float, shingle: int = 3,
                perms: int = 12, bands: int = 3, seed: int = 42) -> list[str]:
    """MinHash + LSH 去掉「只差一两个字」的近似重复。

    思路：把每行切成 n 元组字符片段，用 k 个哈希函数压成 k 维签名；
    签名分 band 建索引，同桶的才可能是近重复，再算真实 Jaccard 确认。
    这样避免了 O(n²) 的两两比较。
    """
    if not (0 < threshold <= 1):
        raise ValueError("--near-dup 需要在 (0, 1] 之间，传 0 表示关闭")
    rows = max(1, perms // bands)
    params = _perm_params(perms, seed)
    kept: list[str] = []
    kept_sets: list[set[str]] = []
    buckets: dict[tuple, list[int]] = collections.defaultdict(list)

    for line in lines:
        hs = _shingle_hashes(line, shingle)
        sig = _signature(hs, params)
        candidates: set[int] = set()
        for b in range(bands):
            key = (b, sig[b * rows: (b + 1) * rows])
            candidates.update(buckets[key][-48:])

        cur = _shingle_set(line, shingle)
        dup = False
        for ci in candidates:
            other = kept_sets[ci]
            inter = len(cur & other)
            if inter and inter / (len(cur) + len(other) - inter) >= threshold:
                dup = True
                break
        if dup:
            continue

        idx = len(kept)
        kept.append(line)
        kept_sets.append(cur)
        for b in range(bands):
            buckets[(b, sig[b * rows: (b + 1) * rows])].append(idx)
    return kept


# ─────────────────────────── 词表 ───────────────────────────

def build_vocab(text: str, max_vocab: int, min_count: int) -> list[str]:
    """只从训练集统计。索引 0 固定留给 [UNK]。"""
    counter = collections.Counter(text)
    budget = max(1, max_vocab - 1)
    chars = [c for c, n in counter.most_common() if n >= min_count][:budget]
    return [UNK] + chars


def coverage(text: str, stoi: dict[str, int]) -> float:
    hit = sum(1 for c in text if c in stoi)
    return hit / max(1, len(text))


# ─────────────────────────── N-gram 基线 ───────────────────────────

def ngram_ppl(train_ids: np.ndarray, val_ids: np.ndarray, n: int, vocab_size: int,
              alpha: float = 0.5) -> float:
    """加 α 平滑的字符级 n-gram 困惑度，作为 Transformer 的对照基线。"""
    def grams(arr):
        if n == 1:
            return [(int(x),) for x in arr]
        return [tuple(int(v) for v in arr[i: i + n]) for i in range(len(arr) - n + 1)]

    train_grams = grams(train_ids)
    counts = collections.Counter(train_grams)
    if n > 1:
        ctx_counts = collections.Counter(g[:-1] for g in train_grams)
    else:
        ctx_counts = None
    total_train = max(1, len(train_ids))

    logp, num = 0.0, 0
    for g in grams(val_ids):
        c_all = counts.get(g, 0)
        c_ctx = ctx_counts.get(g[:-1], 0) if n > 1 else total_train
        p = (c_all + alpha) / (c_ctx + alpha * vocab_size)
        logp += np.log(p)
        num += 1
    return float(np.exp(-logp / max(1, num)))


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description="清洗语料、划分训练/验证集并编码")
    ap.add_argument("--input", nargs="*", default=None,
                    help="输入 txt 路径，可多个；不传则用 data/raw/*.txt")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--max-vocab", type=int, default=4000, help="字表上限（含 [UNK]）")
    ap.add_argument("--min-count", type=int, default=3, help="字频低于此值不进表")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=120)
    ap.add_argument("--min-unique-ratio", type=float, default=0.3)
    ap.add_argument("--near-dup", type=float, default=0.85,
                    help="近重复 Jaccard 阈值，0 表示只用精确去重")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    paths = args.input or sorted(glob.glob(os.path.join(data_dir, "raw", "*.txt")))
    if not paths:
        raise SystemExit(f"没有找到语料：{os.path.join(data_dir, 'raw', '*.txt')}")

    t0 = time.time()
    raw = read_lines(paths)
    print(f"输入 {len(paths)} 个文件，{len(raw):,} 行")

    # 1) 清洗
    reason = collections.Counter()
    cleaned: list[str] = []
    for line in raw:
        line = normalize(line)
        why = "empty" if not line else is_junk(line, args.min_len, args.max_len,
                                               args.min_unique_ratio)
        if why:
            reason[why] += 1
            continue
        cleaned.append(line)
    print(f"清洗后 {len(cleaned):,} 行  " +
          "  ".join(f"{k}={v:,}" for k, v in reason.most_common()))

    # 2) 精确去重
    seen, exact = set(), []
    for line in cleaned:
        key = dedupe_key(line)
        if not key or key in seen:
            continue
        seen.add(key)
        exact.append(line)
    print(f"精确去重 {len(cleaned):,} -> {len(exact):,}（去掉 {len(cleaned) - len(exact):,}，"
          f"{100 * (1 - len(exact) / max(1, len(cleaned))):.1f}%）")

    # 3) 近重复去重
    if args.near_dup > 0:
        t1 = time.time()
        deduped = dedupe_near(exact, args.near_dup, seed=args.seed)
        print(f"近重复去重 {len(exact):,} -> {len(deduped):,}（去掉 {len(exact) - len(deduped):,}，"
              f"{time.time() - t1:.1f}s）")
    else:
        deduped = exact

    # 4) 划分（去重之后才打乱，保证验证集不会撞上训练集原句）
    rng = random.Random(args.seed)
    rng.shuffle(deduped)
    n_val = max(1, int(len(deduped) * args.val_ratio))
    val_lines = deduped[:n_val]
    train_lines = deduped[n_val:]
    overlap = len({dedupe_key(x) for x in val_lines} & {dedupe_key(x) for x in train_lines})
    print(f"划分: 训练 {len(train_lines):,} 行 / 验证 {len(val_lines):,} 行，"
          f"验证集与训练集重叠 {overlap} 行")
    if overlap:
        raise SystemExit("划分后仍有重叠，去重逻辑有 bug")

    train_text = "\n".join(train_lines)
    val_text = "\n".join(val_lines)

    # 5) 词表只从训练集建
    itos = build_vocab(train_text, args.max_vocab, args.min_count)
    stoi = {c: i for i, c in enumerate(itos)}
    print(f"字表 {len(itos)} 项（含 [UNK]），训练集覆盖率 {100 * coverage(train_text, stoi):.2f}%"
          f" / 验证集覆盖率 {100 * coverage(val_text, stoi):.2f}%")

    # 6) 编码落盘
    train_ids = np.fromiter((stoi.get(c, 0) for c in train_text), dtype=np.uint16,
                            count=len(train_text))
    val_ids = np.fromiter((stoi.get(c, 0) for c in val_text), dtype=np.uint16,
                          count=len(val_text))
    os.makedirs(data_dir, exist_ok=True)
    train_ids.tofile(os.path.join(data_dir, "train.bin"))
    val_ids.tofile(os.path.join(data_dir, "val.bin"))
    with open(os.path.join(data_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(itos, f, ensure_ascii=False)
    for name, lines in (("train.txt", train_lines), ("val.txt", val_lines)):
        with open(os.path.join(data_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # 7) N-gram 基线（同一份划分、同一个字表口径）
    print("计算 N-gram 基线 ...")
    baseline = {str(n): round(ngram_ppl(train_ids, val_ids, n, len(itos)), 2)
                for n in (1, 2, 3)}
    print("  " + "  ".join(f"{n}-gram PPL={v}" for n, v in baseline.items()))

    newline_ratio = 100 * float((train_ids == stoi.get("\n", -1)).mean())
    meta = {
        "vocab_size": len(itos),
        "max_vocab": args.max_vocab,
        "min_count": args.min_count,
        "seed": args.seed,
        "sources": [os.path.basename(p) for p in paths],
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "stats": {
            "raw_lines": len(raw),
            "cleaned_lines": len(cleaned),
            "after_exact_dedupe": len(exact),
            "after_near_dedupe": len(deduped),
            "train_lines": len(train_lines),
            "val_lines": len(val_lines),
            "val_train_overlap": overlap,
            "train_coverage": round(100 * coverage(train_text, stoi), 2),
            "val_coverage": round(100 * coverage(val_text, stoi), 2),
            "newline_ratio_pct": round(newline_ratio, 2),
            "dropped": dict(reason),
        },
        "baseline_ppl": baseline,
        "near_dup_threshold": args.near_dup,
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"已写出 {data_dir}：train.bin({train_ids.size:,}) / val.bin({val_ids.size:,})"
          f" / vocab.json / meta.json   用时 {time.time() - t0:.1f}s")
    print(f"换行符占训练 token 的 {newline_ratio:.2f}%"
          f"（行间分隔需要它；如果只想做输入法可以把它从词表里去掉）")


if __name__ == "__main__":
    main()

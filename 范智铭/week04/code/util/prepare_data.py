"""数据准备脚本：终止符 -> 训练/测试切分 -> 词表构建。

对 ``data/corpus_general.txt``（每行一条独立文本）执行三件事：

1. 在每一行末尾加上终止符 ``<eos>``；
2. 按 ``--train-ratio`` 随机切分为训练集 ``train.txt`` 与测试集 ``test.txt``；
3. 用**训练集**的字符频次构建字符级词表 ``vocab.txt``（含特殊 token）。

所有输出均为 UTF-8（无 BOM），换行符统一为 ``\\n``。脚本可重复运行，
随机种子固定，结果可复现。

用法::

    python util/prepare_data.py
    python util/prepare_data.py --train-ratio 0.95 --seed 123 --min-freq 2
    python util/prepare_data.py --dedup   # 先去重再切分（语料重复率约 38%）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from tokenizer import EOS_TOKEN, SPECIAL_TOKENS, CharTokenizer  # noqa: E402

DEFAULT_CORPUS = ROOT / "data" / "corpus_general.txt"
DEFAULT_OUT_DIR = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为字符级语言模型准备训练/测试数据与词表")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="原始语料路径")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例，默认 0.9")
    parser.add_argument("--seed", type=int, default=42, help="随机切分种子，默认 42")
    parser.add_argument(
        "--min-freq", type=int, default=1, help="字符进入词表的最小频次，默认 1（全部保留）"
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="切分前去掉完全重复的行（默认保留；语料重复率较高，建议开启）",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    """读取语料：去掉 BOM、行尾换行与空行，行内内容保持不变。"""
    with path.open(encoding="utf-8-sig") as f:
        lines = [line.rstrip("\r\n") for line in f]
    return [line for line in lines if line.strip()]


def train_test_split(
    lines: list[str], train_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    """按行随机切分，返回 (train, test)。"""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 (0, 1) 之间")
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    n_train = int(len(lines) * train_ratio)
    train = [lines[i] for i in order[:n_train]]
    test = [lines[i] for i in order[n_train:]]
    return train, test


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def display_path(path: Path) -> str:
    """仓库内的路径显示为相对路径，仓库外的路径原样返回。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    # ---------------------------------------------------------------- 读取语料
    if not args.corpus.exists():
        sys.exit(f"找不到语料文件：{args.corpus}")
    lines = read_lines(args.corpus)
    if not lines:
        sys.exit(f"语料为空：{args.corpus}")

    # 语料中若已出现特殊 token 的字面形式，加 <eos> 会产生歧义，直接拒绝
    corpus_text = "\n".join(lines)
    for token in SPECIAL_TOKENS:
        if token in corpus_text:
            sys.exit(f"语料中已存在特殊 token 字面量 {token!r}，请先清理语料")

    source_lines = len(lines)
    duplicates = source_lines - len(set(lines))

    # 可选：去掉完全重复的行。重复行会让同一句同时落进训练集和测试集，造成信息泄漏
    removed_duplicates = 0
    if args.dedup:
        seen: set[str] = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        removed_duplicates = source_lines - len(unique_lines)
        lines = unique_lines

    # ------------------------------------------------------------ 1. 加终止符
    # 每一行是一条独立的文本（一句话/一次发言/一条新闻），因此终止符加在行尾。
    train_raw, test_raw = train_test_split(lines, args.train_ratio, args.seed)
    train_lines = [line + EOS_TOKEN for line in train_raw]
    test_lines = [line + EOS_TOKEN for line in test_raw]

    # ------------------------------------------------------- 3. 基于训练集建词表
    # 只用训练集统计，避免测试集信息泄漏到词表。
    tokenizer = CharTokenizer.build(train_raw, min_freq=args.min_freq)

    # ------------------------------------------------------------------- 写出
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    test_path = out_dir / "test.txt"
    vocab_path = out_dir / "vocab.txt"
    stats_path = out_dir / "stats.json"

    write_lines(train_path, train_lines)
    write_lines(test_path, test_lines)
    tokenizer.save(vocab_path)

    # ------------------------------------------------------------------- 统计
    train_chars = sum(len(line) for line in train_raw)
    test_chars = sum(len(line) for line in test_raw)
    train_tokens = sum(len(tokenizer.encode(line + EOS_TOKEN)) for line in train_raw)
    total_tokens = 0
    unk_tokens = 0
    for line in test_raw:
        ids = tokenizer.encode(line + EOS_TOKEN)
        total_tokens += len(ids)
        unk_tokens += sum(1 for i in ids if i == tokenizer.unk_id)

    stats = {
        "corpus": display_path(args.corpus),
        "encoding": "utf-8",
        "eos_token": EOS_TOKEN,
        "special_tokens": list(SPECIAL_TOKENS),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "min_freq": args.min_freq,
        "dedup": args.dedup,
        "source_lines": source_lines,
        "removed_duplicate_lines": removed_duplicates,
        "duplicate_lines": duplicates,
        "total_lines": len(lines),
        "total_chars": train_chars + test_chars,
        "average_line_chars": round((train_chars + test_chars) / len(lines), 2),
        "train": {
            "lines": len(train_lines),
            "chars": train_chars,
            "tokens": train_tokens,
            "file": display_path(train_path),
        },
        "test": {
            "lines": len(test_lines),
            "chars": test_chars,
            "tokens": total_tokens,
            "unk_tokens": unk_tokens,
            "unk_rate": round(unk_tokens / total_tokens, 6) if total_tokens else 0.0,
            "file": display_path(test_path),
        },
        "vocab": {
            "size": len(tokenizer),
            "file": display_path(vocab_path),
        },
    }
    with stats_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # ------------------------------------------------------------------- 输出
    print("=" * 62)
    print("数据准备完成")
    print("=" * 62)
    print(f"原始语料      : {args.corpus}")
    print(f"原始行数      : {source_lines:,} 行")
    if args.dedup:
        print(f"去重后行数    : {len(lines):,} 行（移除重复 {removed_duplicates:,} 行）")
    else:
        print(f"重复行        : {duplicates:,} 行（占 {duplicates / source_lines:.1%}，未去重）")
    print(f"总字数        : {train_chars + test_chars:,} 字")
    print("-" * 62)
    print(f"训练集        : {len(train_lines):,} 行 -> {train_path}")
    print(f"测试集        : {len(test_lines):,} 行 -> {test_path}")
    print(f"词表          : {len(tokenizer):,} 个 token -> {vocab_path}")
    print(f"特殊 token    : {', '.join(f'{t}={i}' for i, t in enumerate(SPECIAL_TOKENS))}")
    print(f"测试集 OOV    : {unk_tokens:,} / {total_tokens:,} token ({stats['test']['unk_rate']:.4%})")
    print(f"统计信息      : {stats_path}")
    print("-" * 62)
    print("训练集样例（前 3 行）：")
    for line in train_lines[:3]:
        print(f"  {line}")
    print("测试集样例（前 3 行）：")
    for line in test_lines[:3]:
        print(f"  {line}")


if __name__ == "__main__":
    main()

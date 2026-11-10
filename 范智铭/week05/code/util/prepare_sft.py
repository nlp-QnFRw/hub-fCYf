"""prepare_sft.py —— SFT 数据体检 / 统计脚本。

不训练、不改数据，只回答四个问题：

1. 数据能不能用？  字段齐不齐、有没有坏行、train/val 有没有重复泄漏；
2. 长度合不合适？  一条样本占多少 token、会不会被 max_len 截断；
3. 词表覆盖够不够？SFT 里有多少字符是词表外的（会变成 <unk>）；
4. mask 对不对？   打印一条样本，把「不参与 loss」的 prompt 用 [] 标出来。

统计结果写到 data/stats.json，方便写报告 / 对比不同版本的数据。

用法::

    python code/util/prepare_sft.py              # 体检 + 统计，打印报告
    python code/util/prepare_sft.py --show 3     # 额外打印 3 条带 mask 的样本
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# 让 import 能找到 code/ 和 code/util/（本文件在 code/util/ 下，向上两级到 week05）
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "util"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

from omegaconf import OmegaConf  # noqa: E402

from tokenizer import CharTokenizer  # noqa: E402
from util.sft_dataset import (  # noqa: E402
    IGNORE_INDEX,
    encode_example,
    normalize_text,
    preview,
)


def load_records(path: Path) -> list[dict]:
    """读取 jsonl，做字段校验。"""
    records, bad = [], 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                print(f"  ✗ {path.name}:{lineno} JSON 解析失败，已跳过")
                continue
            if "instruction" not in rec or "output" not in rec:
                bad += 1
                print(f"  ✗ {path.name}:{lineno} 缺少 instruction/output，已跳过")
                continue
            records.append(rec)
    if bad:
        print(f"  {path.name}: {bad} 条坏行")
    return records


def describe(values: list[int]) -> dict:
    """长度分布：min / 中位数 / 均值 / p95 / max。"""
    s = sorted(values)
    return {
        "min": s[0],
        "median": int(statistics.median(s)),
        "mean": round(sum(s) / len(s), 1),
        "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
        "max": s[-1],
    }


def check_repetition(name: str, records: list[dict]) -> None:
    """完全重复的问答对，会让模型死记硬背。"""
    counter = Counter(
        (normalize_text(r["instruction"]), normalize_text(r["output"])) for r in records
    )
    dup = {k: v for k, v in counter.items() if v > 1}
    print(f"  {name} 重复问答对：{len(dup)} 组" + (f"（最多重复 {max(dup.values())} 次）" if dup else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--show", type=int, default=1, help="打印几条带 mask 的样本")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    max_len = int(cfg.hparams.max_len)
    tok = CharTokenizer.from_file(cfg)
    specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
    char_vocab = {t for t in tok.itos if t not in specials}

    train_path = ROOT / "data" / cfg.hparams.sft_train_filename
    val_path = ROOT / "data" / cfg.hparams.sft_val_filename

    print("=" * 72)
    print(f"SFT 数据体检 | vocab {len(tok)} | max_len {max_len}")
    print("=" * 72)

    stats = {"vocab_size": len(tok), "max_len": max_len}
    oov_all = Counter()

    for split, path in (("train", train_path), ("val", val_path)):
        print(f"\n[{split}] {path.relative_to(ROOT)}")
        records = load_records(path)
        if not records:
            raise SystemExit(f"{path} 没有有效数据")

        prompt_lens, answer_lens, total_lens = [], [], []
        sup_tokens, all_tokens = 0, 0
        truncated = 0
        oov = Counter()

        for rec in records:
            input_ids, labels, trunc = encode_example(
                tok, rec["instruction"], rec["output"], max_len
            )
            total_lens.append(len(input_ids))
            sup = sum(1 for t in labels if t != IGNORE_INDEX)
            sup_tokens += sup
            all_tokens += len(input_ids)
            truncated += int(trunc)

            # prompt 长度 = 第一个监督位置之前的部分
            first_sup = next(i for i, t in enumerate(labels) if t != IGNORE_INDEX)
            prompt_lens.append(first_sup)
            answer_lens.append(len(input_ids) - first_sup)

            # OOV：不在字符词表里的字符（会编码成 <unk>），先做与训练一致的清洗
            for ch in normalize_text(rec["instruction"]) + normalize_text(rec["output"]):
                if ch not in char_vocab:
                    oov[ch] += 1
        oov_all.update(oov)

        print(f"  条数            : {len(records)}")
        print(f"  prompt 长度     : {describe(prompt_lens)}")
        print(f"  答案长度        : {describe(answer_lens)}")
        print(f"  总长度（含<eos>）: {describe(total_lens)}")
        print(f"  被 max_len 截断 : {truncated} 条")
        print(
            f"  监督 token 占比 : {sup_tokens / all_tokens:.1%} "
            f"（{sup_tokens:,} / {all_tokens:,}）"
        )
        print(f"  OOV 字符        : {len(oov)} 个" + (f" {list(oov)[:10]}" if oov else ""))
        check_repetition(split, records)

        stats[split] = {
            "file": path.name,
            "rows": len(records),
            "prompt_len": describe(prompt_lens),
            "answer_len": describe(answer_lens),
            "total_len": describe(total_lens),
            "truncated": truncated,
            "supervised_tokens": sup_tokens,
            "all_tokens": all_tokens,
            "supervised_ratio": round(sup_tokens / all_tokens, 4),
            "oov_chars": "".join(sorted(oov)),
        }

    # train / val 之间的重复（验证集泄漏会让指标虚高）
    train_keys = {(normalize_text(r["instruction"]), normalize_text(r["output"])) for r in load_records(train_path)}
    val_keys = {(normalize_text(r["instruction"]), normalize_text(r["output"])) for r in load_records(val_path)}
    overlap = train_keys & val_keys
    print(f"\n[泄漏检查] train 与 val 完全重复的问答对：{len(overlap)} 组")
    stats["train_val_overlap"] = len(overlap)

    if oov_all:
        print(f"[词表] 全部 OOV 字符（共 {len(oov_all)} 种）：{''.join(sorted(oov_all))}")
        print("       少量生僻字 OOV 属正常；如果很多，考虑扩充词表后重新预训练")

    # mask 可视化
    if args.show > 0:
        print("\n" + "=" * 72)
        print(f"[mask 可视化] [] 内 = 不参与 loss 的 prompt token（共 {args.show} 条）")
        print("=" * 72)
        for rec in load_records(train_path)[: args.show]:
            print("  " + preview(tok, rec["instruction"], rec["output"], max_len).replace("\n", "\n  "))
            print()

    out = ROOT / "data" / "stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"统计结果已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

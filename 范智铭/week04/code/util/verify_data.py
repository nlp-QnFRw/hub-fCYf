"""校验 prepare_data.py 生成的训练/测试数据与词表。

检查项：
1. 三个产物文件齐全，行数与 stats.json 一致；
2. train.txt / test.txt 每行恰好一个行尾 ``<eos>``，且不再出现其他特殊 token；
3. vocab.txt 前 4 行为 ``<pad> <unk> <bos> <eos>``，无重复，大小与 stats.json 一致；
4. 任意一行编码后所有 id 都在词表范围内；
5. 编码 -> 解码可无损还原（去掉行尾 ``<eos>`` 后与原文相等）；
6. 测试集 OOV 率与 stats.json 记录一致；
7. 输出文件无 BOM、换行为 ``\\n``。

用法::

    python code/verify_data.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizer import EOS_TOKEN, SPECIAL_TOKENS, CharTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    stats_path = DATA_DIR / "stats.json"
    if not stats_path.exists():
        sys.exit("找不到 data/stats.json，请先运行 python code/prepare_data.py")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    train_path = DATA_DIR / "train.txt"
    test_path = DATA_DIR / "test.txt"
    vocab_path = DATA_DIR / "vocab.txt"
    check("产物文件齐全", all(p.exists() for p in (train_path, test_path, vocab_path)))

    train = read_lines(train_path)
    test = read_lines(test_path)
    check(
        "行数与 stats.json 一致",
        len(train) == stats["train"]["lines"] and len(test) == stats["test"]["lines"],
        f"train={len(train):,}, test={len(test):,}",
    )
    check(
        "训练/测试比例符合预期",
        abs(len(train) / (len(train) + len(test)) - stats["train_ratio"]) < 1e-3,
        f"train_ratio={len(train) / (len(train) + len(test)):.4f}",
    )

    # ------------------------------------------------------------ 终止符检查
    bad_eos = [i for i, line in enumerate(train) if not line.endswith(EOS_TOKEN)]
    check("训练集每行以 <eos> 结尾", not bad_eos, f"异常行数={len(bad_eos)}")
    bad_eos = [i for i, line in enumerate(test) if not line.endswith(EOS_TOKEN)]
    check("测试集每行以 <eos> 结尾", not bad_eos, f"异常行数={len(bad_eos)}")
    check(
        "每行只有一个 <eos>",
        all(line.count(EOS_TOKEN) == 1 for line in train + test),
    )
    other_specials = [t for t in SPECIAL_TOKENS if t != EOS_TOKEN]
    check(
        "数据中未混入其他特殊 token",
        not any(t in line for line in train + test for t in other_specials),
    )

    # ---------------------------------------------------------------- 词表检查
    tokenizer = CharTokenizer.from_file(vocab_path)
    check("词表前 4 项为特殊 token", tuple(tokenizer.itos[:4]) == SPECIAL_TOKENS,
          str(tokenizer.itos[:4]))
    check("词表大小与 stats.json 一致", len(tokenizer) == stats["vocab"]["size"],
          f"vocab_size={len(tokenizer):,}")
    check("词表无重复项", len(set(tokenizer.itos)) == len(tokenizer))

    # ------------------------------------------------------- 编码 / 解码往返
    sample = random.Random(0).sample(test, min(2000, len(test)))
    encoded = [tokenizer.encode(line) for line in sample]
    check(
        "抽样编码后 id 均在词表范围内",
        all(0 <= i < len(tokenizer) for ids in encoded for i in ids),
    )

    # 含 OOV 字符的行无法还原（这正是 <unk> 的含义），只对可编码的行做往返校验
    known = [(line, ids) for line, ids in zip(sample, encoded) if tokenizer.unk_id not in ids]
    mismatches = sum(
        1
        for line, ids in known
        if ids[-1] != tokenizer.eos_id
        or tokenizer.decode(ids) != line[: -len(EOS_TOKEN)]
    )
    check(
        "抽样编码/解码可无损还原",
        mismatches == 0,
        f"抽检={len(known)} 行，不匹配={mismatches}",
    )

    # ------------------------------------------------------------------ OOV
    unk = sum(
        1
        for line in test
        for i in tokenizer.encode(line)
        if i == tokenizer.unk_id
    )
    total = sum(len(tokenizer.encode(line)) for line in test)
    rate = unk / total if total else 0.0
    check(
        "测试集 OOV 率与 stats.json 一致",
        abs(rate - stats["test"]["unk_rate"]) < 1e-6,
        f"实测={rate:.4%}, 记录={stats['test']['unk_rate']:.4%}",
    )

    # ------------------------------------------------------------ 文件格式
    has_bom = any(p.read_bytes()[:3] == b"\xef\xbb\xbf" for p in (train_path, test_path, vocab_path))
    check("输出文件无 BOM", not has_bom)
    check(
        "输出文件使用 \\n 换行",
        b"\r\n" not in train_path.read_bytes() and b"\r\n" not in test_path.read_bytes(),
    )

    print("-" * 62)
    print(f"总 token 数：训练 {stats['train']['tokens']:,}，测试 {stats['test']['tokens']:,}")
    if _failures:
        print(f"校验失败：{len(_failures)} 项 -> {', '.join(_failures)}")
        sys.exit(1)
    print("全部校验通过")


if __name__ == "__main__":
    main()

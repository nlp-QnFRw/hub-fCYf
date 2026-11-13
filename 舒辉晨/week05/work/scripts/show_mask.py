"""把一条样本的 loss mask 打印出来 —— 这是理解 SFT 最快的方式。

运行：
    python scripts/show_mask.py              # 看第 0 条
    python scripts/show_mask.py --index 5    # 看第 5 条
    python scripts/show_mask.py --custom "什么是人工智能？" "人工智能是让机器模拟人类智能的技术。"

你会看到：前面 prompt 段全是「—」（不学），从 <|assistant|> 之后才开始「学」。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sftlab.data import read_jsonl  # noqa: E402
from sftlab.template import describe_mask, encode_example, render_template  # noqa: E402
from sftlab.tokenizer import CharTokenizer  # noqa: E402

BAR_WIDTH = 60


def show(tok: CharTokenizer, instruction: str, output: str, max_rows: int = 46) -> None:
    ex = encode_example(tok, instruction, output)
    rows = describe_mask(tok, ex)

    print("\n" + "=" * 74)
    print("模型看到的字符串：")
    print("  " + render_template(tok, instruction, output))
    print(f"\nprompt {ex.prompt_len} token ｜ 回答 {ex.answer_len} token ｜ "
          f"序列共 {len(ex.ids)} token ｜ 参与 loss {ex.supervised_ratio:.1%}")
    print("=" * 74)
    print(f"{'位置':>5} │ {'token':<12} │ 是否学")
    print("-" * 74)

    shown = rows[:max_rows]
    for pos, token, sup in shown:
        label = token.replace("\n", "\\n").replace(" ", "·")
        print(f"{pos:>5} │ {label:<12} │ {'学 ✔' if sup else '—'}")

    if len(rows) > max_rows:
        print(f"  …… 省略 {len(rows) - max_rows} 个 token")

    # 用一条色块把 mask 画出来：█ = 参与 loss，· = 被 mask
    bar = "".join("█" if sup else "·" for _, _, sup in rows)
    print("-" * 74)
    print("mask 分布（█ 学习 / · 忽略）:")
    for i in range(0, len(bar), BAR_WIDTH):
        print("  " + bar[i : i + BAR_WIDTH])

    print("\n注意：label 是「下一 token」，所以位置 t 的预测对应 label[t+1]。")
    print("      prompt 的最后一位（<|assistant|>）虽然自己不算 loss，")
    print("      但它要预测回答的第一个字 —— 这个位置是有梯度的。")


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 loss mask")
    parser.add_argument("--index", type=int, default=0, help="看训练集第几条")
    parser.add_argument("--custom", nargs=2, metavar=("INSTRUCTION", "OUTPUT"),
                        help="自定义一条样本，格式: --custom 问题 回答")
    args = parser.parse_args()

    vocab = ROOT / "data" / "processed" / "vocab.json"
    if not vocab.exists():
        raise SystemExit("请先运行: python scripts/prepare_data.py")
    tok = CharTokenizer.load(vocab)

    if args.custom:
        instruction, output = args.custom
    else:
        rows = read_jsonl(ROOT / "data" / "raw" / "sft_train.jsonl")
        rec = rows[args.index]
        instruction, output = rec["instruction"], rec["output"]

    show(tok, instruction, output)


if __name__ == "__main__":
    main()

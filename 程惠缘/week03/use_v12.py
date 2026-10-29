# -*- coding: utf-8 -*-
"""use_v12.py —— 12 语种识别：命令行交互式使用

【默认就是交互模式】直接回车就能一句一句地问：
    $ python use_v12.py
      > 今天天气不错
        zh  99.94%  ####...
      > Preciso terminar isso
        pt  99.97%  ####...

【其它用法】
    python use_v12.py -t "Bonjour tout le monde"   # 只判一句（可写多次 -t）
    echo "..." | python use_v12.py                 # 从管道批量读
    python use_v12.py -m 别的模型.pt                # 换模型
    python use_v12.py -k 5                         # 看前 5 名

【交互模式里的命令】
    直接输入句子回车 → 判定
    空行 / :q / quit / exit / Ctrl-D  → 退出
    :h  → 帮助
"""
import os
import sys
import argparse

import torch

from model_v12 import MAX_LEN, load_for_inference, predict_proba, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "langid_v12.pt")


def show(model, vocab, labels, text, topk, maxlen, bar_width=32):
    norm = normalize(text)
    if not norm:
        print("  （这句归一化后没有任何字母 —— 纯数字/符号/emoji 判不了）")
        return
    if len(norm) > maxlen:
        print(f"  （只看前 {maxlen} 个字符，这句有 {len(norm)} 个）")
    probs = predict_proba(model, vocab, text, maxlen)
    k = min(topk, len(labels))
    vals, idx = probs.topk(k)
    for i, (p, j) in enumerate(zip(vals.tolist(), idx.tolist())):
        if i >= 2 and p < 0.001:          # 前两名一定显示，之后低于 0.1% 就不占屏
            continue
        bar = "#" * int(round(p * bar_width))
        mark = "  ←" if i == 0 else ""
        print(f"  {labels[j]:<4} {p:6.2%}  {bar:<{bar_width}}{mark}")


def repl(model, vocab, labels, topk, maxlen):
    print("\n输入一句话回车判定；空行或 :q 退出，:h 看帮助。")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not line:
            print("再见。")
            return
        if line in (":q", "quit", "exit"):
            return
        if line == ":h":
            print("  直接输入句子回车即可。空行 / :q / quit / exit / Ctrl-D 退出。")
            print(f"  当前模型支持：{' '.join(labels)}")
            continue
        show(model, vocab, labels, line, topk, maxlen)


def main():
    ap = argparse.ArgumentParser(description="12 语种识别 · 命令行交互")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help="模型 .pt 路径")
    ap.add_argument("-t", "--text", action="append", default=[], help="要判断的句子，可重复")
    ap.add_argument("-k", "--topk", type=int, default=3, help="显示前几名（默认 3）")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"找不到模型：{args.model}\n"
            f"  先训一个：python train_v12.py\n"
            f"  或者指定：python use_v12.py -m 你的模型.pt"
        )

    torch.set_num_threads(1)
    model, vocab, labels, maxlen, ckpt = load_for_inference(args.model)

    print(f"模型：{os.path.basename(args.model)}   结构 {ckpt['arch']}   "
          f"{len(vocab)} 字符 / {len(labels)} 类 / 最长 {maxlen} 字符")
    print(f"语种：{' '.join(labels)}")
    if ckpt.get("metrics", {}).get("test_acc") is not None:
        m = ckpt["metrics"]
        print(f"训练记录：测试集 accuracy {m['test_acc']:.4f}   macro-F1 {m.get('macro_f1', 0):.4f}")

    if args.text:
        for t in args.text:
            print(f"\n{t}")
            show(model, vocab, labels, t, args.topk, maxlen)
    elif not sys.stdin.isatty():
        piped = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
        for line in piped:
            print(f"\n{line}")
            show(model, vocab, labels, line, args.topk, maxlen)
        if not piped:
            print("\n（管道里没有内容，进交互模式）")
            repl(model, vocab, labels, args.topk, maxlen)
    else:
        repl(model, vocab, labels, args.topk, maxlen)


if __name__ == "__main__":
    main()

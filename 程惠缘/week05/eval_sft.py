# -*- coding: utf-8 -*-
"""eval_sft.py —— 给 SFT 模型的「答题能力」装一个仪表盘

在没有这个脚本之前，唯一能看的是「验证 PPL」。但 PPL 19.75 的模型，
完全一致率是 0 —— **PPL 不等于会答题**。所以这里换成四个能直接说明问题的指标：

    · 停住率    模型自己吐出 <|end|> 的比例（停不下来 = 车轱辘话）
    · 严重重复  3-gram 重复占比 > 15% 的样本比例
    · 相似度    1 - 编辑距离 / 标准答案长度（0 = 完全不沾边，1 = 一字不差）
    · 答题正确率 只在「能程序判对错」的题上算（平均值 / 乘积 / 日期差 / 较大数）

最后一个是重点：它把「答得像」和「答得对」分开了。

用法:
    python eval_sft.py                                  # 用默认模型和默认解码参数
    python eval_sft.py --penalty 1.0 --early-stop 999   # 关掉重复惩罚，对比修复前的样子
    python eval_sft.py --show 5                         # 额外打印 5 条错例
"""
import os
import sys
import re
import argparse
import collections
from datetime import date

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_sft as T          # 复用模型加载、字表、解码函数（保证评的是训练脚本里那套）


# ─────────────────────────── 文本指标 ───────────────────────────
def repeat_ratio(text):
    """3-gram 里重复的比例。0 = 没有车轱辘话，越大越严重。"""
    n = len(text) - 2
    if n <= 0:
        return 0.0
    grams = [text[i:i + 3] for i in range(n)]
    return 1.0 - len(set(grams)) / n


def edit_distance(a, b):
    """Levenshtein 距离：把 a 改成 b 最少要几步（增/删/改各算一步）。"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,            # 删
                           cur[j - 1] + 1,         # 增
                           prev[j - 1] + (ca != cb)))   # 改
        prev = cur
    return prev[-1]


def similarity(pred, gold):
    """0~1 的相似度：1 表示一字不差。"""
    if not gold:
        return 0.0
    return max(0.0, 1.0 - edit_distance(pred, gold) / len(gold))


# ─────────────────────────── 答题正确性（程序判卷）───────────────────────────
NUM = re.compile(r"\d+(?:\.\d+)?")


def _numbers(s):
    """从文本里抠出所有数字（去掉千分位逗号）。"""
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    return [float(x) for x in NUM.findall(s)]


def expected_value(instruction):
    """对能程序算的题，返回 (题型, 正确数值)；算不了就返回 None。"""
    ins = instruction
    # 平均值：取最后一个冒号后面的那串数字
    if "平均值" in ins:
        tail = ins.split("：")[-1].split(":")[-1]
        ns = _numbers(tail)
        if len(ns) >= 2:
            return ("平均值", sum(ns) / len(ns))
    # 乘积（排除「乘积最大」那种优化题）
    if "乘积" in ins and "最大" not in ins:
        ns = _numbers(ins)
        if len(ns) == 2:
            return ("乘积", ns[0] * ns[1])
    # 日期差
    if "天数" in ins:
        ds = re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", ins)
        if len(ds) == 2:
            try:
                d1, d2 = date(*map(int, ds[0])), date(*map(int, ds[1]))
                return ("日期差", abs((d2 - d1).days))
            except ValueError:
                return None
    # 取较大数
    if ("较大" in ins or "最大" in ins) and "乘积" not in ins:
        ns = _numbers(ins)
        if len(ns) == 2:
            return ("较大数", max(ns))
    return None


def judge(answer, kind_value):
    """答题是否正确：正确数值出现在回答的数字里就算对。"""
    _, want = kind_value
    for got in _numbers(answer):
        if abs(got - want) <= 0.05:
            return True
    return False


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SFT 模型答题能力评测")
    ap.add_argument("--model", default=os.path.join(HERE, "best_sft_model.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "sft_val.jsonl"))
    ap.add_argument("--penalty", type=float, default=1.3, help="重复惩罚；1.0 = 关闭")
    ap.add_argument("--early-stop", type=int, default=45, help="兜底刹车阈值；调很大等于关闭")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条（0 = 全部）")
    ap.add_argument("--show", type=int, default=0, help="打印几条错例")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)
    if not os.path.exists(args.model):
        raise SystemExit(f"找不到模型：{args.model}\n  先训练：python train_sft.py")

    model, char2idx, idx2char, ckpt = T.load_ckpt(args.model, device)
    rows = T.read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]

    print(f"模型：{os.path.basename(args.model)}   验证集 {len(rows)} 条")
    print(f"解码：重复惩罚 {args.penalty}   兜底刹车 {args.early_stop}"
          f"   最长 {args.max_new} 字\n")

    stops = collections.Counter()
    n_rep = 0
    n_exact = 0
    sim_sum = 0.0
    task_stats = collections.defaultdict(lambda: [0, 0])   # 题型 → [判对, 判了几条]
    failures = []

    for q, gold in rows:
        stats = {}
        pred = T.answer_question(model, char2idx, idx2char, q, device,
                                 max_new=args.max_new, rep_penalty=args.penalty,
                                 early_stop_after=args.early_stop, stats=stats)
        stops[stats.get("stop", "?")] += 1
        if repeat_ratio(pred) > 0.15:
            n_rep += 1
        if pred.strip() == gold.strip():
            n_exact += 1
        sim_sum += similarity(pred.strip(), gold.strip())

        kv = expected_value(q)
        if kv is not None:
            ok = judge(pred, kv)
            task_stats[kv[0]][0] += ok
            task_stats[kv[0]][1] += 1
            if not ok and len(failures) < args.show:
                failures.append((q, gold, pred, kv))

    n = len(rows)
    stopped = stops["end"]
    print("=" * 64)
    print(f"  停住率（自己吐出 <|end|>）      {stopped:>3}/{n:<3} = {stopped/n:6.1%}")
    print(f"  严重重复（3-gram 重复>15%）     {n_rep:>3}/{n:<3} = {n_rep/n:6.1%}")
    print(f"  完全一致（Exact Match）         {n_exact:>3}/{n:<3} = {n_exact/n:6.1%}")
    print(f"  平均相似度（1-编辑距离/长度）    {sim_sum/n:>10.3f}")
    print("=" * 64)

    if task_stats:
        total_ok = sum(v[0] for v in task_stats.values())
        total_n = sum(v[1] for v in task_stats.values())
        print(f"\n  程序判卷的题（共 {total_n} 条）：答对 {total_ok} 条 = {total_ok/total_n:.1%}")
        for kind in sorted(task_stats):
            ok, tot = task_stats[kind]
            print(f"    {kind:<6} {ok:>3}/{tot:<3} = {ok/max(1,tot):6.1%}")
    else:
        print("\n  （没有可程序判卷的题）")

    print(f"\n  停止方式分布：{dict(stops)}   （end=正常停，early=兜底刹车，max=到上限）")

    for q, gold, pred, kv in failures[:args.show]:
        print(f"\n  ✗ 题型 {kv[0]}，正确答案 {kv[1]:g}")
        print(f"    问：{q}")
        print(f"    答：{pred}")
        print(f"    标准：{gold}")


if __name__ == "__main__":
    main()

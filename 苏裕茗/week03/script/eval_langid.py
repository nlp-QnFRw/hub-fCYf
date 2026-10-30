# -*- coding: utf-8 -*-
"""
eval_langid.py —— 语种识别模型评测（作业 12 语种版）

直接复用 train_langid_cls.py 里的模型定义和字符表构建方法，所以
**学生改了训练脚本里的模型结构，评测会自动跟着变**（作业打分的统一口径）。

字符表、标签、层宽、dropout 全部从 checkpoint 里读，训练时用的是什么就是什么，
评测这边不用再配任何参数（下面的命令省略了脚本路径，实际可以写全路径从任意目录运行）：

    python train_langid_cls.py --save langid_rnn_v12.pt                # 先训练并存档
    python eval_langid.py --model langid_rnn_v12.pt                    # test 集
    python eval_langid.py --model langid_rnn_v12.pt --split dev        # dev 集
    python eval_langid.py --model langid_rnn_v12.pt --split test_hard  # 难度切片
    python eval_langid.py --model langid_rnn_v12.pt --openset          # 开放集拒识

会打印：accuracy / macro-F1 / 逐类 P-R-F1 / 最常混淆的组合 / 按长度分档的准确率

数据默认读上一级的 data/（train.csv / dev.csv / test.csv，12 个语种）。
test_hard.csv 和 test_openset.csv 是课程资料，作业数据目录里没有时自动去课程
 data/langid/ 找；两处都没有就跳过该项，不会报错。

【运行位置】--model / --data-dir 的相对路径都按脚本所在目录解析，所以在哪个
目录下敲命令都一样，不用先 cd 进来：

    python 苏裕茗/week03/script/train_langid_cls.py --save langid_rnn_v12.pt
    python 苏裕茗/week03/script/eval_langid.py --model langid_rnn_v12.pt
"""

import os
import sys
import csv
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
# 保证从任意目录运行时都能 import 到同目录下的 train_langid_cls.py
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from train_langid_cls import (LangIDClassifier, build_char_vocab, encode,
                              normalize, macro_f1, confusion, print_confusion,
                              MAX_LEN, EMBED_DIM, HIDDEN_DIM, DROPOUT)

# 作业数据：脚本在 script/ 下，数据在上一级的 data/
DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "data"))
# 课程资料：test_hard.csv / test_openset.csv 在课程 data/langid/ 下
COURSE_DATA = os.path.abspath(os.path.join(HERE, "..", "..", "data", "langid"))


def resolve_path(path, base=HERE):
    """相对路径一律按脚本所在目录解析，从任何 cwd 运行结果都一样。"""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))


def find_aux(name, data_dir):
    """按 --data-dir → 作业数据目录 → 课程资料目录 的顺序找辅助数据文件。"""
    for d in (data_dir, DATA_ROOT, COURSE_DATA):
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    return None


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    labels = ckpt["labels"]
    model = LangIDClassifier(len(vocab), len(labels),
                             embed_dim=ckpt.get("embed_dim", EMBED_DIM),
                             hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
                             dropout=ckpt.get("dropout", DROPOUT))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, labels


def read_csv(path, with_slice=False):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def score_batch(model, texts, vocab):
    """返回每条的 softmax 概率向量。"""
    ids = torch.tensor([encode(t, vocab) for t in texts])
    with torch.no_grad():
        return torch.softmax(model(ids), dim=1)


def accuracy_by_length(texts, preds, golds):
    """语种识别最真实的难点在短文本，按长度分档看才看得见。"""
    buckets = [(2, 7), (8, 12), (12, 20), (20, 30), (30, 999)]
    print("\n  按长度分档的准确率：")
    for lo, hi in buckets:
        idx = [i for i, t in enumerate(texts) if lo <= len(normalize(t)) < hi]
        if not idx:
            continue
        hit = sum(preds[i] == golds[i] for i in idx)
        print(f"    {lo:>3}~{hi if hi < 999 else '+':>3} 字符："
              f"{hit:4d}/{len(idx):4d} = {hit/len(idx):.3f}")


def per_class_report(preds, golds, labels):
    print("\n  逐类指标：")
    print(f"    {'标签':<6}{'样本':>6}{'precision':>12}{'recall':>10}{'f1':>10}")
    for c, lab in enumerate(labels):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"    {lab:<6}{tp+fn:>6}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}")


def run_slices(model, vocab, labels, data_dir):
    """难度集：按 slice 分组算准确率。"""
    path = find_aux("test_hard.csv", data_dir)
    if path is None:
        print("\n=== 难度诊断集：跳过（找不到 test_hard.csv）===")
        return
    rows = read_csv(path)
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"\n=== 难度诊断集 {len(rows)} 条 ===")
    by_slice = {}
    for r in rows:
        by_slice.setdefault(r["slice"], []).append(r)

    for name in ("short", "nodeaccent", "nokana"):
        sub = by_slice.get(name, [])
        sub = [r for r in sub if r["label"] in label2id]     # 只评测模型认识的标签
        if not sub:
            print(f"  [{name}] 无可用样本")
            continue
        probs = score_batch(model, [r["text"] for r in sub], vocab)
        preds = probs.argmax(dim=1).tolist()
        golds = [label2id[r["label"]] for r in sub]
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"  [{name}] {len(sub):4d} 条  accuracy = {acc:.3f}")
        if name == "nokana":
            c = Counter((labels[g], labels[p]) for p, g in zip(preds, golds))
            for (g, p), n in c.most_common(3):
                print(f"      真实 {g} → 预测 {p}: {n} 条")
        # 每类小计
        per = Counter(g for g in golds)
        hit = Counter(g for p, g in zip(preds, golds) if p == g)
        print("      " + "  ".join(f"{labels[g]}:{hit[g]}/{per[g]}"
                                   for g in sorted(per)))


def run_openset(model, vocab, labels, dev_path, data_dir):
    """
    开放集拒识：训练里没出现过的语种（el/bg/sw/ur）喂给模型，看它敢不敢说"不认识"。

    做法：先用验证集标定一个阈值 —— 验证集上 max(softmax) 的 5% 分位数，
    意味着"正常输入里有 95% 的置信度高于它"。测试时低于阈值就判 unknown。
    """
    path = find_aux("test_openset.csv", data_dir)
    if path is None:
        print("\n=== 开放集拒识：跳过（找不到 test_openset.csv）===")
        return
    rows = read_csv(path)
    dev = read_csv(dev_path)
    print(f"\n=== 开放集拒识（{len(rows)} 条未见语种）===")
    print(f"  训练见过的语种：{' '.join(labels)}")
    print(f"  开放集语种：{' '.join(sorted({r['label'] for r in rows}))}")

    dev_probs = score_batch(model, [r["text"] for r in dev], vocab)
    dev_conf = dev_probs.max(dim=1)[0]
    thr = torch.quantile(dev_conf, 0.05).item()

    os_probs = score_batch(model, [r["text"] for r in rows], vocab)
    os_conf = os_probs.max(dim=1)[0]

    keep = (dev_conf >= thr).float().mean().item()
    rej = (os_conf < thr).float().mean().item()
    print(f"  阈值（验证集 max-prob 的 5% 分位）= {thr:.3f}")
    print(f"  正常输入保留率 = {keep:.1%}   开放集拒识率 = {rej:.1%}")
    print(f"  开放集平均置信度 = {os_conf.mean():.3f} （正常输入 = {dev_conf.mean():.3f}）")

    print("\n  没见过的语种被认成了谁（超过阈值的那部分）：")
    preds = os_probs.argmax(dim=1).tolist()
    for lab in sorted({r["label"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        c = Counter(labels[preds[i]] for i in idx)
        top = ", ".join(f"{k}:{v}" for k, v in c.most_common(3))
        mean_conf = os_conf[idx].mean().item()
        print(f"    {lab}（{len(idx)} 条，平均置信度 {mean_conf:.2f}）→ {top}")

    print("\n  阈值扫描（阈值 | 正常保留 | 开放集拒识）：")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        k = (dev_conf >= t).float().mean().item()
        r = (os_conf < t).float().mean().item()
        print(f"    {t:.2f}   |  {k:6.1%}  |  {r:6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "langid_rnn_v12.pt"),
                    help="checkpoint 路径，默认 script/langid_rnn_v12.pt；相对路径按脚本目录解析")
    ap.add_argument("--split", default="test", choices=["test", "dev", "test_hard", "none"])
    ap.add_argument("--openset", action="store_true")
    ap.add_argument("--data-dir", default=DATA_ROOT,
                    help="数据目录，默认 ../data；相对路径按脚本目录解析")
    args = ap.parse_args()
    args.model = resolve_path(args.model)
    args.data_dir = resolve_path(args.data_dir)

    if not os.path.exists(args.model):
        raise SystemExit(f"找不到模型文件：{args.model}\n"
                         f"  先训练并存档（--save 的相对路径落在 script/ 下）："
                         f"python train_langid_cls.py --save langid_rnn_v12.pt")

    if args.split != "none":
        split_path = os.path.join(args.data_dir, f"{args.split}.csv")
        if not os.path.exists(split_path):
            raise SystemExit(f"找不到数据文件：{split_path}\n"
                             f"  当前数据目录：{args.data_dir}\n"
                             f"  期望结构：week03/data/train.csv、dev.csv、test.csv")

    torch.set_num_threads(1)
    model, vocab, labels = load_model(args.model)

    if args.split == "test_hard":
        run_slices(model, vocab, labels, args.data_dir)
    elif args.split != "none":
        rows = read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
        label2id = {l: i for i, l in enumerate(labels)}
        texts = [r["text"] for r in rows]
        probs = score_batch(model, texts, vocab)
        preds = probs.argmax(dim=1).tolist()
        golds = [label2id[r["label"]] for r in rows]
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"=== {args.split}：{len(rows)} 条 ===")
        print(f"  accuracy = {acc:.4f}   macro-F1 = {macro_f1(preds, golds, len(labels)):.4f}")
        per_class_report(preds, golds, labels)
        print_confusion(confusion(preds, golds, labels), labels, top=5)
        accuracy_by_length(texts, preds, golds)

    if args.openset:
        run_openset(model, vocab, labels,
                    os.path.join(args.data_dir, "dev.csv"), args.data_dir)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""train_v12.py —— 12 语种语种识别：训练 + 存档

任务：一句话 → 12 种语言里的哪一种（ar en es hi ja ko pt ru th tr vi zh）

【这个脚本回答作业里的两个坑】
  1. **标签要扩到 12 类**：课堂脚本里 LABELS 写死了 8 个语种，模型输出维度 = len(LABELS)，
     多出来的语种没有输出格子。这里直接从训练集里读出标签，不再写死。
  2. **字符表要重新生成**：课堂用的 vocab_v8.txt 是从 v8 训练集统计的，
     新语种（ko/th/vi/tr）的字符不在里面，会全部落到 <UNK>。
     这里用 build_char_vocab() 从 v12/train.csv 重新统计，并写成 vocab_v12.txt。

【运行】
    python train_v12.py                       # 默认：BiLSTM + 20 epoch
    python train_v12.py --arch rnn --epochs 8 # 等价于「课堂模型直接改成 12 类」
    python train_v12.py --out my.pt           # 换个存档路径

【产出】
    langid_v12.pt      模型（结构 + 权重 + 字符表 + 标签 + 指标，全打包）
    vocab_v12.txt      字符表（字符 \\t id \\t 出现次数），就是作业要求你生成的那个文件
"""
import os
import csv
import time
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_v12 import (MAX_LEN, ARCHS, normalize, build_char_vocab, save_vocab,
                       coverage, encode, build_model)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "v12")

SEED = 42
EMBED_DIM = 64
HIDDEN_DIM = 64       # 单向的隐藏维数；bilstm 每个方向减半，拼起来仍是 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3


# ─── 数据 ──────────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 评测 ──────────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def main():
    ap = argparse.ArgumentParser(description="12 语种识别：训练并生成 .pt")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--arch", default="bilstm", choices=ARCHS,
                    help="默认 bilstm（实测最好）；改成 rnn 就是课堂模型直接扩到 12 类")
    ap.add_argument("--epochs", type=int, default=20,
                    help="默认 20（实测：8 epoch 时 RNN 明显没训够）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "langid_v12.pt"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # ① 标签从数据里来，不写死 —— 这就是作业要改的第一处
    labels = sorted({l for _, l in train_rows})
    label2id = {l: i for i, l in enumerate(labels)}
    n_class = len(labels)
    extra = sorted({l for r in (dev_rows, test_rows) for _, l in r} - set(labels))
    if extra:
        raise SystemExit(f"dev/test 里出现了训练集没见过的标签：{extra}")

    # ② 字符表从训练集重新统计 —— 作业要改的第二处
    vocab, freq = build_char_vocab([t for t, _ in train_rows])
    vocab_path = os.path.join(HERE, "vocab_v12.txt")
    save_vocab(vocab, freq, vocab_path)

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} / 验证 {len(dev_rows)} / 测试 {len(test_rows)} 条")
    print(f"  类别 {n_class} 个：{' '.join(labels)}")
    print(f"  字符表 {len(vocab)} 个字符（含 <PAD>/<UNK>），"
          f"前 {coverage(vocab, freq)} 个覆盖 95%  → {os.path.basename(vocab_path)}")
    print(f"  随机猜的基准：{1 / n_class:.1%}")
    print(f"  结构 {args.arch} / {args.epochs} epoch / 种子 {args.seed}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    # BiLSTM 每个方向减半，两个方向拼起来才是 HIDDEN_DIM 维
    hidden = HIDDEN_DIM // 2 if args.arch == "bilstm" else HIDDEN_DIM
    model = build_model(args.arch, len(vocab), n_class,
                        embed_dim=EMBED_DIM, hidden_dim=hidden, dropout=DROPOUT)
    print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            loss = crit(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        dp, dg = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(dp, dg)) / len(dg)
        print(f"  Epoch {epoch:2d}/{args.epochs}  loss={total / len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    print(f"\n  训练耗时 {time.time() - t0:.0f} 秒")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, n_class)
    print(f"  测试集 accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")

    print("\n  各语种准确率：")
    for c, lab in enumerate(labels):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit / max(1, len(idx)):.3f}")

    mat = [[0] * n_class for _ in range(n_class)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(n_class) for p in range(n_class) if g != p]
    pairs.sort(reverse=True)
    print("\n  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:5]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")

    torch.save({
        "format": 1,
        "arch": args.arch,
        "config": {"embed_dim": EMBED_DIM, "hidden_dim": hidden,
                   "dropout": DROPOUT, "bidirectional": args.arch == "bilstm"},
        "state_dict": model.state_dict(),
        "vocab": dict(vocab),
        "labels": labels,
        "maxlen": MAX_LEN,
        "profile": "v12",
        "trained": {"epochs": args.epochs, "seed": args.seed,
                    "batch": BATCH_SIZE, "lr": LR},
        "metrics": {"test_acc": acc, "macro_f1": f1},
    }, args.out)
    print(f"\n  模型已保存：{args.out}  ({os.path.getsize(args.out) / 1024:.0f} KB)")
    print(f"  字符表已保存：{vocab_path}")
    print(f"\n  拿去用：python use_v12.py -m \"{os.path.basename(args.out)}\"")


if __name__ == "__main__":
    main()

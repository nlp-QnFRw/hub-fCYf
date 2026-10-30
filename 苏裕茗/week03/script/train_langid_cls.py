# -*- coding: utf-8 -*-
"""
train_langid_cls.py —— 多语种「语种识别」分类器（综合实战脚本 · 作业 12 语种版）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：../data（train.csv / dev.csv / test.csv，12 语种）
运行：CPU 即可，单线程实测约 35 秒。

【运行位置】脚本里所有路径（数据目录、--save 存档位置）都以脚本自身所在目录
为基准解析，所以在哪个目录下敲命令都一样，不用先 cd 进来：

    python 苏裕茗/week03/script/train_langid_cls.py                       # 12 语种
    python 苏裕茗/week03/script/train_langid_cls.py --save langid_rnn_v12.pt
    # --save 的相对路径落在 script/ 下（即 script/langid_rnn_v12.pt），
    # 和 eval_langid.py 的默认 --model 是同一个位置。

【本课想让学生看到的四件事】
  1. 同一份代码、同一张字符表，换数据就能处理 8 种语言文字
  2. 字符级不需要分词器：中文按字、英文按字母、印地语按字母+元音标记，一视同仁
  3. 组件是一个个加进去的：去掉 LayerNorm / Dropout / RNN 都能现场对比
  4. 错得最有意思的是 es ↔ pt（同一套拉丁字母），不是"模型没学会"

【这份脚本是「12 类版」—— 语种列表写死，字符表现算】
  · LABELS 直接列出了这 12 个语种，模型的输出维度 = len(LABELS)
  · 字符表不再读预先做好的 vocab_v8.txt，而是用 build_char_vocab() 在训练集上现算
  要换数据集/加语种时，只需改 LABELS 这一处。

【字符表是怎么来的】见 build_char_vocab()：
  把训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号。
"""


"""
解题思路（思路自己写，实现由AI实现）：

L66 LABELS -->改成具体16个语种参数

L269 load_vocab_file 改成用 build_char_vocab，load_vocab_file删掉，在这个例子中

DATA_DIR  --> 改成当前目录的data 文件夹（week3_homework/data/；train/dev/test.csv）

训练过程和验证脚本不用改，改完后跑一次看效果
"""
import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 中文中位长度约 10 字符、德语约 39，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.4
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
# 数据在上一级的 data/（train.csv / dev.csv / test.csv），脚本在 script/ 子目录里
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))

# 作业的 12 个语种（课内 8 个 + ko / th / vi / tr）：模型的输出就是这 12 个格子
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


def resolve_path(path, base=HERE):
    """相对路径一律按脚本所在目录解析，从任何 cwd 运行结果都一样。"""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    多语种的关键点：这张表不需要按语种分开建。汉字、拉丁字母、西里尔字母、
    阿拉伯字母、天城文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    主流程就是用它在训练集上现算字符表（作业的 12 个语种里 ko / th / vi / tr
    是课内没见过的，预先做好的 vocab_v8.txt 覆盖不到它们）；换了训练集同理，
    不重新算的话新语种的字符会全部落到 1 号 <UNK>，模型等于睁眼瞎。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


def unk_rate(rows, vocab):
    """这些句子里有多少字符落到了 <UNK>，用来检查字符表够不够用。"""
    total = unk = 0
    for text, _ in rows:
        for ch in normalize(text):
            total += 1
            unk += ch not in vocab
    return unk / total if total else 0.0


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
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


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -inf 后取最大值就干净了。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                   # (B, H)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                     # (B, C)  未过 softmax 的 logits


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
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


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序 —— 课堂上看 zh/ja 那一格就够了。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="数据目录，默认 ../data；相对路径按脚本目录解析")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="",
                    help="保存模型到该路径，供 eval_langid.py 使用；相对路径按脚本目录解析")
    args = ap.parse_args()
    args.data_dir = resolve_path(args.data_dir)
    save_path = resolve_path(args.save) if args.save else ""

    missing = [n for n in ("train", "dev", "test")
               if not os.path.exists(os.path.join(args.data_dir, f"{n}.csv"))]
    if missing:
        raise SystemExit(
            f"数据目录里缺文件：{' '.join(m + '.csv' for m in missing)}\n"
            f"  当前数据目录：{args.data_dir}\n"
            f"  期望结构：week03/data/train.csv、dev.csv、test.csv\n"
            f"  可以用 --data-dir 指定别的目录（相对路径按脚本目录解析）")

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须和 LABELS 对得上，否则模型的输出格子根本不够用
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

    # 字符表只用训练集现算：dev/test 的字符一个都不能进来，否则等于提前看了考题
    vocab = build_char_vocab([t for t, _ in train_rows])
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（由训练集现算）")
    print(f"  <UNK> 占比：dev {unk_rate(dev_rows, vocab):.3%} / "
          f"test {unk_rate(test_rows, vocab):.3%}")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 测试集
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:4]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if save_path:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, save_path)
        print(f"\n模型已保存：{save_path}")


if __name__ == "__main__":
    main()

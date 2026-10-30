# -*- coding: utf-8 -*-
"""
train_langid_cls.py —— 多语种「语种识别」分类器（综合实战脚本 · 12 语种版）

任务：一句话 → 它是什么语言（12 分类）
模型：支持 4 种结构，默认 Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（12 语种）
运行：CPU 即可；耗时取决于设备与训练参数。

    python train_langid_cls.py --model rnn_layernorm_dropout --save langid_rnn_v12.pt
    python train_langid_cls.py --model embedding_layernorm_dropout --save langid_embedding_v12.pt
    python train_langid_cls.py --model rnn_layernorm --save langid_rnn_ln_v12.pt
    python train_langid_cls.py --model rnn_dropout --save langid_rnn_dropout_v12.pt
    python compare_models.py                                 # 训练并比较四种模型

【本课想让学生看到的四件事】
  1. 同一份代码、同一张字符表，能处理 12 种语言文字
  2. 字符级不需要分词器：中文按字、英文按字母、印地语按字母+元音标记，一视同仁
  3. 通过四种结构对比 RNN、LayerNorm、Dropout 对分类效果的影响
  4. 共享文字系统的语种（如 es ↔ pt）更容易互相混淆

【这份脚本是「12 类版」】
  · LABELS 列出 12 个语种，模型的输出维度 = len(LABELS)
  · 字符表读取 data/langid/vocab/vocab_v12.txt，由 v12/train.csv 生成

【字符表是怎么来的】见 build_char_vocab()：
  把训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号。
  已经生成好的表在 data/langid/vocab/vocab_v12.txt，每行「字符 \\t id \\t 出现次数」。
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
MAX_LEN = 40          # 统一截断到 40 个字符
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# v12 的 12 个语种；顺序固定，用于标签编号和模型输出维度
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]

DEFAULT_MODEL = "rnn_layernorm_dropout"
MODEL_CONFIGS = {
    "rnn_layernorm_dropout": {
        "name": "Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear",
        "use_rnn": True, "use_layer_norm": True, "use_dropout": True,
    },
    "embedding_layernorm_dropout": {
        "name": "Embedding → 掩码 MaxPool → LayerNorm → Dropout → Linear",
        "use_rnn": False, "use_layer_norm": True, "use_dropout": True,
    },
    "rnn_layernorm": {
        "name": "Embedding → RNN → 掩码 MaxPool → LayerNorm → Linear",
        "use_rnn": True, "use_layer_norm": True, "use_dropout": False,
    },
    "rnn_dropout": {
        "name": "Embedding → RNN → 掩码 MaxPool → Dropout → Linear",
        "use_rnn": True, "use_layer_norm": False, "use_dropout": True,
    },
}



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

    多语种的关键点：这张表不需要按语种分开建。汉字、假名、韩文、拉丁字母、
    西里尔字母、阿拉伯字母、天城文、泰文全都排在同一张表里。

    训练主流程没有直接调用它（字符表从 vocab_v12.txt 读取）；
    一旦换了训练集或加了语种，就要用它重新生成一张表，
    否则新语种的字符会全部落到 1 号 <UNK>，模型等于睁眼瞎。
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


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。

    默认读取 data/langid/vocab/vocab_v12.txt，它是从 v12 训练集统计出来的。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


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
    根据 model_type 组合 Embedding、可选 RNN、掩码 MaxPool、可选 LayerNorm/Dropout 和 Linear。

    MaxPool 前用 mask 排除补齐位置。无 RNN 结构直接对字符 Embedding 做池化；
    有 RNN 结构则对每个时间步的隐藏状态做池化。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                 model_type=DEFAULT_MODEL):
        super().__init__()
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"未知模型结构 {model_type!r}，可选：{', '.join(MODEL_CONFIGS)}")
        config = MODEL_CONFIGS[model_type]
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = (nn.RNN(embed_dim, hidden_dim, batch_first=True)
                    if config["use_rnn"] else None)
        feature_dim = hidden_dim if config["use_rnn"] else embed_dim
        self.norm = nn.LayerNorm(feature_dim) if config["use_layer_norm"] else None
        self.dropout = nn.Dropout(dropout) if config["use_dropout"] else None
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        h = self.embedding(x)
        if self.rnn is not None:
            h, _ = self.rnn(h)                     # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        if self.norm is not None:
            pooled = self.norm(pooled)
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        return self.fc(pooled)                     # (B, C)  未过 softmax 的 logits


def build_model(vocab_size, num_classes, model_type=DEFAULT_MODEL,
                embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
    """根据结构 ID 创建分类模型。"""
    return LangIDClassifier(vocab_size, num_classes, embed_dim, hidden_dim, dropout, model_type)


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
    ap.add_argument("--model", choices=list(MODEL_CONFIGS), default=DEFAULT_MODEL,
                    help="模型结构 ID（默认：rnn_layernorm_dropout）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表文件（默认用 v12 训练集生成的 vocab_v12.txt）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据必须包含 LABELS 中的 12 个语种，且不能有额外标签。
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    missing = [l for l in LABELS if l not in seen]
    if extra or missing:
        raise SystemExit(
            f"训练数据标签与 12 语种配置不匹配：额外标签={extra or '无'}，"
            f"缺失标签={missing or '无'}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
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

    vocab = load_vocab_file(args.vocab)
    print(f"模型结构：{args.model} — {MODEL_CONFIGS[args.model]['name']}")
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = build_model(len(vocab), num_classes, args.model)
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

    if args.save:
        save_dir = os.path.dirname(os.path.abspath(args.save))
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "model_type": args.model,
                    "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()

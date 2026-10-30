# -*- coding: utf-8 -*-
"""
train_langid_v12.py —— 12 语种识别训练

  1. 字符表： data/langid/vocab/vocab_v12.txt，
     build_char_vocab() 从 v12/train.csv 现场统计生成
     每行「字符 \t id \t 出现次数」，第 0 号 <PAD>、第 1 号 <UNK>。

  2. 数据目录：data/langid/v12

【运行】
    python train_langid_v12.py                        # 训练 + 测试集评测，存档 langid_rnn_v12.pt
    python train_langid_v12.py --epochs 12            # 多跑几轮
    python train_langid_v12.py --rebuild-vocab        # 强制重建字符表

小模型单线程最快，脚本默认 torch.set_num_threads(1)，不要改成 16。
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

# ─── 超参数 ──────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 数据清洗时长度窗口就是 8~40 字符，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
NUM_THREADS = 1       # 小模型单线程比多线程快，赋值1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")
MODEL_PATH = os.path.join(HERE, "langid_rnn_v12.pt")

# ─── 12 个语种 ───
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]

# 给报告/推理示例用的中文名，用于打印
LANG_NAMES = {
    "zh": "中文", "ja": "日语", "en": "英语", "es": "西班牙语",
    "pt": "葡萄牙语", "ru": "俄语", "ar": "阿拉伯语", "hi": "印地语",
    "ko": "韩语", "th": "泰语", "vi": "越南语", "tr": "土耳其语",
}


# ─── 1. 文本归一化，和建数据时一致 ────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    保留 M 是硬要求：印地语、泰语的元音是组合标记（कि 里的 ि、เก 里的 เ），
    只留字母会把它们砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 / 读写 ──────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里出现过的所有字符，每个字符一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序，保证可复现。

    12 个语种的汉字、假名、谚文、西里尔、阿拉伯、天城文、泰文、拉丁字母
    全部排在同一张表里 —— 这就是「字符级天然多语种」。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    """按数据卡约定的格式写出：字符 \t id \t 出现次数。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def coverage(vocab, freq, ratio=0.95):
    """前多少个字符覆盖 ratio 的字符出现次数（和数据卡里的统计口径一致）。"""
    total = sum(freq.values())
    acc, n = 0, 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq.get(ch, 0)
        if acc >= ratio * total:
            return n
    return n


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


# ─── 4. 模型（与课堂版完全一致）───────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前把补齐位 mask 成 -inf，保证「读过一串 0」的状态不参与池化。
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
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1) 补齐位为 False
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                   # (B, H)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                     # (B, C) 未过 softmax 的 logits


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


def print_confusion(mat, labels, top=5):
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    shown = 0
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        gname = f"{g}/{LANG_NAMES.get(g, '')}"
        pname = f"{p}/{LANG_NAMES.get(p, '')}"
        print(f"    {gname:<8} → {pname:<8} {cnt} 条")
        shown += 1
    if shown == 0:
        print("    （没有错误）")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default=MODEL_PATH)
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="忽略已有字符表，强制从训练集重新统计")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须和 LABELS 对得上，否则模型的输出格子根本不够用
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    missing = [l for l in LABELS if l not in seen]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
        )
    if missing:
        raise SystemExit(f"LABELS 里的这些语种在训练集里没有样本：{' '.join(missing)}")

    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # 字符表：已有就直接读；没有（或指定 --rebuild-vocab）就从 v12/train.csv 现算并写出
    if args.rebuild_vocab or not os.path.exists(args.vocab):
        print(f"字符表不存在或要求重建 → 从 {os.path.basename(args.data_dir)}/train.csv 统计")
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        save_vocab(vocab, freq, args.vocab)
        print(f"  已写出：{os.path.relpath(args.vocab, HERE)}")
    else:
        vocab = load_vocab_file(args.vocab)
        freq = None
        print(f"字符表：读取已有文件 {os.path.relpath(args.vocab, HERE)}")

    print(f"数据目录：{os.path.relpath(args.data_dir, HERE)}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：" +
          " ".join(f"{l}({LANG_NAMES.get(l, '?')})" for l in LABELS))
    coverage_note = f"，前 {coverage(vocab, freq)} 个覆盖 95%" if freq else ""
    print(f"  字符表 {len(vocab)} 个字符{coverage_note}")
    print(f"  随机猜的准确率基准：{1/num_classes:.2%}\n")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    best_acc, best_state = -1.0, None
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
        # 记录验证集最好的一轮，避免最后几轮轻微过拟合影响测试结果
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")
    print(f"验证集最佳准确率：{best_acc:.4f}")

    # 用验证集最优的那一轮参数做最终评测
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── 测试集 ──
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}({LANG_NAMES.get(lab, '')}): {hit}/{len(idx)} = "
              f"{hit/max(1, len(idx)):.3f}")

    # ── 推理示例：每个语种抽一条测试集句子 ──
    print("\n--- 推理示例（每个语种抽一条）---")
    model.eval()
    shown = set()
    demo_rows = []
    for text, lab in test_rows:
        if lab not in shown:
            shown.add(lab)
            demo_rows.append((text, lab))
    with torch.no_grad():
        for text, lab in demo_rows:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            mark = "✓" if LABELS[pid] == lab else "✗"
            print(f"  [{LABELS[pid]}({LANG_NAMES.get(LABELS[pid], '')}) "
                  f"{prob[pid]:.2f}] {mark} 真实={lab}  {text[:44]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{os.path.relpath(args.save, HERE)}")

        # 保存后立刻回读校验：.pt 是二进制 zip，一旦被编辑器/同步工具按文本处理
        # （字节被替换成 U+FFFD 再按 UTF-8 重写）就会变成打不开的坏档。
        # 这里当场加载一遍，坏了立刻报出来，而不是等到评测时才发现。
        try:
            ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
            ok = (ckpt["labels"] == LABELS and len(ckpt["vocab"]) == len(vocab))
        except Exception as e:                       # noqa: BLE001
            raise SystemExit(f"模型保存后校验失败：{args.save}\n  {type(e).__name__}: {e}")
        if not ok:
            raise SystemExit(f"模型保存后校验失败（内容不匹配）：{args.save}")
        size = os.path.getsize(args.save)
        print(f"  回读校验通过（{size:,} 字节）")


if __name__ == "__main__":
    main()

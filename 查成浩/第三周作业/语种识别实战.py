import torch
import torch.nn as nn
import os
import argparse
import csv
import random
import time
from collections import Counter, OrderedDict
import unicodedata
from torch.utils.data import Dataset, DataLoader

#-------------超参数设置----------
SEED = 42
MAX_LEN = 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1

curfilePath = os.path.dirname(os.path.abspath(__file__))
train_filepath = os.path.join(curfilePath, 'data', 'v12', "train.csv")
dev_filepath = os.path.join(curfilePath, 'data', 'v12', "dev.csv")
test_filepath = os.path.join(curfilePath, 'data', 'v12', "test.csv")
vocab_filepath = os.path.join(curfilePath, 'data', 'v12', "vocab.txt")

labels_base = ['ar', 'en', 'es', 'hi', 'ja', 'ko', 'pt', 'ru', 'th', 'tr', 'vi', 'zh']


def load_data(path: str) -> list:
    ret = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in csv.DictReader(f):
                ret.append((line['text'], line['label']))
    return ret


def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


def load_vocab_file(path):
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def build_char_vocab(sentences, min_freq=1):
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)


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


def print_confusion(mat, labels, top=3):
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--thread', type=int, default=NUM_THREADS)
    arg.add_argument('--train_file_path', type=str, default=train_filepath, help='训练数据路径')
    arg.add_argument('--dev_file_path', type=str, default=dev_filepath, help='验证集数据路径')
    arg.add_argument('--test_file_path', type=str, default=test_filepath, help='测试集数据路径')
    arg.add_argument('--savepath', default='', help='模型保存路径，空串表示不保存')
    arg.add_argument('--epochs', type=int, default=EPOCHS)
    arg.add_argument('--per_class', type=int, default=0)
    arg.add_argument('--vocab', default=vocab_filepath)
    avvg = arg.parse_args()

    torch.set_num_threads(avvg.thread)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_data = load_data(avvg.train_file_path)
    dev_data = load_data(avvg.dev_file_path)
    test_data = load_data(avvg.test_file_path)

    labels = sorted({lables for _, lables in train_data})
    external_label = [l for l in labels if l not in labels_base]
    if external_label:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(external_label)}\n"
            f"  当前 LABELS（{len(labels_base)} 类）：{' '.join(labels_base)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    label_id = {l: i for i, l in enumerate(labels_base)}
    label_len = len(label_id)

    if avvg.per_class:
        by_label = {}
        for t, l in train_data:
            by_label.setdefault(l, []).append(t)
        train_data = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(avvg.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_data)

    if os.path.isfile(avvg.vocab):
        vocab = load_vocab_file(avvg.vocab)
    else:
        print(f"字符表不存在，从训练集生成：{avvg.vocab}")
        vocab = build_char_vocab([t for t, _ in train_data])
        with open(avvg.vocab, "w", encoding="utf-8") as f:
            f.write("# char\tid\tfreq\n")
            for ch, i in vocab.items():
                f.write(f"{ch}\t{i}\t0\n")

    print(f"训练 {len(train_data)} 条 / 验证 {len(dev_data)} 条 / 测试 {len(test_data)} 条")
    print(f"  类别 {label_len} 个：{' '.join(labels_base)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(avvg.vocab)}）")
    print(f"  随机猜的准确率基准：{1/label_len:.1%}")

    train_loader = DataLoader(TextDataset(train_data, vocab, label_id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_data, vocab, label_id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_data, vocab, label_id), batch_size=128)

    model = LangIDClassifier(len(vocab), label_len)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, avvg.epochs + 1):
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
        print(f"Epoch {epoch:2d}/{avvg.epochs}  loss={total/len(train_loader):.4f}  dev_acc={acc:.4f}")
    print(f"\n训练耗时：{time.time()-t0:.0f} 秒")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, label_len)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, labels_base), labels_base)

    print("\n  各语种准确率：")
    for c, lab in enumerate(labels_base):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_data[:4]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{labels_base[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if avvg.savepath:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": labels_base, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, avvg.savepath)
        print(f"\n模型已保存：{avvg.savepath}")


if __name__ == '__main__':
    main()

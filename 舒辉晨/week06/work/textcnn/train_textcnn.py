"""
方法三：TextCNN 从零训练（L2.5，不加载预训练权重）

为什么要加这一路：
  原项目只有「规则 / TF-IDF / 微调」三档，中间缺一个「用神经网络但不借预训练」的对照组。
  有了它，「BERT 那 5 个点到底是来自预训练还是来自非线性」才有答案。

配置刻意压低：字级 embedding 128 维、卷积核 2/3/4 各 128 个、最多 256 字。
这不是 SOTA，是「一台 8G 笔记本 10 分钟内能跑完」的合理规模。
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits
from common.labels import CLASSES
from common.metrics import classification_report, to_markdown_table

ROOT = Path(__file__).resolve().parent.parent
PAD, UNK = 0, 1


def build_vocab(texts, max_size=8000):
    c = Counter()
    for t in texts:
        c.update(t)
    chars = [ch for ch, _ in c.most_common(max_size)]
    return {ch: i + 2 for i, ch in enumerate(chars)}


class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data, self.vocab, self.max_len = data, vocab, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        ids = [self.vocab.get(ch, UNK) for ch in d["text"][: self.max_len]]
        ids += [PAD] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(d["label"])


class TextCNN(nn.Module):
    """embedding → 多核宽卷积 → 时间维 max-pool → 拼接 → 全连接"""

    def __init__(self, vocab_size, num_classes=10, emb=128, filters=128, kernels=(2, 3, 4), dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernels), num_classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                  # [B, emb, L]
        feats = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    ys, ps, probs = [], [], []
    for ids, y in loader:
        logits = model(ids.to(device))
        p = torch.softmax(logits.float(), -1)
        ps.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
        ys.extend(y.tolist())
    return {"acc": round(float(accuracy_score(ys, ps)), 4),
            "macro_f1": round(float(f1_score(ys, ps, average="macro")), 4)}, ps, ys, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--filters", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--n_train", type=int, default=0, help=">0 时只用前 N 条训练（数据量实验）")
    ap.add_argument("--out", default="outputs/textcnn_pred.json")
    ap.add_argument("--tag", default="textcnn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, meta = load_splits()
    if args.n_train:
        train = train[: args.n_train]

    vocab = build_vocab([d["text"] for d in train], args.vocab_size)
    train_dl = DataLoader(TextDataset(train, vocab, args.max_len), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TextDataset(val, vocab, args.max_len), batch_size=256)

    model = TextCNN(len(vocab) + 2, len(CLASSES), args.emb, args.filters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[cfg] TextCNN 参数量 {n_param/1e6:.2f}M | vocab {len(vocab)} | 训练 {len(train)} 条 | "
          f"epochs {args.epochs} | device {device}", flush=True)

    log, best = [], 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for ids, y in train_dl:
            ids, y = ids.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(ids), y)
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        res, ps, ys, probs = evaluate(model, val_dl, device)
        log.append({"epoch": ep, "train_loss": round(run / max(1, nb), 4),
                    "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                    "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {ep}: loss={log[-1]['train_loss']} val_acc={res['acc']} "
              f"val_f1={res['macro_f1']} ({log[-1]['elapsed_s']}s)", flush=True)
        best = max(best, res["macro_f1"])

    rep = classification_report(ys, ps)
    rep.update({"method": args.tag, "params_m": round(n_param / 1e6, 2),
                "epochs": args.epochs, "n_train": len(train),
                "elapsed_s": round(time.time() - t0, 1), "best_val_f1": best})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {**meta, "n_eval": len(ys)}, "report": rep,
                               "log": log,
                               "preds": [{"text": d["text"][:60], "true": y, "pred": p}
                                         for d, y, p in zip(val, ys, ps)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown_table({f"TextCNN（{n_param/1e6:.1f}M 参数）": rep}))
    print(f"→ 写入 {out}")


if __name__ == "__main__":
    main()

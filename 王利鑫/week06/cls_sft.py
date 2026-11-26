"""
基于中文预训练模型的二分类情感 SFT 全量微调（水果评论：1=正向，0=负向）。

支持模型（本地目录，不联网下载）：
    --model roberta   -> chinese-roberta-wwm-ext
    --model macbert   -> chinese-macbert-base

特性：
    1. 全量微调，差分学习率（embedding 小 / encoder 中 / 分类头大），防止遗忘预训练知识
    2. warmup（前期稳定）+ 余弦退火（后期收敛平滑）
    3. Focal Loss（关注难样本）
    4. cls 池化
    5. fp16 混合精度加速（3060Ti 不支持 fp8）
    6. 实时折线图展示 train/val loss + val 准确率，tqdm 进度条

用法：
    python cls_sft.py --model roberta
    python cls_sft.py --model macbert

    --model_dir 默认 C:/task/bert-cl/models；也可用 --model_path 直接给完整路径。

数据格式：data/train.csv、data/test.csv，列 review,label（label 0/1）。
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_MODEL_DIR = Path("C:/task/bert-cl/models")   # 远程训练机模型目录，可用 --model_dir 覆盖
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

MODEL_PRESETS = {
    "roberta": "chinese-roberta-wwm-ext",
    "macbert": "chinese-macbert-base",
}


# ---------------------------------------------------------------------------
# Focal Loss（分类，无 ignore_index）
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha.to(logits.device)[targets] * focal
        return focal.mean()


# ---------------------------------------------------------------------------
# 模型：backbone + cls 池化 + 分类头
# ---------------------------------------------------------------------------
class ClsClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.backbone.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]          # cls 池化
        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
class ClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        enc = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True)
        self.input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i],
                "labels": self.labels[i]}


def load_csv(path: Path):
    """读 review,label 两列，返回 (texts, label_ids, label2id, id2label)。"""
    texts, raw_labels = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            texts.append((row["review"] or "").strip())
            raw_labels.append((row["label"] or "").strip())
    unique = sorted(set(raw_labels))
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return texts, [label2id[l] for l in raw_labels], label2id, id2label


def stratified_split(texts, labels, val_ratio, seed):
    """按 label 分层切 train/val。"""
    from collections import defaultdict
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[lab].append(i)
    tr, va = [], []
    for lab, idxs in by_label.items():
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_ratio)
        va.extend(idxs[:n_val])
        tr.extend(idxs[n_val:])
    rng.shuffle(tr)
    rng.shuffle(va)
    return ([texts[i] for i in tr], [labels[i] for i in tr],
            [texts[i] for i in va], [labels[i] for i in va])


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device,
                    amp_enabled, epoch, num_epochs):
    model.train()
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} train", leave=False, ncols=100)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(input_ids, mask)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         avg=f"{total_loss / max(total_n, 1):.4f}",
                         lr=f"{scheduler.get_last_lr()[-1]:.2e}")
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_enabled):
    model.eval()
    total_loss, total_n, correct = 0.0, 0, 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False, ncols=100):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(input_ids, mask)
            loss = loss_fn(logits, labels)
        preds = logits.argmax(-1)

        correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    return total_loss / max(total_n, 1), acc, preds, labels


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="中文二分类情感 SFT 微调")
    ap.add_argument("--model", type=str, default="roberta", choices=list(MODEL_PRESETS))
    ap.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--emb_lr_ratio", type=float, default=0.1)
    ap.add_argument("--enc_lr_ratio", type=float, default=1.0)
    ap.add_argument("--head_lr_ratio", type=float, default=10.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--no_focal", action="store_true")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.fp16 and device.type == "cuda"

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = str(Path(args.model_dir) / MODEL_PRESETS[args.model])
    if not Path(model_path).exists():
        print(f"[错误] 找不到模型目录: {model_path}")
        print(f"       请用 --model_dir 或 --model_path 指定本地模型路径。")
        return

    data_dir = Path(args.data_dir)
    run_dir = Path(args.output_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"模型: {args.model} -> {model_path}")
    print(f"device={device}  fp16={amp_enabled}  focal={not args.no_focal}(gamma={args.focal_gamma})")
    print(f"差分学习率: emb={args.lr * args.emb_lr_ratio:.2e}  "
          f"enc={args.lr * args.enc_lr_ratio:.2e}  head={args.lr * args.head_lr_ratio:.2e}")
    print("=" * 70)

    # ---- 读数据 ----
    train_texts, train_labels, label2id, id2label = load_csv(data_dir / "train.csv")
    test_texts, test_labels, _, _ = load_csv(data_dir / "test.csv")
    num_labels = len(label2id)
    print(f"类别数 = {num_labels}，label 映射 = {label2id}")
    print(f"train {len(train_texts)} 条，test {len(test_texts)} 条")

    tr_texts, tr_labels, va_texts, va_labels = stratified_split(
        train_texts, train_labels, args.val_ratio, args.seed)
    print(f"划分后 train/val = {len(tr_texts)}/{len(va_texts)}")

    # ---- tokenizer / 模型 ----
    print("加载 tokenizer + 模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ClsClassifier(model_path, num_labels).to(device)

    train_ds = ClsDataset(tr_texts, tr_labels, tokenizer, args.max_length)
    val_ds = ClsDataset(va_texts, va_labels, tokenizer, args.max_length)
    test_ds = ClsDataset(test_texts, test_labels, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"每 epoch {len(train_loader)} 个 batch")

    # ---- Loss ----
    if args.no_focal:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = FocalLoss(gamma=args.focal_gamma)

    # ---- 差分学习率优化器 ----
    no_decay = ["bias", "LayerNorm.weight"]
    emb, enc, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.embeddings."):
            emb.append((name, p))
        elif name.startswith("backbone.encoder."):
            enc.append((name, p))
        else:
            head.append((name, p))

    def group_of(tuples, lr):
        decay = [p for n, p in tuples if not any(nd in n for nd in no_decay)]
        no_dec = [p for n, p in tuples if any(nd in n for nd in no_decay)]
        g = []
        if decay:
            g.append({"params": decay, "lr": lr, "weight_decay": args.weight_decay})
        if no_dec:
            g.append({"params": no_dec, "lr": lr, "weight_decay": 0.0})
        return g

    optimizer = torch.optim.AdamW(
        group_of(emb, args.lr * args.emb_lr_ratio)
        + group_of(enc, args.lr * args.enc_lr_ratio)
        + group_of(head, args.lr * args.head_lr_ratio)
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ---- 实时折线图 ----
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    is_live = plt.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo")
    if is_live:
        plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    tr_losses, va_losses, va_accs = [], [], []

    # ---- 训练 ----
    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler,
                                     scaler, device, amp_enabled, epoch, args.epochs)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device, amp_enabled)
        print(f"epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        tr_losses.append(train_loss)
        va_losses.append(val_loss)
        va_accs.append(val_acc)

        xs = list(range(1, epoch + 1))
        ax1.clear()
        ax1.plot(xs, tr_losses, "o-", label="train loss")
        ax1.plot(xs, va_losses, "s-", label="val loss")
        ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.clear()
        ax2.plot(xs, va_accs, "o-", label="val acc")
        ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylabel("acc")
        ax2.legend(); ax2.grid(alpha=0.3)
        if is_live:
            plt.draw(); plt.pause(0.001)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    fig.tight_layout()
    curves_path = run_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"\n训练曲线已保存: {curves_path}")

    # ---- 测试集评估 ----
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, loss_fn, device, amp_enabled)

    # 混淆矩阵（二分类，手动算，不依赖 sklearn）
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t][p] += 1

    print("\n" + "=" * 70)
    print(f"测试集结果 ({args.model})")
    print(f"  准确率: {test_acc:.4f}")
    print("  混淆矩阵 (行=真实, 列=预测):")
    header = "        " + " ".join(f"{id2label[j]:>8}" for j in range(num_labels))
    print(header)
    for i in range(num_labels):
        print(f"  {id2label[i]:>4}  " + " ".join(f"{cm[i][j]:>8}" for j in range(num_labels)))
    print("=" * 70)

    # ---- 保存 ----
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    with open(run_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()},
                   "model": args.model, "model_path": model_path}, f, ensure_ascii=False, indent=2)
    metrics = {
        "model": args.model, "model_path": model_path,
        "test_acc": test_acc, "test_loss": test_loss,
        "best_val_acc": best_acc,
        "confusion_matrix": cm.tolist(), "config": vars(args),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {run_dir}/  (best_model.pt / label2id.json / metrics.json / training_curves.png)")


if __name__ == "__main__":
    main()

"""
基于中文预训练模型的 NER（命名实体识别 / 数据抽取）SFT 全量微调。

支持模型（本地目录，不联网下载）：
    --model roberta   -> chinese-roberta-wwm-ext
    --model macbert   -> chinese-macbert-base

特性：
    1. 全量微调，差分学习率（embedding 小 / encoder 中 / 分类头大），防止遗忘预训练知识
    2. warmup（前期稳定）+ 余弦退火（后期收敛平滑）
    3. Focal Loss（关注难样本，缓解 O 占 80% 的类别失衡）
    4. 混合精度 fp16 加速（3060Ti 不支持 fp8，fp8 需要 H100/Hopper）
    5. 实时折线图展示 train/val loss + val 准确率/F1，tqdm 进度条

用法：
    python ner_sft.py --model roberta --model_dir <你的模型目录>
    python ner_sft.py --model macbert --model_dir <你的模型目录>

    --model_dir 是「包含模型文件夹的父目录」，模型子文件夹名默认取
    chinese-roberta-wwm-ext / chinese-macbert-base；也可用 --model_path 直接给完整路径。

数据格式（BIO 序列标注）：每行 "字符<TAB>标签"，空行分隔句子。
"""
import argparse
import json
import random
from collections import Counter
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
# Focal Loss（带 ignore_index，padding/special 位置不参与）
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha            # [num_labels] 或 None
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [B, T, C], targets: [B, T]
        B, T, C = logits.shape
        logits = logits.reshape(-1, C)
        targets = targets.reshape(-1)

        ce = nn.functional.cross_entropy(logits, targets, reduction="none",
                                         ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            safe = targets.clamp(min=0)
            alpha_t = self.alpha.to(logits.device)[safe]
            alpha_t = alpha_t * (targets != self.ignore_index).float()
            focal = alpha_t * focal

        valid = (targets != self.ignore_index).sum()
        if valid == 0:
            return focal.sum() * 0.0
        return focal.sum() / valid


# ---------------------------------------------------------------------------
# 模型：backbone + dropout + 分类头
# ---------------------------------------------------------------------------
class TokenClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.backbone.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_bio_file(path: Path):
    """读 BIO 文件，返回 [(字符列表, 标签列表), ...]（按空行分句）。"""
    sentences = []
    chars, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if chars:
                    sentences.append((chars, labels))
                    chars, labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    chars.append(parts[0])
                    labels.append(parts[1])
    if chars:
        sentences.append((chars, labels))
    return sentences


class NERDataset(Dataset):
    """把 BIO 句子的字符级标签对齐到 token（子词只保留首 token 的标签）。"""

    def __init__(self, sentences, tokenizer, label2id, max_length):
        input_ids, attention_mask, labels = [], [], []
        for chars, labs in sentences:
            enc = tokenizer(
                chars,                       # 每个字当做一个 word
                is_split_into_words=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            word_ids = enc.word_ids()
            aligned, prev = [], None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)         # [CLS]/[SEP]/padding
                elif wid != prev:
                    aligned.append(label2id[labs[wid]])
                else:
                    aligned.append(-100)         # 子词后续部分忽略
                prev = wid
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            labels.append(aligned)

        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


# ---------------------------------------------------------------------------
# 实体级 F1（NER 不能只看 token 准确率）
# ---------------------------------------------------------------------------
def seq_to_entities(tag_ids, id2label):
    """把一串 tag id 转成 {(类型, 起, 止)} 实体集合。"""
    ents, cur_type, cur_start = set(), None, None
    for i, tid in enumerate(tag_ids):
        lab = id2label[tid]
        if lab.startswith("B-"):
            if cur_type is not None:
                ents.add((cur_type, cur_start, i - 1))
            cur_type, cur_start = lab[2:], i
        elif lab.startswith("I-"):
            if cur_type != lab[2:]:
                if cur_type is not None:
                    ents.add((cur_type, cur_start, i - 1))
                cur_type, cur_start = None, None
        else:  # O
            if cur_type is not None:
                ents.add((cur_type, cur_start, i - 1))
                cur_type, cur_start = None, None
    if cur_type is not None:
        ents.add((cur_type, cur_start, len(tag_ids) - 1))
    return ents


def compute_entity_f1(all_preds, all_labels, id2label):
    """all_preds/all_labels: 每句的 tag id 列表（含 -100）。返回 (P, R, F1)。"""
    tp = fp = fn = 0
    for pred, gold in zip(all_preds, all_labels):
        valid = [i for i in range(len(gold)) if gold[i] != -100]
        p_ents = seq_to_entities([pred[i] for i in valid], id2label)
        g_ents = seq_to_entities([gold[i] for i in valid], id2label)
        tp += len(p_ents & g_ents)
        fp += len(p_ents - g_ents)
        fn += len(g_ents - p_ents)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler,
                    device, amp_enabled, epoch, num_epochs):
    model.train()
    total_loss, total_tokens = 0.0, 0
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

        valid = (labels != -100).sum().item()
        total_loss += loss.item() * valid
        total_tokens += valid
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         avg=f"{total_loss / max(total_tokens, 1):.4f}",
                         lr=f"{scheduler.get_last_lr()[-1]:.2e}")
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_enabled, id2label):
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False, ncols=100):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(input_ids, mask)
            loss = loss_fn(logits, labels)
        preds = logits.argmax(-1)

        valid = (labels != -100)
        correct += ((preds == labels) & valid).sum().item()
        n_valid = valid.sum().item()
        total_loss += loss.item() * n_valid
        total_tokens += n_valid

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / max(total_tokens, 1)
    avg_loss = total_loss / max(total_tokens, 1)
    p, r, f1 = compute_entity_f1(all_preds, all_labels, id2label)
    return avg_loss, acc, p, r, f1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="中文 NER SFT 全量微调")
    # 模型
    ap.add_argument("--model", type=str, default="roberta", choices=list(MODEL_PRESETS),
                    help="模型预设: roberta / macbert")
    ap.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR),
                    help="模型所在父目录（含 chinese-roberta-wwm-ext / chinese-macbert-base 子文件夹）")
    ap.add_argument("--model_path", type=str, default="",
                    help="直接指定模型完整路径（优先于 --model_dir/--model）")
    # 数据 / 输出
    ap.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    # 训练
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5, help="encoder 基础学习率")
    ap.add_argument("--emb_lr_ratio", type=float, default=0.1, help="embedding 学习率 = lr * 该值")
    ap.add_argument("--enc_lr_ratio", type=float, default=1.0, help="encoder 学习率 = lr * 该值")
    ap.add_argument("--head_lr_ratio", type=float, default=10.0, help="分类头学习率 = lr * 该值")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    # loss
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--no_focal", action="store_true", help="关闭 focal loss，用普通交叉熵")
    ap.add_argument("--class_weight", action="store_true", help="给实体类别加温和的类别权重")
    # 精度
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    # ---- 随机种子 ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.fp16 and device.type == "cuda"

    # ---- 解析模型路径（本地，不联网）----
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = str(Path(args.model_dir) / MODEL_PRESETS[args.model])
    if not Path(model_path).exists():
        print(f"[错误] 找不到模型目录: {model_path}")
        print(f"       请用 --model_dir 或 --model_path 指定本地模型路径。")
        return

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"模型: {args.model} -> {model_path}")
    print(f"device={device}  fp16={amp_enabled}  focal={not args.no_focal}(gamma={args.focal_gamma})")
    print(f"差分学习率: emb={args.lr * args.emb_lr_ratio:.2e}  "
          f"enc={args.lr * args.enc_lr_ratio:.2e}  head={args.lr * args.head_lr_ratio:.2e}")
    print("=" * 70)

    # ---- 读数据 + 建 label 映射 ----
    print("读取数据 ...")
    train_sents = load_bio_file(data_dir / "train.txt")
    dev_sents = load_bio_file(data_dir / "dev.txt")
    test_sents = load_bio_file(data_dir / "test.txt")
    print(f"句子数  train/dev/test = {len(train_sents)}/{len(dev_sents)}/{len(test_sents)}")

    all_label_counts = Counter()
    for _, labs in train_sents:
        all_label_counts.update(labs)
    labels = ["O"] + sorted([l for l in all_label_counts if l != "O"])
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print(f"标签数 = {len(labels)}（含 O），前几个: {labels[:8]} ...")

    # ---- tokenizer / 数据集 ----
    print("加载 tokenizer + 模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = TokenClassifier(model_path, len(labels)).to(device)

    train_ds = NERDataset(train_sents, tokenizer, label2id, args.max_length)
    dev_ds = NERDataset(dev_sents, tokenizer, label2id, args.max_length)
    test_ds = NERDataset(test_sents, tokenizer, label2id, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"每 epoch {len(train_loader)} 个 batch")

    # ---- Loss ----
    alpha = None
    if args.class_weight:
        counts = torch.tensor([all_label_counts.get(l, 1) for l in labels], dtype=torch.float32)
        # 开方反频率，压到 [0.2, 5] 之间，避免稀有类权重爆炸
        w = (counts.sum() / (len(labels) * counts)) ** 0.5
        w = w.clamp(0.2, 5.0)
        alpha = w.to(device)
        print("类别权重:", {labels[i]: round(float(w[i]), 2) for i in range(len(labels)) if w[i] > 1.0})
    if args.no_focal:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=alpha)
    else:
        loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=alpha, ignore_index=-100)

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
    tr_losses, va_losses, va_accs, va_f1s = [], [], [], []

    # ---- 训练 ----
    best_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler,
                                     scaler, device, amp_enabled, epoch, args.epochs)
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(
            model, dev_loader, loss_fn, device, amp_enabled, id2label)
        print(f"epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  "
              f"val_F1 {val_f1:.4f} (P {val_p:.4f} R {val_r:.4f})")

        tr_losses.append(train_loss)
        va_losses.append(val_loss)
        va_accs.append(val_acc)
        va_f1s.append(val_f1)

        # 更新折线图
        xs = list(range(1, epoch + 1))
        ax1.clear()
        ax1.plot(xs, tr_losses, "o-", label="train loss")
        ax1.plot(xs, va_losses, "s-", label="val loss")
        ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.clear()
        ax2.plot(xs, va_accs, "o-", label="val acc")
        ax2.plot(xs, va_f1s, "s-", label="val F1")
        ax2.set_title("Accuracy / F1"); ax2.set_xlabel("epoch"); ax2.set_ylabel("score")
        ax2.legend(); ax2.grid(alpha=0.3)
        if is_live:
            plt.draw(); plt.pause(0.001)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    fig.tight_layout()
    curves_path = run_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"\n训练曲线已保存: {curves_path}")

    # ---- 测试集评估（最优模型）----
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, test_p, test_r, test_f1 = evaluate(
        model, test_loader, loss_fn, device, amp_enabled, id2label)

    print("\n" + "=" * 70)
    print(f"测试集结果 ({args.model})")
    print(f"  token 准确率: {test_acc:.4f}")
    print(f"  实体级 P/R/F1: {test_p:.4f} / {test_r:.4f} / {test_f1:.4f}")
    print("=" * 70)

    # ---- 保存 ----
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    with open(run_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()},
                   "model": args.model, "model_path": model_path},
                  f, ensure_ascii=False, indent=2)
    metrics = {
        "model": args.model, "model_path": model_path,
        "test_acc": test_acc, "test_P": test_p, "test_R": test_r, "test_F1": test_f1,
        "best_val_F1": best_f1, "config": vars(args),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {run_dir}/  (best_model.pt / label2id.json / metrics.json / training_curves.png)")


if __name__ == "__main__":
    main()

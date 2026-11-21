"""
基于 XLM-RoBERTa 的三分类文本分类 SFT 微调。

数据：data/train.csv、data/test.csv（两列：评价 / 结果，标签 = 好|中|差）
任务：把一段"玉米田间评价"文本分成 好 / 中 / 差 三类。

本脚本支持 5 种优化手段，全部通过命令行参数切换（可单独使用，也可组合）：

  1) 类别加权      --class_weight          按类别样本数做 balanced 加权
  2) 标签平滑      --label_smoothing 0.1    soft label，缓解过拟合
  3) Focal Loss    --focal_loss --focal_gamma 2.0   关注难分样本、抑制易分样本
  4) 池化方式      --pooling cls|mean|max   取 [CLS] / 平均 / 最大 三种池化
  5) FGM 对抗训练  --fgm --fgm_epsilon 1.0  对词嵌入加梯度扰动再训练，提升鲁棒性

每次运行都会在测试集上打印并保存：准确率 + 混淆矩阵 + 分类报告。

用法示例（在 GPU 机器上，先 pip install -r requirements.txt）：

  # 下载模型慢/失败（国内）时，先设置 HF 镜像再运行：
  #   export HF_ENDPOINT=https://hf-mirror.com

  # baseline（cls 池化 + 普通交叉熵）
  python xlm-roberta-sft.py

  # 1. 类别加权
  python xlm-roberta-sft.py --class_weight

  # 2. 标签平滑
  python xlm-roberta-sft.py --label_smoothing 0.1

  # 3. Focal Loss
  python xlm-roberta-sft.py --focal_loss --focal_gamma 2.0

  # 4. 三种池化
  python xlm-roberta-sft.py --pooling mean
  python xlm-roberta-sft.py --pooling max

  # 5. FGM 对抗训练
  python xlm-roberta-sft.py --fgm --fgm_epsilon 1.0

  # 一键跑全部配置并汇总结果：
  #   Windows:    run_all.bat
  #   Linux/macOS:  bash run_all.sh

其它常用参数：
  --model_name xlm-roberta-base   # 也可填本地路径或镜像缓存路径
  --epochs 3  --batch_size 16  --lr 2e-5  --max_length 256
  --val_ratio 0.1                # 从 train.csv 划出验证集用于早停
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    XLMRobertaModel,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

LABELS = ["好", "中", "差"]                      # 分类顺序，也是混淆矩阵行列顺序
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)


# ---------------------------------------------------------------------------
# 自定义 Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss：降低易分样本的 loss 权重，让模型更关注难分样本。

    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha            # [num_classes]，为 None 时不做类别加权
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)                              # 预测为真实类的概率
        focal = ((1.0 - p_t) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)[targets]
            focal = alpha * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """balanced 类别权重：total / (num_classes * count)，少数类权重更高。"""
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)                        # 防止某类为 0
    return counts.sum() / (num_classes * counts)


# ---------------------------------------------------------------------------
# FGM 对抗训练
# ---------------------------------------------------------------------------
class FGM:
    """Fast Gradient Method：沿梯度方向给词嵌入加扰动，做对抗训练。"""

    def __init__(self, model, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup, f"{name} 没有备份，无法恢复"
                param.data = self.backup[name]
        self.backup = {}


# ---------------------------------------------------------------------------
# 模型：XLM-RoBERTa backbone + 可选池化 + 分类头
# ---------------------------------------------------------------------------
class XLMRobertaClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pooling: str = "cls"):
        super().__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.pooling = pooling
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state              # [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()     # [B, T, 1]

        if self.pooling == "cls":
            pooled = hidden[:, 0, :]                    # [CLS] 向量
        elif self.pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "max":
            # padding 位置置为极小值，再取 max，只统计真实 token
            pooled = (hidden * mask + (1.0 - mask) * -1e4).max(dim=1).values
        else:
            raise ValueError(f"不支持的池化方式: {self.pooling}（可选 cls/mean/max）")

        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.input_ids, self.attention_mask = self._tokenize(texts, tokenizer, max_length)
        self.labels = torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _tokenize(texts, tokenizer, max_length):
        """分块 tokenize，避免一次把 9 万条文本全塞给 tokenizer。"""
        input_ids, attention_mask = [], []
        chunk = 5000
        for i in range(0, len(texts), chunk):
            enc = tokenizer(
                texts[i:i + chunk],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            input_ids.extend(enc["input_ids"])
            attention_mask.extend(enc["attention_mask"])
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def load_csv(path: Path):
    """读取 CSV，返回 (texts, label_ids)。自动去掉 BOM。"""
    import csv
    texts, labels = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row["评价"] or "").strip()
            label = (row["结果"] or "").strip()
            if label not in LABEL2ID:
                continue                       # 跳过异常标签行
            texts.append(text)
            labels.append(LABEL2ID[label])
    return texts, labels


def stratified_split(texts, labels, val_ratio, seed):
    """按类别分层切分，保证训练/验证类别分布一致。"""
    rng = random.Random(seed)
    idx_by_label = {i: [] for i in range(NUM_CLASSES)}
    for i, lab in enumerate(labels):
        idx_by_label[lab].append(i)

    train_idx, val_idx = [], []
    for lab in idx_by_label:
        idxs = idx_by_label[lab]
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_ratio)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (
        [texts[i] for i in train_idx], [labels[i] for i in train_idx],
        [texts[i] for i in val_idx], [labels[i] for i in val_idx],
    )


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, fgm, device,
                    epoch, num_epochs):
    model.train()
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} train",
                leave=False, ncols=100)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()

        if fgm is not None:
            fgm.attack()                              # 加扰动
            logits_adv = model(input_ids, attention_mask)
            loss_adv = loss_fn(logits_adv, labels)
            loss_adv.backward()
            fgm.restore()                             # 恢复原词嵌入

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         avg=f"{total_loss / total_n:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / total_n if total_n else 0.0


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, desc="eval"):
    """返回 (平均 loss, 准确率, 预测, 真实标签)。"""
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    return total_loss / total_n if total_n else 0.0, acc, preds, labels


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------
def format_confusion_matrix(cm):
    header = "真实\\预测 " + " ".join(f"{l:>6}" for l in LABELS)
    lines = [header]
    for i, l in enumerate(LABELS):
        lines.append(f"  {l:>6}   " + " ".join(f"{v:>6}" for v in cm[i]))
    return "\n".join(lines)


def save_results(run_dir, run_name, args, acc, cm, report_dict, loss):
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. 混淆矩阵 CSV（可用 Excel 打开）
    with open(run_dir / "confusion_matrix.csv", "w", encoding="utf-8") as f:
        f.write("真实\\预测," + ",".join(LABELS) + "\n")
        for i, l in enumerate(LABELS):
            f.write(l + "," + ",".join(str(v) for v in cm[i]) + "\n")

    # 2. 指标 JSON
    metrics = {
        "run_name": run_name,
        "test_accuracy": acc,
        "test_loss": loss,
        "num_classes": NUM_CLASSES,
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "config": vars(args),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 3. 混淆矩阵热力图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES), LABELS)
        ax.set_yticks(range(NUM_CLASSES), LABELS)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.set_title(f"{run_name}  测试集准确率 {acc:.4f}")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # 字体/无显示环境也不影响主流程
        print(f"  (混淆矩阵热力图保存失败，可忽略: {e})")


def build_run_name(args):
    parts = []
    if args.focal_loss:
        parts.append(f"focal_g{args.focal_gamma}")
    if args.class_weight:
        parts.append("clsweight")
    if args.label_smoothing > 0:
        parts.append(f"ls{args.label_smoothing}")
    if args.pooling != "cls":
        parts.append(f"pool_{args.pooling}")
    if args.fgm:
        parts.append(f"fgm_e{args.fgm_epsilon}")
    return "_".join(parts) if parts else "baseline"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="XLM-RoBERTa 三分类 SFT 微调")
    # 数据与模型
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    # 训练超参
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2, help="早停耐心（0=关闭）")
    parser.add_argument("--seed", type=int, default=42)
    # 5 种优化手段
    parser.add_argument("--class_weight", action="store_true", help="1) 类别加权")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="2) 标签平滑系数(如 0.1)")
    parser.add_argument("--focal_loss", action="store_true", help="3) 使用 Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss 的 gamma")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "max"], help="4) 池化方式")
    parser.add_argument("--fgm", action="store_true", help="5) 使用 FGM 对抗训练")
    parser.add_argument("--fgm_epsilon", type=float, default=1.0)
    parser.add_argument("--fgm_emb", type=str, default="word_embeddings")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    # ---- 随机种子 ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    run_name = build_run_name(args)
    run_dir = output_dir / run_name
    print("=" * 70)
    print(f"运行配置: {run_name}   device={device}")
    print(f"  类别加权={args.class_weight}  标签平滑={args.label_smoothing}  "
          f"focal_loss={args.focal_loss}(gamma={args.focal_gamma})")
    print(f"  池化={args.pooling}  FGM={args.fgm}(eps={args.fgm_epsilon})")
    print("=" * 70)

    # ---- 读数据 ----
    train_texts, train_labels = load_csv(data_dir / "train.csv")
    test_texts, test_labels = load_csv(data_dir / "test.csv")
    print(f"训练集 {len(train_texts)} 条，测试集 {len(test_texts)} 条")
    train_counts = torch.bincount(torch.tensor(train_labels), minlength=NUM_CLASSES)
    print("训练集类别分布: " + ", ".join(f"{l}={int(c)}" for l, c in zip(LABELS, train_counts)))

    # ---- 划分 train/val（分层）----
    tr_texts, tr_labels, val_texts, val_labels = stratified_split(
        train_texts, train_labels, args.val_ratio, args.seed)

    # ---- tokenizer / 模型 ----
    print("加载 tokenizer ...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    print("加载预训练模型 ...")
    model = XLMRobertaClassifier(args.model_name, NUM_CLASSES, args.pooling).to(device)
    print("模型加载完成，开始 tokenize 数据 ...")

    print(f"tokenize 训练集 ({len(tr_texts)} 条) ...")
    train_ds = TextDataset(tr_texts, tr_labels, tokenizer, args.max_length)
    print(f"tokenize 验证集 ({len(val_texts)} 条) ...")
    val_ds = TextDataset(val_texts, val_labels, tokenizer, args.max_length)
    print(f"tokenize 测试集 ({len(test_texts)} 条) ...")
    test_ds = TextDataset(test_texts, test_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}  "
          f"每 epoch {len(train_loader)} 个 batch")

    # ---- Loss ----
    class_weights = None
    if args.class_weight:
        class_weights = compute_class_weights(torch.tensor(tr_labels), NUM_CLASSES).to(device)
        print("类别权重:", {LABELS[i]: round(float(w), 3) for i, w in enumerate(class_weights)})

    if args.focal_loss:
        loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)
    elif args.label_smoothing > 0:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ---- 优化器 / 调度器 ----
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    fgm = FGM(model, epsilon=args.fgm_epsilon, emb_name=args.fgm_emb) if args.fgm else None

    # ---- 训练 ----
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler,
                                     fgm, device, epoch, args.epochs)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device,
                                           desc=f"Epoch {epoch}/{args.epochs} val")
        print(f"epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"早停：验证集准确率连续 {args.patience} 轮未提升")
                break

    # ---- 用最优验证集权重在测试集上评估 ----
    if best_state is not None:
        model.load_state_dict(best_state)
    print("在测试集上评估 ...")
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, loss_fn, device,
                                                  desc="test")

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=range(NUM_CLASSES))
    report_dict = classification_report(labels.numpy(), preds.numpy(),
                                        labels=range(NUM_CLASSES),
                                        target_names=LABELS, output_dict=True)
    report_str = classification_report(labels.numpy(), preds.numpy(),
                                       labels=range(NUM_CLASSES),
                                       target_names=LABELS, digits=4)

    print("\n" + "=" * 70)
    print(f"结果 ({run_name})")
    print("=" * 70)
    print(f"测试集准确率: {test_acc:.4f}")
    print("\n混淆矩阵:")
    print(format_confusion_matrix(cm))
    print("\n分类报告:")
    print(report_str)

    save_results(run_dir, run_name, args, test_acc, cm, report_dict, test_loss)
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    print(f"\n结果已保存到: {run_dir}/")
    print(f"  混淆矩阵热力图: confusion_matrix.png")
    print(f"  混淆矩阵 CSV:   confusion_matrix.csv")
    print(f"  指标 JSON:      metrics.json")
    print(f"  模型权重:       best_model.pt")


if __name__ == "__main__":
    main()

"""用 ModelScope 的 nlp-base-trainer 微调 week06/roberta 做单句文本分类。

    python train_cls.py                 # 全量数据，3 轮
    python train_cls.py --epochs 1 --limit 30   # 冒烟测试
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from modelscope.metainfo import Preprocessors, Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import Tasks

BASE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=str(BASE / "roberta"))
    p.add_argument("--work_dir", default=str(BASE / "output_cls"))
    p.add_argument("--train_file", default=str(BASE / "train.jsonl"))
    p.add_argument("--dev_file", default=str(BASE / "dev.jsonl"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="只取前 N 条训练，0 表示全量")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--pool", default="cls", choices=["cls", "mean", "max", "cls_mean"],
                   help="池化方式：cls(默认) / mean / max / cls_mean(CLS 与 Mean 拼接后投影)")
    p.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑系数，0 表示关闭")
    p.add_argument("--focal_gamma", type=float, default=0.0, help="Focal Loss 的 gamma，0 表示关闭")
    p.add_argument("--fgm", type=float, default=0.0, help="FGM 对抗训练的 eps，0 表示关闭")
    return p.parse_args()


def patch_pooling(pool):
    """替换 SbertForSequenceClassification 的池化方式。

    默认实现取 outputs[1]（[CLS] 经 pooler），这里改为从 outputs[0]（全部隐状态）
    做 mean / max / cls_mean 池化。cls_mean 会额外挂一个 1024->1024 投影层，
    在 __init__ 阶段创建，保证能被优化器收集到。
    """
    if pool == "cls":
        return

    import torch
    import torch.nn as nn
    from modelscope.models.nlp.structbert import text_classification as tc

    cls_model = tc.SbertForSequenceClassification

    if pool == "cls_mean":
        orig_init = cls_model.__init__

        def _init(self, *a, **kw):
            orig_init(self, *a, **kw)
            h = self.config.hidden_size
            self.concat_proj = nn.Linear(2 * h, h)
            nn.init.normal_(self.concat_proj.weight, std=0.02)
            nn.init.zeros_(self.concat_proj.bias)

        cls_model.__init__ = _init

    def _forward_call(self, **kwargs):
        outputs = self.base_model(**kwargs)
        h = outputs[0]                       # [B, L, H] 最后一层隐状态
        mask = kwargs.get("attention_mask")
        cls_pooled = outputs[1]              # [CLS] 经 pooler

        if pool == "mean":
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        elif pool == "max":
            m = mask.unsqueeze(-1).bool()
            pooled = h.masked_fill(~m, -1e4).max(1).values
        else:                                # cls_mean：拼接后投影回 H 维
            m = mask.unsqueeze(-1).float()
            mean_pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            pooled = torch.tanh(self.concat_proj(torch.cat([cls_pooled, mean_pooled], -1)))

        pooled = self.dropout(pooled)
        outputs["logits"] = self.classifier(pooled)
        outputs.kwargs = kwargs
        return outputs

    cls_model._forward_call = _forward_call


class FGM:
    """Fast Gradient Method：按梯度方向扰动 word_embeddings 的权重。"""

    def __init__(self, model, eps=1.0, name="word_embeddings"):
        self.model = model
        self.eps = eps
        self.name = name
        self.backup = {}
        self.matched = [n for n, _ in model.named_parameters() if name in n]
        assert self.matched, f"未找到名称包含 {name} 的参数"

    def attack(self):
        import torch

        for n, p in self.model.named_parameters():
            if self.name in n and p.requires_grad and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = torch.norm(p.grad, dim=1, keepdim=True)
                p.data.add_(self.eps * p.grad / (norm + 1e-8))

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def patch_loss(label_smoothing=0.0, focal_gamma=0.0):
    """覆盖 compute_loss，注入标签平滑 / Focal Loss。

    只替换 loss 的数值，其余字段（logits 等）沿用原实现。
    """
    if label_smoothing <= 0 and focal_gamma <= 0:
        return

    import torch
    import torch.nn.functional as F
    from modelscope.models.nlp.structbert import text_classification as tc

    orig_compute_loss = tc.SbertForSequenceClassification.compute_loss

    def _compute_loss(self, outputs, labels, **kwargs):
        out = orig_compute_loss(self, outputs, labels, **kwargs)
        if labels is not None and self.config.problem_type == "single_label_classification":
            logits = outputs.logits.view(-1, self.num_labels)
            target = labels.view(-1)
            if focal_gamma > 0:
                ce = F.cross_entropy(logits, target, reduction="none",
                                     label_smoothing=label_smoothing)
                pt = torch.exp(-ce)
                out["loss"] = ((1.0 - pt) ** focal_gamma * ce).mean()
            else:
                out["loss"] = F.cross_entropy(logits, target,
                                              label_smoothing=label_smoothing)
        return out

    tc.SbertForSequenceClassification.compute_loss = _compute_loss


def install_fgm(trainer, eps):
    """把 OptimizerHook 的 after_train_iter 换成 FGM 版本。

    trainer 的 hooks 是在 train() 里才构建的，所以这里直接补丁类方法，
    并把 fgm 实例挂在 trainer 上按需查用。
    """
    from modelscope.outputs import ModelOutputBase
    from modelscope.trainers.hooks.optimizer.base import OptimizerHook

    fgm = FGM(trainer.model, eps)
    trainer._fgm = fgm
    print(f"[FGM] eps={eps}, 扰动参数: {fgm.matched}")

    if getattr(OptimizerHook, "_fgm_patched", False):
        return fgm

    orig_after = OptimizerHook.after_train_iter

    def after_train_iter(self, trainer):
        active = getattr(trainer, "_fgm", None)
        keys = [k for k in trainer.train_outputs.keys() if "loss" in k]
        if active is None or not keys or self.cumulative_iters != 1:
            return orig_after(self, trainer)

        trainer.train_outputs[keys[0]].backward()   # 干净的梯度
        active.attack()                              # 按梯度扰动 word_embeddings
        out = trainer.model.forward(**trainer.data_batch)
        if isinstance(out, ModelOutputBase):
            out = out.to_dict()
        out["loss"].backward()                       # 对抗样本的梯度
        active.restore()                             # 恢复原始权重
        if self.grad_clip is not None:
            self.processor.clip_grads(trainer.model.parameters(), **self.grad_clip)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    OptimizerHook.after_train_iter = after_train_iter
    OptimizerHook._fgm_patched = True
    return fgm


def main():
    args = parse_args()

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.dev_file, split="train")
    if args.limit:
        train_ds = train_ds.select(range(min(args.limit, len(train_ds))))
        eval_ds = eval_ds.select(range(min(args.limit, len(eval_ds))))

    labels = sorted({r["label"] for r in train_ds})

    def cfg_modify_fn(cfg):
        # backbone(fill-mask) -> 单句分类，分类头随机初始化
        cfg.task = Tasks.text_classification
        cfg["preprocessor"] = {
            "type": Preprocessors.sen_cls_tokenizer,
            "max_length": args.max_length,
        }
        # 字段名必须和 jsonl 的 key 一致
        cfg["dataset"] = {
            "train": {
                "labels": labels,
                "first_sequence": "text",
                "label": "label",
            }
        }
        cfg.train.max_epochs = args.epochs
        cfg.train.optimizer.lr = args.lr
        cfg.train.dataloader.batch_size_per_gpu = args.batch
        cfg.train.dataloader.workers_per_gpu = 0  # Windows 必须为 0
        cfg.evaluation.dataloader.batch_size_per_gpu = max(args.batch * 2, 8)
        cfg.evaluation.dataloader.workers_per_gpu = 0
        cfg.train.lr_scheduler.total_iters = -(-len(train_ds) // args.batch) * args.epochs
        cfg.train.hooks = [
            {"type": "CheckpointHook", "interval": 1},
            {"type": "TextLoggerHook", "interval": 1},
            {"type": "IterTimerHook"},
            {"type": "EvaluationHook", "by_epoch": True, "interval": 1},
            # restore_best 在 torch>=2.6 下会因 weights_only=True 读取 trainer_state 失败
            {"type": "BestCkptSaverHook", "metric_key": "accuracy",
             "rule": "max", "save_best": True, "restore_best": False},
        ]
        return cfg

    os.environ.pop("LOCAL_RANK", None)
    patch_pooling(args.pool)
    patch_loss(args.label_smoothing, args.focal_gamma)
    print(f"[pool] {args.pool}  [ls] {args.label_smoothing}  "
          f"[focal] {args.focal_gamma}  [fgm] {args.fgm}")

    trainer = build_trainer(
        name=Trainers.nlp_base_trainer,
        default_args=dict(
            model=str(Path(args.model_dir)).replace("\\", "/"),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            work_dir=args.work_dir,
            cfg_modify_fn=cfg_modify_fn,
            seed=args.seed,
        ),
    )
    if args.fgm > 0:
        install_fgm(trainer, args.fgm)
    trainer.train()
    print(f"\n训练结束，产物目录: {args.work_dir}")


if __name__ == "__main__":
    main()

#region 0. 导入库
import argparse
import sys
import time

from datasets import load_from_disk
from transformers import (DataCollatorForSeq2Seq, Trainer, TrainerCallback,
                          TrainingArguments)

from common import DATA_DIR, OUT_DIR, ART_DIR, get_lora_config, get_tokenizer, load_base, save_json
#endregion

#region 1. 回调：把 loss / lr / 耗时按步记下来，给报告画曲线   【类:StepLogger】
class StepLogger(TrainerCallback):
    """记录每个 logging 步的 loss、学习率、累计耗时。结果存在 self.rows。"""
    def __init__(self):
        self.rows, self.t0 = [], time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Trainer 每次打日志时触发。返回：None。"""
        if logs and "loss" in logs:
            self.rows.append({"step": state.global_step, "loss": round(logs["loss"], 4),
                              "lr": logs.get("learning_rate", 0), "elapsed": round(time.time() - self.t0, 1)})
#endregion

#region 2. 命令行参数
def parse_args():
    """解析参数。返回：Namespace。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 20 步，用来校准每步耗时")
    ap.add_argument("--low-mem", action="store_true", help="显存不足时降 batch、缩长度")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练，0 表示全量")
    return ap.parse_args()
#endregion

#region 3. 主流程：挂 LoRA → 训练 → 存 adapter
def main():
    args = parse_args()
    bs, accum, max_len = (4, 4, 384) if args.low_mem else (8, 2, 512)     # 有效 batch 恒为 16
    tok = get_tokenizer(padding_side="right")                            # 因果 LM 训练必须右侧 padding
    train = load_from_disk(str(DATA_DIR / "train"))
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))

    model = load_base(trainable=True)
    model.add_adapter(get_lora_config())                                 # 只插入 LoRA，不动原权重
    model.enable_input_require_grads()                                   # 梯度检查点 + PEFT 的必需开关
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"可训练参数 {n_train:,} / {n_all:,} = {n_train / n_all:.2%}")

    logger = StepLogger()
    steps_per_epoch = max(1, len(train) // (bs * accum))
    targs = TrainingArguments(
        output_dir=str(OUT_DIR), per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, fp16=False, gradient_checkpointing=True, optim="adamw_torch",
        logging_steps=10, save_strategy="no", eval_strategy="no", report_to="none",
        dataloader_num_workers=0, max_grad_norm=1.0, remove_unused_columns=True,
        max_steps=20 if args.smoke else -1,
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    trainer = Trainer(model=model, args=targs, train_dataset=train, data_collator=collator, callbacks=[logger])

    t0 = time.time()
    trainer.train()
    cost = time.time() - t0
    done_steps = trainer.state.global_step
    print(f"\n训练结束: {done_steps} 步，耗时 {cost:.1f}s，平均 {cost / max(1, done_steps):.2f}s/步")
    if args.smoke:
        print("显存峰值: %.2f GB" % (torch_peak()))
        return 0

    model.save_pretrained(str(OUT_DIR))                                  # 只存 9MB 的 adapter
    save_json({"steps": done_steps, "seconds": round(cost, 1), "s_per_step": round(cost / max(1, done_steps), 3),
               "steps_per_epoch": steps_per_epoch, "epochs": args.epochs, "batch": bs, "accum": accum,
               "max_len": max_len, "lr": args.lr, "n_train": len(train),
               "trainable_params": n_train, "all_params": n_all, "log": logger.rows},
              ART_DIR / "train_log.json")
    print(f"adapter 已存到 {OUT_DIR}")
    return 0
#endregion

def torch_peak():
    """返回显存峰值 GB。返回：float。"""
    import torch
    return torch.cuda.max_memory_allocated() / 1024 ** 3

if __name__ == "__main__":
    sys.exit(main())

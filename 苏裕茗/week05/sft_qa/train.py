import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EncoderDecoderModel, BertTokenizer, get_linear_schedule_with_warmup

# ========== 配置 ==========
MODEL_PATH = "./bert-base-chinese"
TRAIN_PATH = "./sft_train/sft_train.jsonl"
VAL_PATH = "./sft_train/sft_val.jsonl"
OUTPUT_DIR = "./bert-sft-qa"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-5
WARMUP_STEPS = 100
LOG_STEPS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 加载模型和 tokenizer ==========
print("加载 tokenizer 和模型...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_PATH, MODEL_PATH)

# 配置生成相关特殊 token
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# 生成配置（推理时使用）
model.config.max_length = MAX_LENGTH
model.config.min_length = 5
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.length_penalty = 2.0
model.config.num_beams = 4

model.to(device)

# ========== 数据集 ==========
class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # encoder 输入：instruction（问题）
        inputs = self.tokenizer(
            item["instruction"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # decoder 标签：output（答案）
        labels = self.tokenizer(
            item["output"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_ids = labels["input_ids"].squeeze()
        # padding 位置设为 -100，CrossEntropyLoss 会忽略这些位置
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": label_ids,
        }


train_dataset = QADataset(TRAIN_PATH, tokenizer, MAX_LENGTH)
val_dataset = QADataset(VAL_PATH, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")

# ========== 训练前：固定 batch 验证 ==========
print("\n========== 训练前验证 ==========")
model.eval()
fixed_batch = next(iter(val_loader))
fixed_batch = {k: v.to(device) for k, v in fixed_batch.items()}

with torch.no_grad():
    outputs = model(**fixed_batch)
    loss_before = outputs.loss.item()

# 记录 decoder 最后一层 dense 权重作为参考，观察梯度是否真正更新了参数
ref_param_name = "decoder.bert.encoder.layer.11.output.dense.weight"
ref_param = dict(model.named_parameters())[ref_param_name]
ref_param_before = ref_param.data.clone()
ref_norm_before = ref_param_before.norm().item()

print(f"固定 batch loss (训练前): {loss_before:.4f}")
print(f"参考参数 {ref_param_name} 的 L2 范数 (训练前): {ref_norm_before:.6f}")

# ========== 优化器和学习率调度 ==========
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)

# ========== 训练循环 ==========
print("\n========== 开始训练 ==========")
model.train()

global_step = 0
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()

        # 计算全局梯度范数，验证梯度是否正常回传
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        epoch_loss += loss.item()

        if global_step % LOG_STEPS == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[Epoch {epoch+1}/{EPOCHS} | Step {global_step}] "
                f"Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f} | LR: {current_lr:.2e}"
            )

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f">>> Epoch {epoch+1} 平均 Loss: {avg_epoch_loss:.4f}")

# ========== 训练后：同一 batch 验证 ==========
print("\n========== 训练后验证 ==========")
model.eval()

with torch.no_grad():
    outputs = model(**fixed_batch)
    loss_after = outputs.loss.item()

ref_param_after = ref_param.data.clone()
ref_norm_after = ref_param_after.norm().item()
param_diff = (ref_param_after - ref_param_before).norm().item()

print(f"固定 batch loss (训练后): {loss_after:.4f}")
print(f"参考参数 {ref_param_name} 的 L2 范数 (训练后): {ref_norm_after:.6f}")
print(f"参数变化量 (L2): {param_diff:.6f}")
print(f"Loss 下降量: {loss_before - loss_after:.4f}")

# ========== 保存模型 ==========
print(f"\n保存模型到 {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("训练完成！")

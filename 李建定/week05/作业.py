"""
SFT（监督微调）训练与推理脚本，支持 RNN / LSTM / Transformer 切换。
包含：EOS停止符、动态分桶、梯度累积、学习率衰减、早停与断点续训、生成测试。
"""

import math
import argparse
import json
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import Counter

# CPU 多线程优化
torch.set_num_threads(16)
torch.set_num_interop_threads(2)

# ─────────────────────────── 特殊符号 ───────────────────────────
PAD_CHAR = "\u0000"
EOS_CHAR = "<EOS>"           # 新增：结束符，让模型学会停下
SEP = "\n###\n"
IGNORE_INDEX = -100


# ─────────────────────────── 数据 ───────────────────────────

def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question")
            completion = obj.get("completion") or obj.get("output") or obj.get("answer")
            if prompt is not None and completion is not None:
                samples.append({"prompt": str(prompt), "completion": str(completion)})
    return samples


def build_vocab_from_samples(samples, max_vocab=None):
    counter = Counter()
    for s in samples:
        counter.update(s["prompt"])
        counter.update(s["completion"])
        counter.update(SEP)

    limit = (max_vocab - 1) if max_vocab else None
    chars = [c for c, _ in counter.most_common(limit)] if limit else sorted(counter.keys())

    # 0号位是 PAD，1号位是 EOS
    char2idx = {PAD_CHAR: 0, EOS_CHAR: 1}
    for c in chars:
        if c not in char2idx:
            char2idx[c] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class SFTDataset(Dataset):
    def __init__(self, samples, char2idx, max_len=128):
        self.samples = []
        self.max_len = max_len

        for obj in samples:
            # 在 completion 后面加上 EOS
            prompt_ids = [char2idx.get(c, 0) for c in obj["prompt"] + SEP]
            completion_ids = [char2idx.get(c, 0) for c in obj["completion"]] + [char2idx[EOS_CHAR]]

            ids = prompt_ids + completion_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids

            ids = ids[: self.max_len]
            labels = labels[: self.max_len]

            if all(l == IGNORE_INDEX for l in labels):
                continue

            self.samples.append({
                "ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "length": len(ids)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = list(range(len(dataset)))
        self.indices.sort(key=lambda i: dataset[i]["length"])
        self.batches = [self.indices[i:i + batch_size] for i in range(0, len(self.indices), batch_size)]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.batches)


def sft_collate_fn(batch):
    max_len = max(x["ids"].size(0) for x in batch)
    
    padded_ids, padded_labels, attn_mask = [], [], []
    for x in batch:
        ids, labels = x["ids"], x["labels"]
        pad_len = max_len - ids.size(0)
        
        padded_ids.append(torch.cat([ids, torch.full((pad_len,), 0, dtype=torch.long)]))
        padded_labels.append(torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
        attn_mask.append(torch.cat([torch.ones(ids.size(0), dtype=torch.long),
                                    torch.zeros(pad_len, dtype=torch.long)]))

    return torch.stack(padded_ids), torch.stack(padded_labels), torch.stack(attn_mask)


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, padding_mask=None):
        e = self.drop(self.embed(x))
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).cpu()
            packed_e = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_e)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.pos = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        T = x.size(1)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        return self.drop(x + self.pos(pos_ids))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(seq_len, d_model, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x, padding_mask=None):
        e = self.embed(x)
        e = self.pos(e)
        T = x.size(1)
        causal = self.causal_mask[:T, :T]
        out = self.encoder(e, mask=causal, src_key_padding_mask=padding_mask)
        out = self.norm(out)
        return self.fc(out)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, model_type="transformer", grad_accum=1):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    
    if train:
        optimizer.zero_grad()

    for i, (x, y, attn_mask) in enumerate(loader):
        x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)
        padding_mask = (attn_mask == 0)
        logits = model(x, padding_mask=padding_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            loss = loss / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        n_valid = (y != IGNORE_INDEX).sum().item()
        if n_valid > 0:
            total_loss += loss.item() * grad_accum * n_valid
            total_tokens += n_valid

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 50))
    return avg_loss, ppl


@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, device, max_new_tokens=30):
    model.eval()
    
    # 【调试】打印输入
    print(f"\n[Debug] 输入的原始 prompt: '{prompt}'")
    ids = [char2idx.get(c, 0) for c in prompt + SEP]
    print(f"[Debug] 包含 SEP 后的长度: {len(ids)}")
    
    x = torch.tensor([ids], dtype=torch.long).to(device)
    generated_text = ""

    for _ in range(max_new_tokens):
        x_cond = x[:, -128:] 
        logits = model(x_cond)
        next_token_logits = logits[0, -1, :].clone()
        
        # 屏蔽 PAD
        next_token_logits[0] = -float('inf') 
        
        # 纯贪心选择
        next_token = torch.argmax(next_token_logits).item()
        next_char = idx2char.get(next_token, "")
        
        # 遇到 EOS 才停
        if next_token == 1:
            break
        
        generated_text += next_char
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

    return generated_text

# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=30) # 增加了默认轮数
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--grad_accum", type=int,   default=1)
    parser.add_argument("--embed_dim",  type=int,   default=256) # 从 128 提升到 256
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=4)   # 从 2 层提升到 4 层
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--max_vocab",  type=int,   default=4000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--train_file", default="sft_train.jsonl")
    parser.add_argument("--val_file",   default="sft_val.jsonl")
    parser.add_argument("--save",       default="best_sft_model.pt")
    parser.add_argument("--patience",   type=int,   default=4)   # 适当提高耐心
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--nhead",      type=int,   default=8)   # 因为 embed_dim=256，8头比较好
    parser.add_argument("--dim_ff",     type=int,   default=1024) # 前馈层相应提升
    parser.add_argument("--test_prompt",type=str,   default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = load_jsonl(args.train_file)
    val_samples   = load_jsonl(args.val_file)
    if not train_samples:
        raise FileNotFoundError(f"未找到训练数据: {args.train_file}")

    char2idx, idx2char = build_vocab_from_samples(train_samples, args.max_vocab)
    vocab_size = len(char2idx)

    train_ds = SFTDataset(train_samples, char2idx, max_len=args.seq_len)
    val_ds   = SFTDataset(val_samples,   char2idx, max_len=args.seq_len)

    train_sampler = BucketBatchSampler(train_ds, args.batch_size, shuffle=True)
    val_sampler   = BucketBatchSampler(val_ds, args.batch_size, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=sft_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,   collate_fn=sft_collate_fn)

    if args.model == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size, seq_len=args.seq_len, d_model=args.embed_dim,
            nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_ff, dropout=args.dropout,
        ).to(device)
    else:
        model = LM(
            vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, model_type=args.model, dropout=args.dropout,
        ).to(device)

    if args.test_prompt:
        if os.path.exists(args.save):
            checkpoint = torch.load(args.save, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            print(f"✅ 成功加载模型: {args.save}")
            output = generate(model, args.test_prompt, char2idx, idx2char, device)
            print(f"\n【Prompt】\n{args.test_prompt}\n\n【Model Output】\n{output}\n")
        else:
            print(f"❌ 未找到模型文件 {args.save}，请先训练。")
        return

    print(f"device: {device}  model: {args.model.upper()}")
    print(f"训练样本数: {len(train_samples)}  验证样本数: {len(val_samples)}")
    print(f"词表大小: {vocab_size}  参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    start_epoch = 1
    checkpoint = None
    if args.resume and os.path.exists(args.save):
        print(f"\n[续训模式] 加载 {args.save} ...")
        checkpoint = torch.load(args.save, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        best_val_ppl = checkpoint.get("best_val_ppl", float("inf"))
    else:
        best_val_ppl = float("inf")
    
    patience_counter = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 60)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True, model_type=args.model, grad_accum=args.grad_accum)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer,
                                        device, train=False, model_type=args.model)

        scheduler.step(va_loss)

        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            patience_counter = 0
            marker = "  *"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
                "best_val_ppl": best_val_ppl,
            }, args.save)
        else:
            patience_counter += 1
            marker = f"  (early-stop: {patience_counter}/{args.patience})"

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

        if patience_counter >= args.patience:
            print(f"\n[早停触发] 验证集 PPL 连续 {args.patience} 轮未下降，停止训练。")
            break

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()

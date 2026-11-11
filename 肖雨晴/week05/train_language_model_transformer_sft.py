"""
基于字符级 Transformer 预训练模型进行监督微调（SFT）。

默认读取同目录下的 ``best_transformer_model.pt``，使用 ``sft_train``
目录中的 JSONL 数据训练。损失只计算回答部分，不计算用户指令和 padding。

用法:
    python language_model_transformer.py
    python language_model_transformer.py --epochs 10 --batch_size 32 --lr 1e-4
"""

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
IGNORE_INDEX = -100


# ─────────────────────────── 数据 ───────────────────────────

def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc

            instruction = record.get("instruction")
            output = record.get("output")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path} 第 {line_no} 行缺少非空 instruction")
            if not isinstance(output, str) or not output.strip():
                raise ValueError(f"{path} 第 {line_no} 行缺少非空 output")
            records.append(record)

    if not records:
        raise ValueError(f"SFT 数据为空: {path}")
    return records


def format_prompt(record):
    instruction = record["instruction"].strip()
    input_text = record.get("input", "")
    if isinstance(input_text, str) and input_text.strip():
        instruction = f"{instruction}\n{input_text.strip()}"
    return f"用户：{instruction}\n助手："


def extend_vocab(base_char2idx, records):
    """保留预训练 token id，并将 SFT 数据中的新字符追加到词表末尾。"""
    char2idx = dict(base_char2idx)
    expected_ids = list(range(len(char2idx)))
    if sorted(char2idx.values()) != expected_ids:
        raise ValueError("预训练词表 token id 必须从 0 开始连续编号")

    chars = set()
    for record in records:
        chars.update(format_prompt(record))
        chars.update(record["output"].strip())

    for char in sorted(chars):
        if char not in char2idx:
            char2idx[char] = len(char2idx)
    for token in (PAD_TOKEN, EOS_TOKEN):
        if token not in char2idx:
            char2idx[token] = len(char2idx)
    return char2idx


class SFTDataset(Dataset):
    def __init__(self, records, char2idx, seq_len):
        if seq_len < 2:
            raise ValueError("seq_len 至少为 2")

        self.examples = []
        self.truncated_count = 0
        eos_id = char2idx[EOS_TOKEN]
        max_tokens = seq_len + 1  # 右移前可比模型输入多一个目标 token

        for record in records:
            prompt_ids = [char2idx[char] for char in format_prompt(record)]
            answer_ids = [char2idx[char] for char in record["output"].strip()]

            if len(prompt_ids) + len(answer_ids) + 1 > max_tokens:
                self.truncated_count += 1
                # 至少保留一个提示字符和 EOS，优先保留完整回答。
                max_answer_len = max_tokens - 2
                answer_ids = answer_ids[:max_answer_len]
                prompt_space = max_tokens - len(answer_ids) - 1
                prompt_ids = prompt_ids[-prompt_space:]

            tokens = prompt_ids + answer_ids + [eos_id]
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)

            # labels[prompt_len - 1] 是第一个回答字符，应参与损失计算。
            prompt_target_count = max(0, len(prompt_ids) - 1)
            labels[:prompt_target_count] = IGNORE_INDEX
            self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_sft(batch, pad_token_id):
    max_len = max(input_ids.size(0) for input_ids, _ in batch)
    inputs = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, (input_ids, target_ids) in enumerate(batch):
        length = input_ids.size(0)
        inputs[row, :length] = input_ids
        labels[row, :length] = target_ids
    return inputs, labels


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_heads=4, dropout=0.3, max_seq_len=512):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须为正数且能被正整数 num_heads 整除")
        if hidden_dim <= 0 or num_layers <= 0 or max_seq_len <= 0:
            raise ValueError("hidden_dim、num_layers 和 max_seq_len 必须为正数")

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x, padding_mask=None):
        if x.dim() != 2 or not 1 <= x.size(1) <= self.max_seq_len:
            raise ValueError(f"输入形状必须为 (B, T)，且 1 <= T <= {self.max_seq_len}")
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        embeddings = self.embed(x) * math.sqrt(self.embed_dim)
        embeddings = embeddings + self.position_embed(positions)
        output = self.transformer(
            self.drop(embeddings),
            mask=self.causal_mask[:seq_len, :seq_len],
            src_key_padding_mask=padding_mask,
        )
        return self.fc(self.drop(output))


def load_pretrained_model(checkpoint, char2idx, seq_len, dropout=None):
    pretrained_state = checkpoint["model_state"]
    pretrained_vocab = checkpoint["char2idx"]
    pretrained_args = checkpoint.get("args", {})

    embed_dim = pretrained_args.get("embed_dim", pretrained_state["embed.weight"].size(1))
    hidden_dim = pretrained_args.get(
        "hidden_dim", pretrained_state["transformer.layers.0.linear1.weight"].size(0)
    )
    num_layers = pretrained_args.get("num_layers")
    num_heads = pretrained_args.get("num_heads")
    if num_layers is None or num_heads is None:
        raise ValueError("checkpoint args 缺少 num_layers 或 num_heads，无法还原模型结构")
    model_dropout = pretrained_args.get("dropout", 0.3) if dropout is None else dropout

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=model_dropout,
        max_seq_len=seq_len,
    )

    # 先复制形状不变的 Transformer 参数，再适配扩展后的词表和位置长度。
    model_state = model.state_dict()
    resized_names = {"embed.weight", "position_embed.weight", "fc.weight", "fc.bias"}
    for name, value in pretrained_state.items():
        if name in resized_names:
            continue
        if name not in model_state or model_state[name].shape != value.shape:
            raise ValueError(f"预训练参数形状不兼容: {name}")
        model_state[name].copy_(value)

    old_vocab_size = len(pretrained_vocab)
    if pretrained_state["embed.weight"].size(0) != old_vocab_size:
        raise ValueError("checkpoint 词表大小与 embedding 行数不一致")
    model_state["embed.weight"][:old_vocab_size].copy_(pretrained_state["embed.weight"])
    model_state["fc.weight"][:old_vocab_size].copy_(pretrained_state["fc.weight"])
    model_state["fc.bias"][:old_vocab_size].copy_(pretrained_state["fc.bias"])

    old_positions = pretrained_state["position_embed.weight"]
    copied_positions = min(seq_len, old_positions.size(0))
    model_state["position_embed.weight"][:copied_positions].copy_(
        old_positions[:copied_positions]
    )
    model.load_state_dict(model_state)

    config = {
        "model": "transformer",
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": model_dropout,
        "seq_len": seq_len,
    }
    return model, config, old_vocab_size, copied_positions


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, pad_token_id, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        padding_mask = input_ids.eq(pad_token_id)

        with torch.set_grad_enabled(train):
            logits = model(input_ids, padding_mask=padding_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        supervised_tokens = labels.ne(IGNORE_INDEX).sum().item()
        total_loss += loss.item() * supervised_tokens
        total_tokens += supervised_tokens

    if total_tokens == 0:
        raise ValueError("没有可用于 SFT 的回答 token")
    average_loss = total_loss / total_tokens
    return average_loss, math.exp(average_loss)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="基于预训练字符级 Transformer 的 SFT 训练")
    parser.add_argument("--checkpoint", default=str(script_dir / "best_transformer_model.pt"))
    parser.add_argument("--train_data", default=str(script_dir / "sft_train" / "sft_train.jsonl"))
    parser.add_argument("--val_data", default=str(script_dir / "sft_train" / "sft_val.jsonl"))
    parser.add_argument("--save", default=str(script_dir / "best_sft_transformer_model.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=None,
                        help="默认沿用预训练 dropout；传值可覆盖")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.seq_len < 2:
        parser.error("--epochs、--batch_size 必须大于 0，--seq_len 至少为 2")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("--lr 必须大于 0，--weight_decay 不能小于 0")
    if args.dropout is not None and not 0 <= args.dropout < 1:
        parser.error("--dropout 必须在 [0, 1) 范围内")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {"model_state", "char2idx"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise ValueError("checkpoint 必须包含 model_state 和 char2idx")

    train_records = load_jsonl(args.train_data)
    val_records = load_jsonl(args.val_data)
    base_vocab_size = len(checkpoint["char2idx"])
    char2idx = extend_vocab(checkpoint["char2idx"], train_records + val_records)
    idx2char = {index: char for char, index in char2idx.items()}

    train_dataset = SFTDataset(train_records, char2idx, args.seq_len)
    val_dataset = SFTDataset(val_records, char2idx, args.seq_len)
    collate_fn = partial(collate_sft, pad_token_id=char2idx[PAD_TOKEN])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model, model_config, copied_vocab_size, copied_positions = load_pretrained_model(
        checkpoint, char2idx, args.seq_len, dropout=args.dropout
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print(f"device: {device}  预训练模型: {checkpoint_path}")
    print(f"SFT 数据: train={len(train_dataset):,}  val={len(val_dataset):,}")
    print(
        f"词表: {base_vocab_size:,} -> {len(char2idx):,}，"
        f"已复制 {copied_vocab_size:,} 行预训练 embedding/output 权重"
    )
    print(
        f"上下文长度: {args.seq_len}，已复制 {copied_positions} 个预训练位置向量；"
        f"截断 train={train_dataset.truncated_count}  val={val_dataset.truncated_count}"
    )
    print(f"模型参数量: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, char2idx[PAD_TOKEN], train=True
        )
        val_loss, val_ppl = run_epoch(
            model, val_loader, criterion, optimizer, device, char2idx[PAD_TOKEN], train=False
        )

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_args = {
                **model_config,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "train_data": str(Path(args.train_data).resolve()),
                "val_data": str(Path(args.val_data).resolve()),
                "base_checkpoint": str(checkpoint_path.resolve()),
            }
            save_checkpoint(args.save, {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "best_val_loss": best_val_loss,
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": save_args,
                "special_tokens": {"pad": PAD_TOKEN, "eos": EOS_TOKEN},
            })
            marker = f"  * 已保存 -> {args.save}"

        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_ppl:>10.2f}  "
            f"{val_loss:>10.4f}  {val_ppl:>10.2f}{marker}",
            flush=True,
        )

    print(f"\nSFT 完成。最佳验证 Loss: {best_val_loss:.4f}，模型保存在 {args.save}")


if __name__ == "__main__":
    main()

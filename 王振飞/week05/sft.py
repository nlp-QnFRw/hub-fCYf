"""在预训练字符级 Transformer 上进行指令微调，并支持交互式生成。"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from language_model_transformer import LM

BASE = Path(__file__).resolve().parent
PRETRAINED_MODEL = BASE / "transformer_model.pt"
SFT_MODEL = BASE / "sft_model.pt"
TRAIN_FILE = BASE / "sft_train.jsonl"
VAL_FILE = BASE / "sft_val.jsonl"
SEQ_LEN = 160
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 3e-4
SFT_DROPOUT = 0.1
MAX_NEW_TOKENS = 100
PROMPT = "问题：{}\n回答："
PAD, UNK, EOS = "<PAD>", "<UNK>", "<EOS>"
IGNORE = -100


def read_data(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip() for k in ("instruction", "output")):
                    raise ValueError("instruction 和 output 必须是非空字符串")
            except (ValueError, json.JSONDecodeError) as e:
                raise ValueError(f"{path} 第 {line_no} 行无效: {e}") from e
            rows.append(item)
    if not rows:
        raise ValueError(f"数据为空: {path}")
    return rows


def extend_vocab(original, rows):
    vocab = dict(original)
    chars = set(PROMPT.replace("{}", ""))
    for row in rows:
        chars.update(row["instruction"])
        chars.update(row["output"])
    for token in [PAD, UNK, EOS] + sorted(chars):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text, vocab):
    return [vocab.get(ch, vocab[UNK]) for ch in text]


class SFTDataset(Dataset):
    def __init__(self, rows, vocab, seq_len, name):
        self.samples = []
        unknown = 0
        for i, row in enumerate(rows, 1):
            prompt = PROMPT.format(row["instruction"])
            answer = row["output"]
            unknown += sum(ch not in vocab for ch in prompt + answer)
            prompt_ids = encode(prompt, vocab)
            ids = prompt_ids + encode(answer, vocab) + [vocab[EOS]]
            # x 和 y 均比完整序列短一个字符。
            if len(ids) - 1 > seq_len:
                raise ValueError(f"{name} 第 {i} 条样本需要 {len(ids)-1} 个位置，超过 seq_len={seq_len}；请增大 --seq_len")
            x = ids[:-1]
            y = [IGNORE] * (len(prompt_ids) - 1) + ids[len(prompt_ids):]
            self.samples.append((x, y))
        if unknown:
            print(f"{name}: {unknown} 个未登录字符将使用 <UNK>")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch, pad_id):
    length = max(len(x) for x, _ in batch)
    xs = [x + [pad_id] * (length - len(x)) for x, _ in batch]
    ys = [y + [IGNORE] * (length - len(y)) for _, y in batch]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def make_model(config, vocab_size):
    return LM(vocab_size, config["embed_dim"], config["hidden_dim"],
              config["num_layers"], config["num_heads"], config["seq_len"], config["dropout"])


def initialize(pretrained, vocab, seq_len):
    config = dict(pretrained["args"])
    # SFT 数据量较小，降低 dropout，减少欠拟合。
    config["dropout"] = SFT_DROPOUT
    if config.get("model") != "transformer":
        raise ValueError("预训练模型不是 Transformer")
    old_vocab = pretrained["char2idx"]
    if sorted(old_vocab.values()) != list(range(len(old_vocab))):
        raise ValueError("预训练词表的编号不连续")
    old_length = config["seq_len"]
    if seq_len < old_length:
        raise ValueError(f"seq_len 不能小于预训练模型的 {old_length}")
    config["seq_len"] = seq_len
    model = make_model(config, len(vocab))
    state = pretrained["model_state"]
    # 旧字符和旧位置的参数逐项保留；新增部分保留模型的随机初始化。
    expanded = model.state_dict()
    for key, old in state.items():
        if key in ("embed.weight", "fc.weight", "fc.bias", "pos_embed.weight"):
            expanded[key][:old.shape[0]].copy_(old)
        else:
            expanded[key] = old
    model.load_state_dict(expanded, strict=True)
    return model, config


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = (y != IGNORE).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
    return total_loss / total_tokens


def train():
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PRETRAINED_MODEL, map_location="cpu", weights_only=True)
    if checkpoint.get("sft_format"):
        raise ValueError("--model 请指定原始预训练模型；本脚本不会把 SFT 检查点当作预训练模型")
    train_rows, val_rows = read_data(TRAIN_FILE), read_data(VAL_FILE)
    vocab = extend_vocab(checkpoint["char2idx"], train_rows)
    model, config = initialize(checkpoint, vocab, SEQ_LEN)
    train_ds = SFTDataset(train_rows, vocab, SEQ_LEN, "训练集")
    val_ds = SFTDataset(val_rows, vocab, SEQ_LEN, "验证集")
    def batch_fn(batch):
        return collate(batch, vocab[PAD])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=batch_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=batch_fn)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"设备: {device}；训练: {len(train_ds)} 条；验证: {len(val_ds)} 条；词表: {len(checkpoint['char2idx'])} → {len(vocab)}；上下文: {config['seq_len']}")
    best = float("inf")
    for number in range(1, EPOCHS + 1):
        tr = epoch(model, train_loader, device, optimizer)
        val = epoch(model, val_loader, device)
        marker = ""
        if val < best:
            best = val
            torch.save({"model_state": model.state_dict(), "char2idx": vocab,
                        "idx2char": {i: c for c, i in vocab.items()}, "args": config,
                        "sft_format": 1, "prompt_template": PROMPT,
                        "sft_train_args": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                                           "learning_rate": LEARNING_RATE}, "val_loss": val}, SFT_MODEL)
            marker = " *"
        print(f"epoch {number:>3}  train loss {tr:.4f}  val loss {val:.4f}  val ppl {math.exp(min(val, 20)):.2f}{marker}")
    print(f"最佳模型已保存到: {SFT_MODEL}")


def validate():
    if not SFT_MODEL.exists():
        raise FileNotFoundError(f"未找到 {SFT_MODEL}，请先运行: python sft.py train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(SFT_MODEL, map_location="cpu", weights_only=True)
    vocab, config = checkpoint["char2idx"], checkpoint["args"]
    rows = read_data(VAL_FILE)
    dataset = SFTDataset(rows, vocab, config["seq_len"], "验证集")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda batch: collate(batch, vocab[PAD]))
    model = make_model(config, len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss = epoch(model, loader, device)
    print(f"验证集 loss: {loss:.4f}，PPL: {math.exp(min(loss, 20)):.2f}")


def generate():
    if not SFT_MODEL.exists():
        raise FileNotFoundError(f"未找到 {SFT_MODEL}，请先运行: python sft.py train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(SFT_MODEL, map_location=device, weights_only=True)
    if not checkpoint.get("sft_format"):
        raise ValueError("生成时请用 --model 指定 SFT 模型")
    vocab, config = checkpoint["char2idx"], checkpoint["args"]
    model = make_model(config, len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    reverse = checkpoint["idx2char"]
    print("输入问题，输入 quit 退出。")
    while True:
        try:
            question = input("\n问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        prompt = checkpoint["prompt_template"].format(question)
        ids = encode(prompt, vocab)
        if len(ids) >= config["seq_len"]:
            print("问题过长，请缩短后重试。")
            continue
        result = []
        with torch.inference_mode():
            for _ in range(min(MAX_NEW_TOKENS, config["seq_len"] - len(ids))):
                logits = model(torch.tensor([ids], device=device))[0, -1].clone()
                logits[vocab[PAD]] = -float("inf")
                logits[vocab[UNK]] = -float("inf")
                next_id = logits.argmax().item()
                if next_id == vocab[EOS]:
                    break
                result.append(reverse[next_id])
                ids.append(next_id)
        print("回答：" + "".join(result))


def main():
    parser = argparse.ArgumentParser(description="字符级 Transformer SFT")
    parser.add_argument("mode", choices=("train", "val", "test"),
                        help="train=训练，val=验证，test=交互式测试")
    mode = parser.parse_args().mode
    if mode == "train":
        train()
    elif mode == "val":
        validate()
    else:
        generate()


if __name__ == "__main__":
    main()

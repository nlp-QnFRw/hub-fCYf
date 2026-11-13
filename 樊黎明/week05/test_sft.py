"""
SFT 模型测试脚本：验证集 loss/PPL + 生成质量（完全匹配率、字符级 F1），并抽样打印。
用法:
    python test_sft.py                       # 全量算 loss，抽 8 条生成
    python test_sft.py --limit 50 --max_new 40
    python test_sft.py --checkpoint sft_model.pt --data sft_val.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from language_model import LM, SFTDataset, sft_collate, PROMPT_TEMPLATE

BASE_DIR = Path(__file__).resolve().parent


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = LM(
        vocab_size=len(ck["char2idx"]),
        embed_dim=a["embed_dim"],
        hidden_dim=a["hidden_dim"],
        num_layers=a["num_layers"],
        model_type=a["model"],
        dropout=a["dropout"],
        nhead=a.get("nhead", 4),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck["char2idx"], ck["idx2char"]


@torch.no_grad()
def eval_loss(model, ds, batch_size=32):
    """只在回答部分统计（SFTDataset 已把指令段 label 置为 -100）。"""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=sft_collate)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss, correct, tokens = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total_loss += ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        mask = y != -100
        correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tokens += mask.sum().item()
    return total_loss / tokens, correct / tokens


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_new=40):
    """贪婪解码，保证同一 prompt 结果可复现，遇到换行即停。"""
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        return ""
    x = torch.tensor(ids).unsqueeze(0)
    out = []
    for _ in range(max_new):
        nxt = model(x)[:, -1, :].argmax(-1).item()
        ch = idx2char[nxt]
        if ch == "\n":
            break
        out.append(ch)
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    return "".join(out)


def char_f1(pred, gold):
    """字符级 F1：忽略顺序，衡量答对多少字。"""
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p = same / len(pred)
    r = same / len(gold)
    return 2 * p * r / (p + r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(BASE_DIR / "sft_model.pt"))
    parser.add_argument("--data",       default=str(BASE_DIR / "sft_val.jsonl"))
    parser.add_argument("--limit",      type=int, default=100, help="参与生成评测的样本数")
    parser.add_argument("--max_new",    type=int, default=40,  help="最多生成多少个字符")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"checkpoint: {args.checkpoint}")
    model, char2idx, idx2char = load_model(args.checkpoint)

    # 1) 验证集 loss / PPL / 逐字准确率
    ds = SFTDataset(args.data, char2idx, max_len=256)
    loss, token_acc = eval_loss(model, ds, args.batch_size)
    print(f"验证样本: {len(ds):,} 条")
    print(f"loss: {loss:.4f}   PPL: {torch.exp(torch.tensor(loss)).item():.2f}   "
          f"逐字准确率: {token_acc:.2%}")

    # 2) 生成质量
    samples = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[: args.limit]

    exact, f1_sum = 0, 0.0
    rows = []
    for d in samples:
        prompt = PROMPT_TEMPLATE.format(d["instruction"])
        pred = generate(model, char2idx, idx2char, prompt, args.max_new)
        gold = d["output"]
        exact += int(pred == gold)
        f1_sum += char_f1(pred, gold)
        rows.append((d["instruction"], gold, pred))

    n = len(samples)
    print(f"生成评测: {n} 条   完全匹配率: {exact / n:.2%}   平均字符 F1: {f1_sum / n:.2%}")

    print("\n样例（指令 / 标准答案 / 模型输出）:")
    for i, (q, gold, pred) in enumerate(rows[:8], 1):
        print(f"[{i}] 指令: {q}")
        print(f"    标准: {gold}")
        print(f"    生成: {pred}")


if __name__ == "__main__":
    main()

"""
SFT 训练（instruction -> output）。

- 词表里加了 <eos>：<unk>=0，<eos>=1
- 每条样本：instruction\noutput<eos>
- instruction 只当上下文、不参与 loss；只对 output(+<eos>) 算 loss
- "最好" = sft_val loss 最低，创新低保存到 sft_model.pt
- 每轮打印生成正确率：精确 / 包含答案 / 关键答案

保存的是完整模型（权重 + 词表）。后面用：
    model = load_model("test5/sft_model.pt")
    print(model.generate("告诉我世界上第一台电脑的名称和发明者？\n"))
"""
import json
import random
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 让 "test5" 能作为包被导入（命令行直接跑 和 PyCharm 都兼容）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test5.test05 import (DecoderLM, data_handel,
                          batch_size, seq_len, device, seed,
                          SFT_TRAIN_PATH, SFT_VAL_PATH, MODEL_PATH,
                          UNK_ID, EOS_ID, IGNORE_INDEX)

FT_MODEL_PATH = Path(__file__).resolve().parent / "sft_model.pt"   # 训练后保存的模型
ft_epochs = 3        # 训练轮数
ft_lr = 3e-5         # 从预训练继续，用小学习率
max_new = 40         # 生成时最多续写多少字（有 <eos> 后会提前停）
eval_n = 40          # 每轮拿多少条 sft_val 算生成正确率（CPU 上控制时间，GPU 可调大）


class SFTDataset(Dataset):
    """SFT 数据：instruction 作为上下文不参与 loss，只对 output(+<eos>) 算 loss"""
    def __init__(self, samples, char_index):
        self.seq_len = seq_len
        newline = char_index.get("\n", UNK_ID)
        data, mask = [], []            # mask[i]=True 表示第 i 个 token 属于 output/eos，参与 loss
        for instruction, output in samples:
            instr_ids = [char_index.get(c, UNK_ID) for c in instruction]
            out_ids = [char_index.get(c, UNK_ID) for c in output] + [EOS_ID]
            data.extend(instr_ids + [newline] + out_ids)
            mask.extend([False] * (len(instr_ids) + 1) + [True] * len(out_ids))
        self.data = torch.tensor(data, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.seq_len + 1)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1].clone()
        y[~self.mask[start + 1: start + self.seq_len + 1]] = IGNORE_INDEX   # instruction 忽略
        y[y == UNK_ID] = IGNORE_INDEX                                       # <unk> 忽略
        return x, y


def load_sft_samples(path):
    """读取 SFT jsonl，返回 [(instruction, output), ...]"""
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        samples.append((obj["instruction"], obj["output"]))
    return samples


def save_model(path, model, char_index, index_char):
    """保存完整模型：权重 + 词表，后面 load_model 能直接还原使用"""
    torch.save({
        "model_state": model.state_dict(),
        "char_index": char_index,
        "index_char": index_char,
        "vocab_size": len(char_index),
    }, path)


def load_model(path, map_location=None):
    """载入 save_model 存的完整模型，返回 model（词表已挂到 model 上）"""
    if map_location is None:
        map_location = device
    ckpt = torch.load(path, map_location=map_location)
    model = DecoderLM(ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.char_index = ckpt["char_index"]   # 挂到模型上，model.generate() 直接用
    model.index_char = ckpt["index_char"]
    model.eval()
    return model


def extract_key(answer):
    """从参考答案里抽"关键答案"：数字、引号里的内容"""
    keys = re.findall(r"\d+(?:\.\d+)?", answer)
    keys += re.findall(r"[“「\"']([^”」\"']+)[”」\"']", answer)
    return [k for k in keys if k]


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """验证集上的平均 loss（忽略 -100 的位置）"""
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.transpose(1, 2), y)
        valid = int((y != IGNORE_INDEX).sum().item())
        total += loss.item() * valid
        n += valid
    return total / n if n else 0.0


@torch.no_grad()
def eval_generation(model, samples):
    """统计生成正确率：精确 / 包含答案 / 关键答案（只取前 eval_n 条）"""
    model.eval()
    exact = contains = key_match = 0
    for instr, ref in samples[:eval_n]:
        gen = model.generate(instr + "\n").strip()
        ref_s = ref.strip()
        if gen == ref_s:
            exact += 1
        if ref_s in gen:
            contains += 1
        keys = extract_key(ref_s)
        if keys:
            if any(k in gen for k in keys):
                key_match += 1
        else:
            if ref_s in gen:
                key_match += 1
    n = min(len(samples), eval_n)
    return exact / n, contains / n, key_match / n


def main():
    random.seed(seed)
    torch.manual_seed(seed)

    # 用 test05 的 data_handel 拿词表（已经包含 <eos>）
    _, _, char_index, index_char = data_handel()
    vocab_size = len(char_index)
    print(f"device: {device}  vocab: {vocab_size}  <eos> id={EOS_ID}")

    model = DecoderLM(vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))   # 从预训练继续
    model.char_index = char_index
    model.index_char = index_char
    print(f"已加载预训练权重: {MODEL_PATH}")

    train_loader = DataLoader(SFTDataset(load_sft_samples(SFT_TRAIN_PATH), char_index),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(SFTDataset(load_sft_samples(SFT_VAL_PATH), char_index),
                            batch_size=batch_size, shuffle=False, drop_last=True)
    val_samples = load_sft_samples(SFT_VAL_PATH)

    opt = torch.optim.Adam(model.parameters(), lr=ft_lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)   # instruction 位置不参与 loss

    best_val = float("inf")
    for epoch in range(1, ft_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                       # b,t,vcb
            loss = loss_fn(logits.transpose(1, 2), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_loss = evaluate(model, val_loader, loss_fn)
        exact, contains, key_match = eval_generation(model, val_samples)

        if val_loss < best_val:
            best_val = val_loss
            save_model(FT_MODEL_PATH, model, char_index, index_char)
            saved = "  * 已保存"
        else:
            saved = ""

        print(f"epoch {epoch}/{ft_epochs}  val_loss {val_loss:.4f}  "
              f"生成: 精确 {exact:.1%} 包含 {contains:.1%} 关键答案 {key_match:.1%}{saved}")

    print(f"训练完成，最优模型: {FT_MODEL_PATH}")


if __name__ == "__main__":
    main()

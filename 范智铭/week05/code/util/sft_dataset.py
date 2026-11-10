"""SFTDataset：把 instruction/output 组织成「输入 + 标签掩码」。

与预训练数据集（week04 的 util/dataset.py，把整篇语料拼起来切块）不同，
SFT 的一条样本就是一条问答，并且 **只对答案部分计算 loss**：

    输入  : 问：{instruction} 答：{output}<eos>
    标签  : -100 ... -100      |  {output 的 token}  <eos>
             ↑ prompt 部分 mask 掉（-100 = CrossEntropyLoss 忽略）
                                   ↑ 只有这些位置参与反向传播

为什么必须 mask prompt：
  * 模型已经会「读问题」，SFT 要学的是「怎么答」；
  * 如果 prompt 也参与 loss，模型会把大量容量花在复述问题上，
    并且一批里不同题目的长度差异会把损失带偏；
  * 与 prompt.py 的模板严格对应：答：之后的第一个 token 才是监督起点。

长度处理：
  位置编码只有 max_len 个位置，所以单条样本最多 max_len 个 token。
  数据里最长的样本 117 token，正常不会被截断；超长时的策略是
  优先保留 prompt、把答案从尾部截断，并保证答案至少留 1 个 token。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from prompt import format_prompt

# SFT 数据位于 week05/data/ 下；本文件在 week05/code/util/ 下，向上两级到 week05/
ROOT = Path(__file__).resolve().parent.parent.parent

#: CrossEntropyLoss 的 ignore_index，同时用来标记「不计算 loss 的位置」
IGNORE_INDEX = -100

#: 清洗表：不换行空格 / 零宽字符在词表里不存在，会被编码成 <unk>，
#: 这里统一换成普通空格或直接删掉（与 week04 prepare_data.py 的清洗规则一致）。
_CLEAN_TABLE = str.maketrans({"\u00a0": " ", "\u3000": " ", "\u200b": None, "\ufeff": None})


def normalize_text(text: str) -> str:
    """清洗 SFT 文本：NBSP → 空格、去掉零宽字符，并去掉首尾空白。

    训练、验证、推理三处共用，保证模型看到的文本分布一致。
    """
    return str(text).translate(_CLEAN_TABLE).strip()


def encode_example(
    tok,
    instruction: str,
    output: str,
    max_len: int,
) -> tuple[list[int], list[int], bool]:
    """把一条问答编码成 (input_ids, labels, truncated)。

    Args:
        tok: CharTokenizer。
        instruction: 问题。
        output: 答案。
        max_len: 单条样本最大 token 数（= 位置编码长度）。

    Returns:
        input_ids: 送进模型的 token 序列（prompt + 答案 + <eos>）。
        labels:    与 input_ids 等长，prompt 部分为 IGNORE_INDEX。
        truncated: 是否因为超长被截断（用于数据统计）。
    """
    instruction, output = normalize_text(instruction), normalize_text(output)
    prompt_ids = tok.encode(format_prompt(instruction))
    answer_ids = tok.encode(output) + [tok.eos_id]

    # prompt 太长：先截 prompt，给答案留出至少 1 个 token
    if len(prompt_ids) > max_len - 1:
        prompt_ids = prompt_ids[: max_len - 1]

    # 答案太长：从尾部截断，但至少保留 1 个 token
    room = max_len - len(prompt_ids)
    truncated = len(answer_ids) > room
    answer_ids = answer_ids[:room]

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return input_ids, labels, truncated


class SFTDataset(Dataset):
    """读取 sft_train.jsonl / sft_val.jsonl，产出错位一位的 (x, y)。

    内部保存完整的 ``(input_ids, labels)`` 用于统计（长度、监督 token 数），
    ``__getitem__`` 返回 ``(input_ids[:-1], labels[1:])`` 供模型做 next-token 预测。
    """

    def __init__(self, cfg, tok, split: str):
        if split not in ("train", "val"):
            raise ValueError(f"split 必须是 'train' 或 'val'，收到 {split!r}")

        self.cfg = cfg
        self.tok = tok
        self.max_len: int = cfg.hparams.max_len
        self.split = split

        filename = (
            cfg.hparams.sft_train_filename
            if split == "train"
            else cfg.hparams.sft_val_filename
        )
        data_path = ROOT / "data" / filename
        if not data_path.exists():
            raise FileNotFoundError(f"找不到 SFT 数据文件：{data_path}")

        self.records: list[dict] = []
        with data_path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{data_path.name}:{lineno} 不是合法 JSON：{exc}") from exc
                for key in ("instruction", "output"):
                    if key not in rec:
                        raise KeyError(f"{data_path.name}:{lineno} 缺少字段 {key!r}")
                self.records.append(rec)

        # 预编码：整条数据只做一次 tokenize，训练时零开销
        self.examples: list[tuple[list[int], list[int]]] = []
        self.truncated = 0
        for rec in self.records:
            input_ids, labels, truncated = encode_example(
                tok, rec["instruction"], rec["output"], self.max_len
            )
            self.examples.append((input_ids, labels))
            self.truncated += int(truncated)

        if not self.examples:
            raise RuntimeError(f"{data_path} 里没有有效样本")

    # ------------------------------------------------------------------ 统计
    @property
    def num_answer_tokens(self) -> int:
        """参与 loss 的 token 总数（答案 + <eos>）。"""
        return sum(sum(1 for t in labels if t != IGNORE_INDEX) for _, labels in self.examples)

    @property
    def num_tokens(self) -> int:
        return sum(len(ids) for ids, _ in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        """返回错位一位的 (x, y)：x = ids[:-1]，y = labels[1:]。

        因果语言模型的 logits[t] 预测的是「第 t+1 个 token」，所以必须错位：
        输入 ``问：... 答：`` 的最后一个位置负责预测答案的第一个字。
        忘记错位是最常见的 SFT bug —— loss 会虚高且模型永远学不会。
        """
        input_ids, labels = self.examples[idx]
        return (
            torch.tensor(input_ids[:-1], dtype=torch.long),
            torch.tensor(labels[1:], dtype=torch.long),
        )


def make_collate_fn(tok):
    """批次内右侧 padding 到最长样本。

    因果注意力的特性保证了「右侧 pad」不会污染真实 token：
    真实 token 只看自己左边，pad 出现在它们后面，永远不会被看到。
    所以这里不需要 attention mask，只需要把 pad 位置的 label 设成 IGNORE_INDEX。
    """

    def collate(batch):
        max_len = max(len(input_ids) for input_ids, _ in batch)
        x = torch.full((len(batch), max_len), tok.pad_id, dtype=torch.long)
        y = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        for i, (input_ids, labels) in enumerate(batch):
            # __getitem__ 已经返回 tensor，直接原地拷贝，避免 torch.tensor(tensor) 的告警
            n = input_ids.numel()
            x[i, :n] = input_ids
            y[i, :n] = labels
        return x, y

    return collate


def preview(tok, instruction: str, output: str, max_len: int = 128) -> str:
    """把一条样本的 input/labels 可视化，方便检查 mask 是否正确。"""
    input_ids, labels, _ = encode_example(tok, instruction, output, max_len)
    tokens = [tok.itos[i] for i in input_ids]
    shown_tokens = [
        f"[{t}]" if lab == IGNORE_INDEX else t for t, lab in zip(tokens, labels)
    ]
    n_sup = sum(1 for lab in labels if lab != IGNORE_INDEX)
    return (
        f"全长 {len(input_ids)} token（其中 {n_sup} 个参与 loss，"
        f"{len(input_ids) - n_sup} 个被 mask）\n"
        + "".join(shown_tokens)
    )


__all__ = [
    "IGNORE_INDEX",
    "SFTDataset",
    "encode_example",
    "make_collate_fn",
    "normalize_text",
    "preview",
]

"""chat template 与 loss mask —— 整个项目最核心的 30 行。

SFT 改的不是模型结构，而是**数据格式**和**哪些位置算 loss**：

    <|user|> 什么是人工智能？ <|assistant|> 人工智能是…… <|end|>
    └────────── prompt ──────────┘└────── answer ──────┘

    对应的 labels（-100 表示不参与 loss）：
    -100 -100 -100 ... -100        │ 人 工 智 能 是 ... <|end|>

为什么 prompt 部分要 mask 掉？
    不做 mask 的话，模型会花同样的力气去学「怎么复述问题」——它本来就会。
    显存和算力就那么多，把梯度全砸在「怎么答」上，样本效率才高。
    这也是 PPT 第 28 页「只学怎么答，不背问什么」的代码形态。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tokenizer import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    USER_TOKEN,
    CharTokenizer,
)

# -100 是 PyTorch cross_entropy 的 ignore_index 默认值：label 等于 -100 的位置
# 既不产生 loss，也不产生梯度。整套 mask 机制就靠这一个约定。
IGNORE_INDEX = -100


@dataclass
class Example:
    """编码后的一条训练样本，附带分段信息（画 mask 可视化时要用）。"""

    ids: list[int]
    labels: list[int]
    prompt_len: int
    answer_len: int
    instruction: str = ""
    output: str = ""

    @property
    def supervised_ratio(self) -> float:
        return self.answer_len / max(1, len(self.ids))

    def to_dict(self) -> dict:
        return {
            "ids": self.ids,
            "labels": self.labels,
            "prompt_len": self.prompt_len,
            "answer_len": self.answer_len,
            "instruction": self.instruction,
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(**d)


def build_prompt(tok: CharTokenizer, instruction: str) -> list[int]:
    """推理时用的前缀：<|user|>问题<|assistant|>，模型从这里开始往下生成。"""
    return tok.encode(f"{USER_TOKEN}{instruction.strip()}{ASSISTANT_TOKEN}")


def encode_example(tok: CharTokenizer, instruction: str, output: str) -> Example:
    """把一条 (问题, 回答) 编成 ids + labels，并把 prompt 段全部 mask 掉。"""
    prompt_ids = build_prompt(tok, instruction)
    # 回答末尾一定要补 <|end|>：它是「说完了」的信号，没它模型学会的是永不停止地往下编。
    answer_ids = tok.encode(f"{output.strip()}{END_TOKEN}")

    ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)

    return Example(
        ids=ids,
        labels=labels,
        prompt_len=len(prompt_ids),
        answer_len=len(answer_ids),
        instruction=instruction,
        output=output,
    )


def render_template(tok: CharTokenizer, instruction: str, output: str) -> str:
    """把一条样本还原成「模型看到的原始字符串」，用于肉眼核对格式。"""
    return f"{USER_TOKEN}{instruction.strip()}{ASSISTANT_TOKEN}{output.strip()}{END_TOKEN}"


def describe_mask(tok: CharTokenizer, ex: Example) -> list[tuple[int, str, bool]]:
    """给 mask 可视化用的三元组列表：(位置, token 字符串, 是否参与 loss)。"""
    rows: list[tuple[int, str, bool]] = []
    for i, tid in enumerate(ex.ids):
        rows.append((i, tok.itos[tid], ex.labels[i] != IGNORE_INDEX))
    return rows

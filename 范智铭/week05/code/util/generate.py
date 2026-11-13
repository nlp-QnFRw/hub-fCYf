"""推理侧工具：SFT 模型的「问 → 答」生成与答案困惑度。

训练脚本（sft_train.py）与交互 demo（demo.py）共用这一份实现，
保证两边看到的 prompt 模板、截断方式、停止条件完全一致。

关键约定（来自 prompt.py）：
  * 推理输入前缀 = ``问：{instruction} 答：``
  * 生成到 ``<eos>`` 即停；
  * 位置编码只有 max_len 个位置，生成时输入窗口自动左移。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from prompt import format_prompt
from util.sft_dataset import IGNORE_INDEX, encode_example


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None) -> int:
    """按 temperature / top-k / top-p 从 logits 中采样一个 token id。"""
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # 保留累计概率首次超过 top_p 之前（含）的 token，至少保留 1 个
        cut = int(torch.searchsorted(cumulative, torch.tensor(top_p, device=probs.device)).item()) + 1
        keep = sorted_idx[:cut]
        mask = torch.full_like(logits, float("-inf"))
        mask[keep] = logits[keep]
        logits = mask

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_answer(
    model,
    tok,
    instruction: str,
    device,
    max_len: int = 128,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> str:
    """给定问题，自回归生成答案（不包含 ``答：`` 前缀和 ``<eos>`` 本身）。

    Args:
        temperature: 0 表示贪心解码（评测用），>0 时按概率采样（demo 用）。
        top_k / top_p: 采样时的截断策略，None 表示不启用。
        repetition_penalty: >1 时抑制已生成字符的重复（字符级模型容易「的的的」）。
    """
    model.eval()
    ids = tok.encode(format_prompt(instruction))
    # prompt 过长时先截断，保证至少留 1 个位置给答案
    if len(ids) > max_len - 1:
        ids = ids[: max_len - 1]

    generated: list[int] = []
    # 特殊 token 不能出现在答案里，生成前把它们屏蔽掉
    banned = [tok.pad_id, tok.unk_id, tok.bos_id]

    for _ in range(max_new_tokens):
        x = torch.tensor(ids[-max_len:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1, :].float()

        for tid in banned:
            logits[tid] = float("-inf")

        if repetition_penalty and repetition_penalty > 1.0 and generated:
            history = torch.tensor(sorted(set(generated)), device=logits.device)
            picked = logits[history]
            logits[history] = torch.where(
                picked > 0, picked / repetition_penalty, picked * repetition_penalty
            )

        next_id = _sample_next(logits, temperature, top_k, top_p)
        if next_id == tok.eos_id:  # 遇到终止符，停止生成
            break

        ids.append(next_id)
        generated.append(next_id)

    return tok.decode(generated, skip_special=True)


@torch.no_grad()
def answer_loss(
    model,
    tok,
    instruction: str,
    output: str,
    device,
    max_len: int = 128,
) -> tuple[float, int]:
    """只用答案 token 计算平均交叉熵（与训练时的 mask 方式完全一致）。

    Returns:
        (每个答案 token 的平均 loss, 参与计算的 token 数)
    """
    model.eval()
    input_ids, labels, _ = encode_example(tok, instruction, output, max_len)
    x = torch.tensor(input_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(labels[1:], dtype=torch.long, device=device).unsqueeze(0)

    logits = model(x)
    total = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    n = int((y != IGNORE_INDEX).sum().item())
    if n == 0:
        return float("inf"), 0
    return total.item() / n, n


@torch.no_grad()
def answer_perplexity(
    model,
    tok,
    instruction: str,
    output: str,
    device,
    max_len: int = 128,
) -> float:
    """答案部分的 perplexity（越低说明模型越认同这个答案）。"""
    loss, n = answer_loss(model, tok, instruction, output, device, max_len)
    if n == 0:
        return float("inf")
    return math.exp(loss)


__all__ = [
    "generate_answer",
    "answer_loss",
    "answer_perplexity",
]

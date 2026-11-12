"""生成 / 采样。

和 `web/generation_lab.html` 里讲的选词策略一一对应：
    temperature        改分布的「陡峭程度」（T→0 接近 argmax，T 越大越平）
    top_k              只在前 k 个候选里采样
    top_p              按累积概率截断，候选集大小随分布自适应
    repetition_penalty 对已经出现过的 token 降权，缓解小模型「复读」

本项目没有实现 KV Cache：序列只有两三百个 token，省下来的时间还不如代码清晰值钱。
真正的 cache 逻辑见 PPT 第 13 页的动图。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .model import GPT
from .template import build_prompt
from .tokenizer import CharTokenizer


def _apply_repetition_penalty(logits: torch.Tensor, ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """对出现过的 token：正分数除以 penalty，负分数乘以 penalty，都会「变差一点」。"""
    if penalty == 1.0:
        return logits
    for b in range(ids.size(0)):
        for token in set(ids[b].tolist()):
            if logits[b, token] > 0:
                logits[b, token] /= penalty
            else:
                logits[b, token] *= penalty
    return logits


@torch.no_grad()
def sample(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_ids: Sequence[int] = (),
) -> torch.Tensor:
    """自回归循环：每步只出一个 token，再拼回输入继续。返回完整 ids（含 prompt）。"""
    model.eval()
    stop = set(stop_ids)

    for _ in range(max_new_tokens):
        # 超过 block_size 就得截断左边 —— 模型没有位置编码能表示更远的位置
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # 只关心最后一个位置的预测

        logits = _apply_repetition_penalty(logits, idx, repetition_penalty)

        if temperature <= 0:
            # 贪心：每步取概率最大的那个，结果完全可复现，但容易陷入复读
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                # 从高到低累加概率，够 top_p 就截断：候选集大小由分布自己决定
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        # 只要 batch 里有一条命中结束符就整体停下（batch=1 时正好是「答完即停」）
        if stop and next_id.item() in stop:
            break

    return idx


@torch.no_grad()
def chat(
    model: GPT,
    tok: CharTokenizer,
    instruction: str,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: torch.device | str = "cpu",
) -> str:
    """SFT 模型的用法：套上模板问一句，生成到 <|end|> 为止。"""
    ids = build_prompt(tok, instruction)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = sample(
        model,
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_ids=[tok.end_id],
    )
    new_ids = out[0, len(ids) :].tolist()
    # 结束符本身不显示给用户
    text = tok.decode([i for i in new_ids if i != tok.end_id], skip_special=True)
    return text.strip()


@torch.no_grad()
def raw_continue(
    model: GPT,
    tok: CharTokenizer,
    text: str,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: torch.device | str = "cpu",
) -> str:
    """Base Model 的用法：不给模板、不给结束符，纯续写 —— 它根本不知道该停。"""
    ids = tok.encode(text)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = sample(
        model,
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_ids=[],  # 没有停止符：这正是「续写器」和「助手」的差别之一
    )
    return tok.decode(out[0, len(ids) :].tolist(), skip_special=True)

"""demo.py —— 训练好的字符级 Transformer 的实际应用示例。

用法::

    python code/demo.py                    # 默认：用 1 epoch 的小模型跑 4 个例子
    python code/demo.py --ckpt checkpoints/best.pt

三个场景：
  1. 续写（给定开头，模型生成后面的字）
  2. 判断通顺度（算一句话的 perplexity，越低越像人话）
  3. 交互式（一直让你输入开头，直到 Ctrl+C）
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# 让 import 能找到 code/ 和 code/util/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "util"))

from my_module import MyModule  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


# ---------- 核心推理逻辑 ----------

@torch.no_grad()
def generate(model, tok, prompt: str, device, max_new: int = 50, temperature: float = 0.8):
    """给定开头 prompt，模型自回归往后生成 max_new 个字符。

    Args:
        temperature: >1 更随机，<1 更保守，0 近似贪心
    """
    model.eval()
    ids = tok.encode(prompt)   # 开头编码
    for _ in range(max_new):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        T = x.shape[1]
        # 取最后一个位置的 logits，只关心"下一个字"的预测
        logits = model(x)[:, -1, :] / temperature   # [1, vocab]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()   # 按概率采样
        ids.append(next_id)
        if next_id == tok.eos_id:   # 生成到 <eos> 就停
            break
    return tok.decode(ids, skip_special=True)


@torch.no_grad()
def perplexity(model, tok, text: str, device) -> float:
    """算一句话的 perplexity（越低 = 模型觉得这句话越像真的）。"""
    model.eval()
    ids = tok.encode(text)
    if len(ids) < 2:
        return float("inf")
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)                                    # [1, T, V]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=tok.pad_id,
    )
    return math.exp(loss.item())


# ---------- 三种演示 ----------

def demo_continuation(model, tok, device):
    print("=" * 60)
    print("【场景1】续写：给开头，模型往后写")
    print("=" * 60)

    prompts = [
        "如何才能",
        "请问",
        "今天天气",
        "微信",
    ]

    for prompt in prompts:
        result = generate(model, tok, prompt, device, max_new=40, temperature=0.8)
        print(f"  开头: {prompt}")
        print(f"  续写: {result}")
        print()


def demo_perplexity(model, tok, device):
    print("=" * 60)
    print("【场景2】通顺度判断：perplexity 越低越像人话")
    print("=" * 60)

    sentences = [
        "如何破解别人邮箱密码？",
        "请问您有大陆开户的银行卡吗？",
        "今 天 天 气 不 错",         # 故意加空格，OOV 更多
        "xyzxyz 明天天气怎么样？",     # 开头乱码
        "明天天气怎么样？",           # 自然句子
    ]

    for s in sentences:
        pp = perplexity(model, tok, s, device)
        flag = "✓ 通顺" if pp < 500 else "✗ 可疑"
        print(f"  {pp:8.1f}  {flag}  {s}")


def demo_interactive(model, tok, device):
    print("=" * 60)
    print("【场景3】交互式：输入开头，模型续写（Ctrl+C 退出）")
    print("=" * 60)
    try:
        while True:
            prompt = input("请输入开头 >>> ").strip()
            if not prompt:
                continue
            result = generate(model, tok, prompt, device, max_new=60, temperature=0.8)
            print(f"  -> {result}")
            pp = perplexity(model, tok, prompt, device)
            print(f"     这句 perplexity: {pp:.1f}")
    except KeyboardInterrupt:
        print("\n再见 👋")


# ---------- 入口 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--cfg", type=str, default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    args = parser.parse_args()

    # 加载配置和 checkpoint
    cfg = OmegaConf.load(args.cfg)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"找不到 checkpoint：{ckpt_path}\n请先跑 train.py 训练出模型")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer.from_file(cfg)

    # 用当前 config 创建模型 → 加载权重 → 移到 device
    model = MyModule(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"加载 checkpoint: {ckpt_path}")
    print(f"  val_loss={ckpt.get('val_loss', '?'):.4f}, epoch={ckpt.get('epoch', '?')}")
    print(f"  device={device}, vocab={len(tok)}")
    print()

    if args.interactive:
        demo_interactive(model, tok, device)
    else:
        demo_continuation(model, tok, device)
        demo_perplexity(model, tok, device)
        print("\n交互式加上 --interactive 参数")


if __name__ == "__main__":
    main()

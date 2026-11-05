"""
generate.py -- 用训练好的 checkpoint 续写文本

用法:
    python generate.py                                   # 默认提示词，采样 80 个字
    python generate.py --prompt "今天天气" --max-new-tokens 120
    python generate.py --ckpt runs/best.pt --temperature 0.8 --top-p 0.9
    python generate.py --num 3 --top-k 20                # 采 3 条

采样参数怎么选：
    temperature 越低越保守（接近贪心），越高越随机；
    top_k 只从概率最高的 k 个里抽，top_p 从累计概率到 p 的最小集合里抽；
    两个都设时先 top_k 再 top_p。贪心解码（temperature 极小）容易复读，一般不用。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

import data as data_mod
from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Transformer 语言模型采样")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "best.pt"))
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--prompt", default="", help="留空则从句子开头（换行符）起笔")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40, help="0 表示关闭")
    ap.add_argument("--top-p", type=float, default=0.95, help="1.0 表示关闭")
    ap.add_argument("--num", type=int, default=1, help="生成几条")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"找不到 checkpoint {args.ckpt}，请先运行 train.py")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens 必须 >= 1")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig.from_dict(ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    stoi, itos = data_mod.load_vocab(args.data_dir)
    meta = data_mod.load_meta(args.data_dir)
    if cfg.vocab_size != len(itos):
        raise SystemExit(f"checkpoint 字表({cfg.vocab_size}) 与 data/vocab.json({len(itos)}) 不一致，"
                         f"请确认用的是同一份数据")

    prompt = args.prompt
    unk = stoi[data_mod.UNK]
    oov = [c for c in prompt if c not in stoi]
    if oov:
        print(f"提示词里有 {len(oov)} 个未登录字，将按 [UNK] 处理: {''.join(sorted(set(oov)))}",
              file=sys.stderr)

    if prompt:
        ids = data_mod.encode(prompt, stoi).astype("int64")
        idx = torch.from_numpy(ids).unsqueeze(0).to(device)
    else:
        # 语料里句子是用换行分隔的，所以从换行符起笔等价于「开始一个新句子」
        start = stoi.get("\n", unk)
        idx = torch.tensor([[start]], dtype=torch.long, device=device)

    # 采样用的随机数发生器必须和 logits 在同一个设备上
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"模型: {args.ckpt}  step={ckpt.get('step')}  "
          f"{cfg.n_layer}层 {cfg.n_embd}维  字表 {len(itos)}")
    if meta.get("baseline_ppl"):
        print("基线 PPL: " + "  ".join(f"{k}-gram={v}" for k, v in meta["baseline_ppl"].items()))

    top_k = args.top_k or None
    top_p = args.top_p if args.top_p < 1.0 else None

    for i in range(max(1, args.num)):
        out = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                             top_k=top_k, top_p=top_p, generator=gen)
        text = data_mod.decode(out[0, len(idx[0]):].tolist(), itos)
        print(f"\n--- 采样 {i + 1} ---")
        print(f"{prompt}{text}".replace("\n", "\n"))

    if meta.get("stats", {}).get("newline_ratio_pct"):
        print(f"\n注：训练语料里换行符占 {meta['stats']['newline_ratio_pct']}%，"
              f"所以模型会时不时输出换行——那是它在模仿语料的分句习惯。")


if __name__ == "__main__":
    main()

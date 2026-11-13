# -*- coding: utf-8 -*-
"""
demo_transformer_continuation.py —— SFT 训练完的演示层：对比「参考回答」和「模型回答」

【它做什么】
    加载 language_model_transformer.py（SFT 版）训出来的 ckpt，用同一套 chat 模板推理：
      1) 默认模式：取验证集前 N 条，把「指令 / 参考回答 / 模型回答」并排列出来
      2) 交互模式：课上一句一句输入指令，模型现场回答
      3) 顺带算三个指标：回答段 PPL、token 准确率、完全匹配率

【为什么演示层要单独一个文件】
    训练脚本只关心 loss；演示关心的是「人怎么看」。放在一起会让训练脚本越来越长，
    也容易在改演示时把训练逻辑改坏。所以这里只 import，不复制模型代码。

用法:
    python demo_transformer_continuation.py                       # 看验证集前 5 条
    python demo_transformer_continuation.py --n 10                # 看 10 条
    python demo_transformer_continuation.py --interactive         # 交互模式
    python demo_transformer_continuation.py --temperature 0.7 --top_k 40   # 采样，答案更多样
    python demo_transformer_continuation.py --ckpt sft_best.pt --no_metrics
"""

import argparse
import os
import random
import unicodedata

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from language_model_transformer import (
    TransformerLM, SFTDataset, collate, decode, encode, generate, load_jsonl,
    pick_device, run_epoch, torch_load, ASSISTANT, END, PAD, SPECIALS, USER,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _dw(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def load_model(path, device):
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(HERE, path)
    ck = torch_load(path, "cpu")
    a = ck["args"]
    model = TransformerLM(len(ck["char2idx"]), a["d_model"], a["heads"], a["layers"],
                          a["seq_len"], 0.0, use_pos=True, causal=True).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def resolve(path):
    """ckpt 里记的相对路径按脚本目录解析。"""
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(HERE, path)


@torch.no_grad()
def metrics(model, rows, char2idx, seq_len, batch_size, device):
    """整体指标：回答段 loss / PPL / token 准确率（都不含 prompt 段与 padding）。"""
    ds = SFTDataset(rows, char2idx, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                    drop_last=False)
    return run_epoch(model, dl, None, device, train=False)


def exact_match(model, rows, char2idx, idx2char, max_new, device):
    """完全匹配率：贪心生成的回答和参考回答一模一样。小模型上通常很低，属正常。"""
    hit = 0
    for r in rows:
        ans = generate(model, char2idx, idx2char, r["instruction"], max_new, 0.0, 0, device, 1.3)
        if ans.strip() == r["output"].strip():
            hit += 1
    return hit / max(1, len(rows))


def show_one(model, char2idx, idx2char, row, args, device, idx, total):
    ans = generate(model, char2idx, idx2char, row["instruction"], args.max_new,
                   args.temperature, args.top_k, device)
    print("\n" + "─" * 78)
    print(f"[{idx}/{total}] 指令：{row['instruction']}")
    print(f"  参考：{row['output']}")
    print(f"  模型：{ans if ans.strip() else '（空回答）'}")
    ok = ans.strip() == row["output"].strip()
    print(f"  一致：{'是' if ok else '否'}　长度 {len(ans)} 字")
    print("─" * 78)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "sft_best.pt"))
    ap.add_argument("--val-file", default=None, help="默认用 ckpt 里记录的验证集")
    ap.add_argument("--n", type=int, default=5, help="展示多少条")
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = 贪心（推荐演示用，结果确定）")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.3,
                    help="重复惩罚，1.0 = 关闭；小模型贪心容易复读，1.2~1.4 效果明显")
    ap.add_argument("--no_metrics", action="store_true", help="跳过整体指标（更快）")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    model, ck = load_model(args.ckpt, device)
    a, c2i, i2c = ck["args"], ck["char2idx"], ck["idx2char"]
    print(f"ckpt: {os.path.basename(args.ckpt)}　参数量 "
          f"{sum(p.numel() for p in model.parameters()):,}　字表 {len(c2i)}　"
          f"seq_len {a['seq_len']}")

    # 验证集路径：命令行优先，否则用训练时记录的
    val_path = args.val_file or os.path.join(resolve(a.get("data_dir")) or HERE,
                                             a.get("val_file", "sft_val.jsonl"))
    val_rows = load_jsonl(val_path) if os.path.exists(val_path) else []
    if not val_rows:
        print(f"[提示] 找不到验证集 {val_path}，只做交互/空跑。")

    if val_rows and not args.no_metrics:
        loss, ppl, acc = metrics(model, val_rows, c2i, a["seq_len"], 32, device)
        em = exact_match(model, val_rows[:50], c2i, i2c, args.max_new, device)
        print(f"\n验证集指标（{len(val_rows)} 条）")
        print(f"  回答段 loss = {loss:.4f}　PPL = {ppl:.2f}　token 准确率 = {acc:.3f}")
        print(f"  完全匹配率（前 50 条，贪心）= {em:.1%}")
        print("  说明：token 准确率是「逐字猜对的比例」，完全匹配要求整句一致，后者天然低得多。")

    if args.interactive:
        print("\n交互模式：输入指令，回车看回答；直接回车退出。")
        while True:
            try:
                q = input("\n指令> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            ans = generate(model, c2i, i2c, q, args.max_new, args.temperature,
                           args.top_k, device, args.rep_penalty)
            print(f"回答：{ans if ans.strip() else '（空回答）'}")
        return

    rows = val_rows[:args.n]
    if not rows:
        return
    print(f"\n对比 {len(rows)} 条（temperature={args.temperature}，"
          f"top_k={args.top_k or '不限'}）")
    ok = 0
    for i, r in enumerate(rows, 1):
        ok += show_one(model, c2i, i2c, r, args, device, i, len(rows))
    print(f"\n本组一致：{ok}/{len(rows)}")


if __name__ == "__main__":
    main()

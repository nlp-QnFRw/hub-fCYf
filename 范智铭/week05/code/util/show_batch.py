"""show_batch.py —— 把训练时真正喂给模型的 (x, y) 打印出来。

回答三个问题：

1. 一条 59 token 的样本（12 个参与 loss、47 个被 mask），x / y 到底长什么样？
2. 为什么 x 比原句少一个 token、y 少一个 token，它们怎么和原句对上的？
3. 组成一个 batch 后 padding 是什么样、loss 在哪些位置计算？

用法::

    python code/util/show_batch.py              # 打印第 1 条样本 + 2 条样本的 batch
    python code/util/show_batch.py --idx 5      # 换一条样本
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "util"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

from omegaconf import OmegaConf  # noqa: E402

from my_module import MyModule  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from util.sft_dataset import (  # noqa: E402
    IGNORE_INDEX,
    SFTDataset,
    make_collate_fn,
    normalize_text,
    preview,
)

PAD = -1  # 仅用于打印：表示 padding 位置（真实 pad_id 是 0）


def fmt(tok, ids) -> str:
    """把 id 列表转成 '字(id)' 形式，跳过特殊 token 只显示 id。"""
    out = []
    for i in ids:
        if i == IGNORE_INDEX:
            out.append("-100")
        elif i == PAD:
            out.append("<pad>")
        else:
            ch = tok.itos[i]
            out.append(f"{ch}({i})" if i > 3 else f"{ch}({i})")
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "sft_best.pt"))
    parser.add_argument("--idx", type=int, default=0, help="看第几条训练样本")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["cfg"])  # 结构与权重一致
    tok = CharTokenizer.from_file(model_cfg)
    model = MyModule(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SFTDataset(cfg, tok, "train")
    collate = make_collate_fn(tok)

    # ---------------------------------------------------------------- 1. 单条样本
    rec = ds.records[args.idx]
    input_ids, labels = ds.examples[args.idx]
    x, y = ds[args.idx]

    print("=" * 78)
    print(f"【一条样本】训练集第 {args.idx} 条 | 原句 {len(input_ids)} token")
    print("=" * 78)
    print(f"问题：{normalize_text(rec['instruction'])[:70]}")
    print(f"答案：{normalize_text(rec['output'])[:70]}")
    print()
    print("模板拼出的完整文本（59 token，末尾是 <eos>）：")
    print("  " + preview(tok, rec["instruction"], rec["output"], cfg.hparams.max_len).replace("\n", "\n  "))

    n_prompt = sum(1 for t in labels if t == IGNORE_INDEX)
    n_sup = len(labels) - n_prompt
    print(f"\n  原句 {len(input_ids)} 个位置：前 {n_prompt} 个是 prompt（label=-100），后 {n_sup} 个是答案（label=token id）")
    print(f"  x = input_ids[:-1] → {len(x)} 个 token")
    print(f"  y = labels[1:]     → {len(y)} 个 token，其中 {int((y != IGNORE_INDEX).sum())} 个参与 loss")

    # ---------------------------------------------------------------- 2. 对齐表
    print("\n" + "=" * 78)
    print("【对齐关系】模型在位置 p 的输出，用来预测原句的第 p+1 个 token")
    print("=" * 78)
    print(f"{'p':>3} | {'x[p] = 原句第 p 个 token':<28} | {'y[p] = 原句第 p+1 个 token':<28} | 算 loss?")
    print("-" * 78)

    lo, hi = max(0, n_prompt - 4), min(len(y), n_prompt + 5)
    if lo > 0:
        print(f"{'...':>3} | {'...':<28} | {'...':<28} |")
    for p in range(lo, hi):
        xt = f"{tok.itos[x[p]]}({x[p]})"
        yt = tok.itos[y[p]] if y[p] != IGNORE_INDEX else "-100"
        yt = f"{yt}({y[p]})" if y[p] != IGNORE_INDEX else yt
        mark = "✔" if y[p] != IGNORE_INDEX else ""
        print(f"{p:>3} | {xt:<28} | {yt:<28} | {mark}")
    if hi < len(y):
        print(f"{'...':>3} | {'...':<28} | {'...':<28} |")

    print(f"\n  关键：p={n_prompt - 1} 时，x 是「答：」的最后一个字，y 是答案的第一个字。")
    print(f"  模型在最后一个 prompt 位置输出 logits，正好和答案第一个 token 比较。")

    # ---------------------------------------------------------------- 3. 逐位置 loss
    print("\n" + "=" * 78)
    print("【实际算 loss】模型在这条样本上，每个答案 token 的 loss 与预测")
    print("=" * 78)
    with torch.no_grad():
        logits = model(x.unsqueeze(0))[0]  # (T, V)
        losses = F.cross_entropy(
            logits, y, ignore_index=IGNORE_INDEX, reduction="none"
        )
        total = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX, reduction="sum").item()
    print(f"{'p':>3} | {'正确 token':<12} | {'loss':>7} | 模型最想输出的 top3")
    print("-" * 78)
    sup_positions = [p for p in range(len(y)) if y[p] != IGNORE_INDEX]
    for p in sup_positions:
        top3 = torch.topk(logits[p], 3).indices.tolist()
        top3_s = ", ".join(f"{tok.itos[i]}({i})" for i in top3)
        print(f"{p:>3} | {tok.itos[y[p]]:<12} | {losses[p].item():>7.3f} | {top3_s}")
    print(f"\n  sum(loss) = {total:.4f}，除以 {len(sup_positions)} 个 token = {total / len(sup_positions):.4f}")
    print(f"  （训练时就是 cross_entropy 默认的 mean：对 12 个位置取平均）")

    # ---------------------------------------------------------------- 4. batch 与 padding
    print("\n" + "=" * 78)
    print("【组成 batch】两条不同长度的样本 → 右侧 padding")
    print("=" * 78)
    # 找一条更短的样本，方便看到 padding
    short_idx = next(
        (i for i in range(len(ds)) if i != args.idx and len(ds.examples[i][0]) <= 50), None
    )
    batch_items = [ds[args.idx], ds[short_idx]] if short_idx is not None else [ds[args.idx]]
    bx, by = collate(batch_items)
    print(f"  样本 A：原句 {len(ds.examples[args.idx][0])} token → x {len(ds[args.idx][0])}")
    if short_idx is not None:
        print(f"  样本 B：原句 {len(ds.examples[short_idx][0])} token → x {len(ds[short_idx][0])}")
    print(f"  collate 后：x {tuple(bx.shape)}，y {tuple(by.shape)}，T = 这批最长 = {bx.shape[1]}")

    for b in range(bx.shape[0]):
        valid = int((bx[b] != tok.pad_id).sum())
        sup = int((by[b] != IGNORE_INDEX).sum())
        pads = bx.shape[1] - valid
        print(
            f"  样本 {chr(65 + b)}：有效 x {valid} 个，右侧 pad {pads} 个（pad_id={tok.pad_id}），"
            f"监督 token {sup} 个"
        )

    print("\n  样本 B 的尾部（看 pad 如何填在右边）：")
    b = bx.shape[0] - 1
    tail = range(max(0, bx.shape[1] - 8), bx.shape[1])
    print("  " + " ".join(
        ("<pad>" if bx[b, p].item() == tok.pad_id else tok.itos[bx[b, p].item()]) for p in tail
    ))
    print("  " + " ".join(
        ("-100" if by[b, p].item() == IGNORE_INDEX else str(by[b, p].item())) for p in tail
    ))
    print("\n  padding 位置：x 是 pad_id=0，y 是 -100 → 既不参与 loss，也不会被因果 mask 之前的 token 看到")

    # ---------------------------------------------------------------- 5. 真实 batch 32
    print("\n" + "=" * 78)
    print("【真实训练 batch】前 32 条样本一起 collate")
    print("=" * 78)
    big_x, big_y = collate([ds[i] for i in range(32)])
    valid_lens = [int((r != tok.pad_id).sum()) for r in big_x]
    print(f"  x {tuple(big_x.shape)} | y {tuple(big_y.shape)}")
    print(f"  有效长度：最短 {min(valid_lens)}，最长 {max(valid_lens)}，平均 {sum(valid_lens) / len(valid_lens):.1f}")
    print(f"  参与 loss 的 token：{int((big_y != IGNORE_INDEX).sum())} / {big_x.numel()} ({int((big_y != IGNORE_INDEX).sum()) / big_x.numel():.1%})")

    with torch.no_grad():
        big_logits = model(big_x)
        big_loss = F.cross_entropy(
            big_logits.reshape(-1, big_logits.size(-1)), big_y.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    print(f"  logits {tuple(big_logits.shape)} → reshape {(big_logits.shape[0] * big_logits.shape[1], big_logits.size(-1))}")
    print(f"  训练 loss = {big_loss.item():.4f}（这就是日志里 step_loss 的算法）")


if __name__ == "__main__":
    main()

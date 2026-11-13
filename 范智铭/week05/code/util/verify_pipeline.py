"""verify_pipeline.py —— SFT 管线自检：让模型在 32 条样本上过拟合。

这是排查 SFT「loss 不降 / 答非所问」最快的办法：

  * 若 300 步内 loss 能降到 0.1 以下  → 数据、掩码、错位、梯度都是对的，
    效果不好就是模型容量 / 数据量的问题；
  * 若 loss 卡在 3~5 不动            → 大概率是 label 没错位（模型在预测前一个字符）
    或 prompt 掩码写错（把答案也 mask 掉了）。

用法::

    python code/util/verify_pipeline.py                 # 默认 32 条、300 步
    python code/util/verify_pipeline.py --n 64 --steps 200
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
from util.sft_dataset import IGNORE_INDEX, SFTDataset, make_collate_fn  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--n", type=int, default=32, help="过拟合多少条样本")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    ckpt = torch.load(ROOT / cfg.sft.init_ckpt, map_location="cpu", weights_only=False)
    # 用 checkpoint 里的结构配置，保证和权重一致
    model_cfg = OmegaConf.create(ckpt["cfg"])
    model_cfg.hparams.vocab_path = cfg.hparams.vocab_path
    model_cfg.hparams.sft_train_filename = cfg.hparams.sft_train_filename
    model_cfg.hparams.sft_val_filename = cfg.hparams.sft_val_filename

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer.from_file(model_cfg)
    model = MyModule(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    ds = SFTDataset(model_cfg, tok, "train")
    collate = make_collate_fn(tok)
    x, y = collate([ds[i] for i in range(args.n)])
    x, y = x.to(device), y.to(device)
    print(
        f"过拟合验证 | {args.n} 条样本 | batch 最长 {x.shape[1]} token | "
        f"监督 token {int((y != IGNORE_INDEX).sum())} | device={device}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")

    final = min(history[-20:])
    print(f"\n最低 loss（最后 20 步）: {final:.4f}")
    if final < 0.5:
        print("✓ 管线正常：模型能把少量样本背下来，loss 降不下去就是容量/数据量的问题")
    else:
        print("✗ 管线可能有 bug：检查 label 是否错位、prompt 是否被误 mask")


if __name__ == "__main__":
    main()

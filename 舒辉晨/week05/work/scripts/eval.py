"""评估：在验证集上算 loss / ppl，并生成一批样例写成报告。

运行：
    python scripts/eval.py                          # 评 SFT 的 best 检查点
    python scripts/eval.py --ckpt outputs/pretrain/best.pt --n 5
    python scripts/eval.py --out outputs/eval_report.md

输出两样东西：
    终端：loss / ppl / 参数量 / 样例对比
    文件：outputs/eval_report.md（可以直接贴进课件或作业）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sftlab.data import (  # noqa: E402
    PretrainDataset,
    SFTDataset,
    make_loader,
    read_jsonl,
)
from sftlab.generate import chat, raw_continue  # noqa: E402
from sftlab.train import load_model_for_inference  # noqa: E402
from sftlab.utils import describe_device, get_device, ppl  # noqa: E402


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[float, int]:
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            _, loss = model(ids, labels)
        total += float(loss)
        batches += 1
    return total / max(1, batches), batches


def main() -> None:
    parser = argparse.ArgumentParser(description="评估检查点")
    parser.add_argument("--ckpt", default="outputs/sft/best.pt")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--n", type=int, default=5, help="生成多少条样例")
    parser.add_argument("--out", default="outputs/eval_report.md")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = get_device(args.device)
    model, tok, ckpt = load_model_for_inference(args.ckpt, device)
    stage = ckpt.get("stage", "sft")

    rows = read_jsonl(ROOT / "data" / "raw" / f"sft_{args.split}.jsonl")
    # 两个阶段的数据组织方式不同，评估也必须用各自的那一套，否则 loss 不可比：
    # 预训练看整段文本的全位置 loss，SFT 只看回答部分。
    if stage == "pretrain":
        ds = PretrainDataset(rows, tok, block_size=model.cfg.block_size)
        loader = make_loader(ds, 32, False, tok.pad_id, "pretrain")
    else:
        ds = SFTDataset(rows, tok, max_len=model.cfg.block_size)
        loader = make_loader(ds, 32, False, tok.pad_id, "sft")

    loss, batches = evaluate_loader(model, loader, device)
    print(f"设备     : {describe_device(device)}")
    print(f"检查点   : {args.ckpt} (stage={stage}, step={ckpt.get('step')})")
    print(f"参数量   : {model.num_params()/1e6:.2f}M")
    print(f"{args.split} loss : {loss:.4f}  ppl {ppl(loss):.2f}  ({batches} 个 batch)")
    if stage == "sft":
        print(f"mask 比例: 仅 {ds.stats.supervised_ratio:.1%} 的 token 参与 loss")
    print("-" * 70)

    samples = []
    for rec in rows[: args.n]:
        q = rec["instruction"]
        if stage == "sft":
            pred = chat(model, tok, q, max_new_tokens=100, device=device)
        else:
            pred = "…" + raw_continue(model, tok, q, max_new_tokens=100, device=device)
        samples.append((q, rec["output"], pred))
        print(f"Q: {q[:50]}\n  参考: {rec['output'][:50]}\n  生成: {pred[:50]}\n")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 评估报告：{Path(args.ckpt).name}",
        "",
        f"- 阶段：`{stage}`",
        f"- 参数量：{model.num_params()/1e6:.2f}M",
        f"- 数据集：`sft_{args.split}.jsonl`（{len(ds)} 条）",
        f"- loss：**{loss:.4f}**，ppl：**{ppl(loss):.2f}**",
    ]
    if stage == "sft":
        lines.append(f"- 参与 loss 的 token 比例：{ds.stats.supervised_ratio:.1%}")
    lines += [
        "",
        "## 样例",
        "",
        "| # | 问题 | 参考回答 | 模型生成 |",
        "|---|------|----------|----------|",
    ]
    for i, (q, ref, pred) in enumerate(samples, 1):
        clean = lambda s: s.replace("|", "／").replace("\n", " ")[:60]  # noqa: E731
        lines.append(f"| {i} | {clean(q)} | {clean(ref)} | {clean(pred)} |")
    lines += ["", "> 生成质量的上限由数据量和模型规模决定，本项目的目标是跑通 SFT 的完整链路。", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {args.out}")


if __name__ == "__main__":
    main()

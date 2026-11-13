"""和训练好的模型对话，并且把 Base / SFT 摆在一起对比。

运行：
    python scripts/chat.py                      # 交互模式（回车提问，/exit 退出）
    python scripts/chat.py -p "什么是人工智能？"  # 单次提问
    python scripts/chat.py --only sft           # 只看 SFT
    python scripts/chat.py --compare            # Base 与 SFT 并排（默认，两个检查点都在时）

对比的意义（对应 PPT 第 26 页）：
    Base：给它一个问题，它按「续写」的本能往下写，不知道停，也不会守格式；
    SFT ：同一个问题，它按 <|user|>/<|assistant|> 的格式作答，说完吐 <|end|> 就停。
    参数一模一样，差的是数据格式与 loss mask。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sftlab.generate import chat, raw_continue  # noqa: E402
from sftlab.train import load_model_for_inference  # noqa: E402
from sftlab.utils import describe_device, get_device  # noqa: E402

BASE_CKPT = ROOT / "outputs" / "pretrain" / "best.pt"
# 默认用 last.pt：它训练得最充分，演示「格式学对了」最直观。
# 想更看重泛化就换成 best.pt（验证集 loss 更低，但欠拟合，答得更碎）。
SFT_CKPT = ROOT / "outputs" / "sft" / "last.pt"


def wrap(text: str, width: int = 68) -> str:
    """按宽度硬折行，中英文混排够用了。"""
    text = " ".join(text.split())
    return "\n".join(text[i : i + width] for i in range(0, len(text), width)) or "（空）"


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 模型对话")
    parser.add_argument("-p", "--prompt", default=None, help="单次提问；不填则进入交互模式")
    parser.add_argument("--only", choices=["sft", "base", "both"], default="both")
    parser.add_argument("--base-ckpt", default=str(BASE_CKPT))
    parser.add_argument("--sft-ckpt", default=str(SFT_CKPT))
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"设备: {describe_device(device)}")

    def load(kind: str, path: str, title: str):
        if args.only not in (kind, "both") or not Path(path).exists():
            return None
        model, tok, ckpt = load_model_for_inference(path, device)
        label = f"{title}（val {ckpt.get('val_loss', float('nan')):.3f}）"
        return {"model": model, "tok": tok, "label": label}

    models = {
        "base": load("base", args.base_ckpt, "Base · 预训练续写器"),
        "sft": load("sft", args.sft_ckpt, "SFT · 问答助手"),
    }
    models = {k: v for k, v in models.items() if v}

    if not models:
        raise SystemExit(
            "没有找到任何检查点，请先训练：\n"
            "  python scripts/train.py --stage pretrain\n"
            "  python scripts/train.py --stage sft"
        )
    if args.only == "both" and len(models) == 1:
        print("[提示] 只有一个检查点，先按单个模型演示；跑完 SFT 后就能看到对比。")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    def answer_once(question: str) -> None:
        print("\n" + "─" * 74)
        print(f"问：{question}")
        print("─" * 74)
        if "sft" in models:
            m = models["sft"]
            print(f"\n【{m['label']}】")
            print(wrap(chat(m["model"], m["tok"], question, **gen_kwargs)))
        if "base" in models:
            m = models["base"]
            # Base 模型不认识模板，直接把问题丢进去让它「接着说」
            print(f"\n【{m['label']}】")
            print(wrap("…" + raw_continue(m["model"], m["tok"], question, **gen_kwargs)))
        print()

    if args.prompt:
        answer_once(args.prompt)
        return

    print("\n交互模式：输入问题回车，/exit 退出，/sample 看一条训练集里的问题\n")
    samples: list[str] = []
    if "sft" in models:
        from sftlab.data import read_jsonl

        rows = read_jsonl(ROOT / "data" / "raw" / "sft_val.jsonl")
        samples = [r["instruction"] for r in rows[:20]]
        for s in samples[:2]:
            print(f"  试试：{s}")

    while True:
        try:
            q = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/sample":
            print("  " + samples[0] if samples else "  （无）")
            continue
        answer_once(q)


if __name__ == "__main__":
    main()

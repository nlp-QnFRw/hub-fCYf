"""demo.py —— 用 SFT 之后的模型做「问 → 答」。

用法::

    python code/demo.py                        # 跑内置问题 + 验证集样例
    python code/demo.py --interactive          # 交互式问答（Ctrl+C 退出）
    python code/demo.py --compare              # 预训练 best.pt 与 SFT 模型并排对比
    python code/demo.py --temperature 0.7 --top-k 40 --top-p 0.9
    python code/demo.py --ckpt checkpoints/best.pt   # 指定任意 checkpoint

两种解码方式：
  * 贪心（--temperature 0）：答案稳定、可复现，适合评测；
  * 采样（temperature>0 + top-k/top-p）：答案更多样，适合闲聊体验。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# 让 import 能找到 code/ 和 code/util/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "util"))

# Windows 控制台/管道默认 GBK，"✓ ✗ 👋" 之类可能编码失败
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

from my_module import MyModule  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from util.generate import answer_perplexity, generate_answer  # noqa: E402

#: 内置演示问题（尽量贴近训练数据里的题型，又不等同于任何一条训练样本）
DEMO_QUESTIONS = [
    "给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 128 数字2: 395",
    "计算一组数字的平均值。 数字：10, 20, 30, 40, 50",
    "检查这个电子邮件地址是否符合格式：li.lei@example.com",
    "把下面这句话翻译成英文：今天天气很好。",
    "写一句关于环境保护的宣传语。",
]


def resolve_path(path: str | Path) -> Path:
    """相对路径按 week05/ 解析（这样 --ckpt checkpoints/sft_best.pt 在任何目录下都能跑）。"""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def rel(path: Path) -> str:
    """尽量显示相对于 week05/ 的短路径。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_model(ckpt_path: Path, device):
    """加载 checkpoint → 重建模型 → 载入权重。结构以 checkpoint 内保存的 cfg 为准。"""
    ckpt_path = resolve_path(ckpt_path)
    if not ckpt_path.exists():
        sys.exit(
            f"找不到 checkpoint：{ckpt_path}\n"
            f"请先运行 python code/sft_train.py 训练 SFT 模型"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    tok = CharTokenizer.from_file(cfg)
    model = MyModule(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    print(
        f"加载 {rel(ckpt_path)} | "
        f"epoch {ckpt.get('epoch', '?')} | val_loss {ckpt.get('val_loss', float('nan')):.4f} | "
        f"vocab {len(tok)}"
    )
    return model, tok


def ask(model, tok, question, device, args):
    """回答问题，并给出答案困惑度。"""
    answer = generate_answer(
        model,
        tok,
        question,
        device,
        max_len=args.max_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    ppl = answer_perplexity(model, tok, question, answer, device, max_len=args.max_len) if answer else float("inf")
    return answer, ppl


def run_batch(model, tok, device, args, questions=None):
    print("=" * 72)
    print("【问答】问：{问题}  答：{模型生成}")
    print("=" * 72)
    for q in questions or DEMO_QUESTIONS:
        answer, ppl = ask(model, tok, q, device, args)
        print(f"  问：{q}")
        print(f"  答：{answer}")
        print(f"      （答案 ppl {ppl:.1f}）\n")


def run_val_examples(model, tok, device, args, n=5):
    """拿验证集前 n 条做对比：参考答案 vs 模型生成。"""
    import json

    val_path = ROOT / "data" / "sft_val.jsonl"
    if not val_path.exists():
        return
    print("=" * 72)
    print(f"【验证集样例】前 {n} 条：参考答案 vs 模型生成")
    print("=" * 72)
    with val_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:n]
    for rec in rows:
        answer, _ = ask(model, tok, rec["instruction"], device, args)
        ok = answer.strip() == rec["output"].strip()
        print(f"  {'✓' if ok else '✗'} 问：{rec['instruction'][:70]}")
        print(f"       参考：{rec['output'][:70]}")
        print(f"       生成：{answer[:70]}\n")


def run_compare(base_ckpt: Path, sft_ckpt: Path, device, args):
    """同一批问题，并排看预训练模型和 SFT 模型的回答。"""
    print("=" * 72)
    print("【对比】预训练（best.pt）  vs  SFT（sft_best.pt）")
    print("=" * 72)
    base_model, base_tok = load_model(base_ckpt, device)
    sft_model, sft_tok = load_model(sft_ckpt, device)
    for q in DEMO_QUESTIONS:
        base_ans, _ = ask(base_model, base_tok, q, device, args)
        sft_ans, sft_ppl = ask(sft_model, sft_tok, q, device, args)
        print(f"  问：{q}")
        print(f"  预训练：{base_ans}")
        print(f"  SFT   ：{sft_ans}")
        print(f"          （SFT 答案 ppl {sft_ppl:.1f}）\n")


def run_interactive(model, tok, device, args):
    print("=" * 72)
    print("【交互式问答】输入问题回车，Ctrl+C 退出")
    print("=" * 72)
    try:
        while True:
            question = input("问：").strip()
            if not question:
                continue
            answer, ppl = ask(model, tok, question, device, args)
            print(f"答：{answer}")
            print(f"（答案 ppl {ppl:.1f}）\n")
    except (KeyboardInterrupt, EOFError):
        print("\n再见 👋")


def main():
    parser = argparse.ArgumentParser(description="SFT 模型问答 demo")
    parser.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "sft_best.pt"))
    parser.add_argument("--base-ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--cfg", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--compare", action="store_true", help="并排对比预训练模型与 SFT 模型")
    parser.add_argument("--temperature", type=float, default=0.7, help="0 = 贪心解码")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 位置编码长度来自 checkpoint 的配置，这里补上方便下游截断
    ckpt_path = resolve_path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"找不到 checkpoint：{ckpt_path}\n请先运行 python code/sft_train.py")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args.max_len = int(OmegaConf.create(ckpt["cfg"]).hparams.max_len)

    if args.compare:
        run_compare(resolve_path(args.base_ckpt), ckpt_path, device, args)
        return

    model, tok = load_model(ckpt_path, device)
    if args.interactive:
        run_interactive(model, tok, device, args)
    else:
        run_batch(model, tok, device, args)
        run_val_examples(model, tok, device, args)
        print("交互式问答：python code/demo.py --interactive")


if __name__ == "__main__":
    main()

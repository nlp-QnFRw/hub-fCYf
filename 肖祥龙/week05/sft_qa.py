"""
微调模型的交互式问答验证程序。

用法：
    python sft_qa.py
    python sft_qa.py --model_path best_model_sft.pt --max_new_tokens 64
"""

import argparse
import time
from pathlib import Path

import torch

from sft_common import SCRIPT_DIR, generate, load_sft_checkpoint


def run(model, vocab, device, max_new_tokens):
    """
    连续接收用户问题并显示模型回答，退出时返回 None。

    参数：
        model：已经加载权重的 SFT 模型。
        vocab：检查点中保存的字符及特殊标识词表。
        device：执行推理的设备。
        max_new_tokens：每次回答最多生成的 token 数。
    """
    print("=" * 56)
    print("  SFT 模型手动问答验证")
    print("  输入问题并回车，查看模型回答；q = 退出")
    print("  每个问题独立作答，不携带前面的问题和回答。")
    print(f"  问题最多 {model.max_seq_len - 2} 个字符，回答上限 {max_new_tokens} 个 token。")
    print("=" * 56)

    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            print("退出。")
            break

        try:
            started = time.perf_counter()
            # 使用与训练一致的角色模板，沿用已有生成逻辑和结束标识判断。
            result = generate(model, vocab, question, device, max_new_tokens)
            elapsed = time.perf_counter() - started
        except ValueError as exc:
            print(f"输入有误：{exc}，请重新输入。")
            continue
        except KeyboardInterrupt:
            print("\n本次生成已中断，可以继续输入问题。")
            continue

        answer = result["text"] or "（模型直接结束，未生成回答正文）"
        print(f"回答> {answer}")
        if result["ended"]:
            print(f"（用时 {elapsed:.2f} 秒，回答正常结束）")
        else:
            print(f"（用时 {elapsed:.2f} 秒，已达到回答长度上限，回答可能不完整）")


def main():
    """解析启动参数，加载一次模型后进入交互循环，无返回值。"""
    parser = argparse.ArgumentParser(description="交互式手动验证 SFT 模型，输入 q 退出。")
    parser.add_argument(
        "--model_path", type=Path, default=SCRIPT_DIR / "best_model_sft.pt",
        help="模型路径，默认加载本脚本目录下的 best_model_sft.pt",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64, help="每次回答的生成长度上限")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens 必须大于 0")
    if not args.model_path.is_file():
        parser.error(f"找不到模型文件：{args.model_path}")

    selected = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if selected == "auto" else selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用 CUDA，请使用 --device cpu")

    print("正在加载模型……", flush=True)
    model, checkpoint = load_sft_checkpoint(args.model_path, device)
    vocab = checkpoint["char2idx"]
    print(f"模型：{args.model_path.resolve()}")
    print(f"设备：{device}；词表：{len(vocab)}；上下文：{model.max_seq_len}")
    run(model, vocab, device, args.max_new_tokens)


if __name__ == "__main__":
    main()

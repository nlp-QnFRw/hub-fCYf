"""加载 SFT 模型，根据用户问题生成回答。"""

import argparse
import sys
from pathlib import Path

import torch

from sft_common import SCRIPT_DIR, generate, load_sft_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=SCRIPT_DIR / "best_model_sft.pt")
    parser.add_argument("--prompt", required=True, help="用户问题，不需要手动添加角色标识")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    selected = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if selected == "auto" else selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用 CUDA，请使用 --device cpu")
    model, checkpoint = load_sft_checkpoint(args.model_path, device)
    try:
        result = generate(model, checkpoint["char2idx"], args.prompt, device, args.max_new_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    print(result["text"])
    if not result["ended"]:
        print("提示：达到生成长度上限，尚未生成 <|end|>。", file=sys.stderr)


if __name__ == "__main__":
    main()

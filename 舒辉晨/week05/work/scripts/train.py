"""训练入口：两个阶段共用一个脚本。

    python scripts/train.py --stage pretrain     # 阶段一：续写器（全位置 loss）
    python scripts/train.py --stage sft          # 阶段二：问答助手（只算回答）

    常用参数：
    --max-steps 50     只跑 50 步，用来快速验证流程是否跑得通
    --epochs 3         临时改训练轮数
    --device cpu       强制用 CPU（默认 auto，有显卡就用显卡）
    --init-from xxx.pt 指定从哪个检查点热启动（默认读 configs/sft.yaml 里的 init_from）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sftlab.train import run_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 两阶段训练")
    parser.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    parser.add_argument("--config", default=None, help="默认用 configs/<stage>.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="提前结束，用于冒烟测试")
    parser.add_argument("--init-from", default=None, help="SFT 阶段的预训练检查点")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = args.config or str(ROOT / "configs" / f"{args.stage}.yaml")
    run_train(
        config_path=config,
        stage=args.stage,
        device_pref=args.device,
        epochs=args.epochs,
        max_steps=args.max_steps,
        init_from=args.init_from,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

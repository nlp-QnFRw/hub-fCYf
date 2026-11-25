"""下载 BERT / 中文 RoBERTa 基座到 models/，之后训练全部离线加载。

huggingface.co 在国内直连会超时，脚本固定走 hf-mirror 镜像；
只取推理/微调需要的文件，跳过 tf / flax / onnx，省一半磁盘。

用法：
    python scripts/download_models.py            # 两个都下
    python scripts/download_models.py --only bert
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODELS = {
    "bert": "google-bert/bert-base-chinese",
    "roberta": "hfl/chinese-roberta-wwm-ext",
}
ALLOW = ["config.json", "vocab.txt", "tokenizer*", "special_tokens_map.json",
         "model.safetensors", "pytorch_model.bin", "added_tokens.json"]
ROOT_MODELS = ROOT / "models"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "bert", "roberta"])
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    keys = list(MODELS) if args.only == "all" else [args.only]
    for key in keys:
        repo = MODELS[key]
        dst = ROOT_MODELS / key
        if (dst / "config.json").exists() and any(dst.glob("*model*")):
            print(f"[skip] {dst} 已存在")
            continue
        t0 = time.time()
        print(f"[download] {repo} -> {dst}")
        snapshot_download(repo_id=repo, local_dir=str(dst), allow_patterns=ALLOW,
                          endpoint="https://hf-mirror.com", max_workers=8)
        size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e6
        print(f"  ok  {size:.0f} MB  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

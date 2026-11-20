#region 0. 模块说明
"""
下载额外的中文预训练主干到 `pretrain_models/`，供 BERT 与 LLM 两条线使用。

为什么走镜像：
  本机 huggingface.co 直连返回 502，`hf-mirror.com` 正常。脚本统一设
  `HF_ENDPOINT=https://hf-mirror.com`，**不改全局配置**（避免影响其它项目）。

为什么要有这一步而不是直接 from_pretrained("hfl/xxx")：
  训练中途联网取模型 = 训练不可复现（源站改文件、断网、限速都会让 run 失败）。
  先把权重落到本地目录，之后所有 run 都是离线的。common/paths.py 的 model_path()
  会自动优先使用本地目录。

使用方式：
  python scripts/04_fetch_backbones.py                 # 下全部
  python scripts/04_fetch_backbones.py --only roberta-wwm
  python scripts/04_fetch_backbones.py --check         # 只检查本地有没有
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import HUB_FALLBACK, LOCAL_MODELS, ROOT, has_local  # noqa: E402

#endregion

#region 2. 全局配置
# 与 common/paths.py 的查找逻辑保持一致：pretrain_models 在项目根的上一级（Week6/）或更上层
def pretrain_root() -> Path:
    """找到实际存在的 pretrain_models 目录（不存在则建在项目根上一级）。"""
    for p in [ROOT, *ROOT.parents[:5]]:
        cand = p / "pretrain_models"
        if cand.is_dir():
            return cand
    cand = ROOT.parent / "pretrain_models"
    cand.mkdir(parents=True, exist_ok=True)
    return cand


# 允许下载的模型（别名 → Hub id）。bert-base 已经有本地权重，不在列表里。
TO_FETCH: dict[str, str] = {
    "roberta-wwm": HUB_FALLBACK["roberta-wwm"],
    "macbert": HUB_FALLBACK["macbert"],
    "qwen3-0.6b": HUB_FALLBACK["qwen3-0.6b"],
}
#endregion

#region 3. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="下载额外中文主干到本地 pretrain_models/")
    ap.add_argument("--only", default=None, help="只下某个别名，如 roberta-wwm")
    ap.add_argument("--check", action="store_true", help="只检查本地是否已就绪")
    ap.add_argument("--endpoint", default="https://hf-mirror.com", help="镜像端点")
    args = ap.parse_args()

    if args.check:
        print("本地模型就绪情况：")
        for alias in LOCAL_MODELS:
            print(f"  {alias:<12} {'✓' if has_local(alias) else '✗ 缺失'}")
        return

    targets = {args.only: TO_FETCH[args.only]} if args.only else TO_FETCH
    root = pretrain_root()
    print(f"下载目录：{root}")
    print(f"镜像端点：{args.endpoint}\n")

    os.environ["HF_ENDPOINT"] = args.endpoint  # 必须在 import huggingface_hub 之前设
    from huggingface_hub import snapshot_download

    for alias, repo_id in targets.items():
        local_name = LOCAL_MODELS[alias]
        dst = root / local_name
        if has_local(alias):
            print(f"[跳过] {alias} 已在本地：{dst}")
            continue
        print(f"[下载] {alias} ← {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dst),
                # 必须同时接受 .safetensors 与 .bin：hfl 的两个中文主干只提供 pytorch_model.bin，
                # 只写 *.safetensors 会"下载成功但没有权重"（has_local 校验兜住了这个坑）
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin",
                                "*.model", "*.merges", "*.vocab"],
                max_workers=2,  # 并发高了会被 hf-mirror 限流（HTTP 429）
            )
        except Exception as e:  # 网络问题不该中断整批
            print(f"[失败] {alias}: {type(e).__name__}: {e}")
            print("       可重跑本脚本续传；或只跑已有的主干（方案允许降级）")
            continue
        ok = has_local(alias)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1024**3
        print(f"[完成] {alias} → {dst}  {size:.2f} GB  {'✓' if ok else '✗ 权重校验未通过'}")

    print("\n" + "=" * 60)
    for alias in LOCAL_MODELS:
        print(f"  {alias:<12} {'✓ 本地' if has_local(alias) else '✗ 缺失（使用时会回退 Hub id）'}")
    print("\n下一步：python scripts/10_run_queue.py --list   # 看看可跑哪些 run")


if __name__ == "__main__":
    main()
#endregion

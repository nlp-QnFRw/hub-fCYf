"""
本地预训练模型定位

为什么还要这一层：模型可能放在 `work/pretrain_models/`，也可能放在工作区上层的
`pretrain_models/`（多项目共用一份权重，省磁盘）。从本文件向上逐级查找，
并且**必须校验目录里真有权重文件**，否则会拿到半下载的空目录。
"""

from pathlib import Path


def _has_weights(p: Path) -> bool:
    if not p.is_dir():
        return False
    return bool(list(p.glob("*.safetensors"))) or (p / "pytorch_model.bin").exists()


def resolve(name: str) -> Path:
    here = Path(__file__).resolve()
    candidates = [p / "pretrain_models" / name for p in here.parents]
    for c in candidates:
        if _has_weights(c):
            return c
    raise FileNotFoundError(
        "未找到本地模型 %s，尝试过：\n  %s" % (name, "\n  ".join(map(str, candidates)))
    )


def macbert_path() -> Path:
    return resolve("chinese-macbert-base")


def rbt3_path() -> Path:
    return resolve("rbt3")


def qwen_path() -> Path:
    return resolve("Qwen3-0.6B")


if __name__ == "__main__":
    for fn in (macbert_path, rbt3_path, qwen_path):
        try:
            print(f"{fn.__name__:<14} -> {fn()}")
        except FileNotFoundError as e:
            print(f"{fn.__name__:<14} -> 缺失")

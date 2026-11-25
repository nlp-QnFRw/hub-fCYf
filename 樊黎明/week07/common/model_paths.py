"""
预训练模型路径解析（同时兼容「母版」与「交付副本」两套目录树）

为什么需要这个模块：
  同一个项目会存在于两套目录树里，且层级深度不同：
    母版：    项目材料准备/event_extraction/<方法目录>/xxx.py
    交付副本：八斗课程/week7 信息抽取/event_extraction/<方法目录>/xxx.py
  脚本所在目录到 pretrain_models/ 的层级深度不同（一个差 1 级、一个差 2 级），
  所以任何写死的 "../pretrain_models" 或 "../../pretrain_models"
  都必然在其中一套里失效。
  正确做法：从本文件所在目录开始**向上逐级查找**模型目录。

  模型目录有两种摆放方式，都要能命中：
    直接放在项目根：week07/roberta/                （week06/07 交付副本就是这种）
    放在统一容器里：.../pretrain_models/roberta/   （母版目录树）

  目录里必须有真实权重文件（*.safetensors / pytorch_model.bin）才算命中，
  避免拿到半下载的空目录——这是这类「本地模型 + 离线课堂」项目最常见的坑。

使用方式：
  from model_paths import resolve_model_path, BERT_MODEL_DIR
  BERT_PATH = resolve_model_path(BERT_MODEL_DIR)   # → .../week07/roberta
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

from pathlib import Path

# ── 模型目录名（必须是磁盘上真实存在的文件夹名）────────────────────────────────
# 注意：nlp-base-trainer 是 ModelScope 的 trainer 名字，不是目录名，写它必然找不到。
BERT_MODEL_DIR = "roberta"   # week07/roberta：本地 24 层 / hidden 1024 中文预训练基座

PRETRAIN_DIR_NAME = "pretrain_models"   # 母版目录树里统一存放预训练模型的容器名


def _is_model_dir(path: Path) -> bool:
    """目录里必须有真实权重文件才算模型（避免拿到半下载的空目录）"""
    if not path.is_dir():
        return False
    if any(path.glob("*.safetensors")):
        return True
    return (path / "pytorch_model.bin").exists()


def candidate_paths(dir_name: str, extra_roots: list[Path] | None = None) -> list[Path]:
    """列出所有候选路径（供 resolve_model_path 使用，也便于报错时展示）"""
    here = Path(__file__).resolve()
    candidates: list[Path] = []

    # 1. 调用方显式指定的根目录（可给模型目录本身，也可给它的父目录）
    for root in (extra_roots or []):
        root = Path(root)
        candidates.append(root if root.name == dir_name else root / dir_name)

    # 2. 从本文件所在目录向上逐级查找 <dir_name>
    #    副本：week07/roberta/
    #    母版：.../pretrain_models/roberta/
    for parent in here.parents:
        candidates.append(parent / dir_name)
        candidates.append(parent / PRETRAIN_DIR_NAME / dir_name)

    return candidates


def resolve_model_path(dir_name: str = BERT_MODEL_DIR,
                       extra_roots: list[Path] | None = None) -> Path:
    """
    定位本地预训练模型目录，返回绝对路径。

    优先级：
      1. extra_roots 里显式给出的目录
      2. 逐级向上找到的第一个含真实权重的 <dir_name> 或 pretrain_models/<dir_name>

    找不到时抛 FileNotFoundError，并打印全部已尝试路径（不静默回退到联网下载，
    保证课堂离线可用）。
    """
    candidates = candidate_paths(dir_name, extra_roots)
    for path in candidates:
        if _is_model_dir(path):
            return path

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"未找到本地模型目录：{dir_name}\n"
        f"已尝试的路径：\n  {tried}\n\n"
        f"请确认 {dir_name}/（或 pretrain_models/{dir_name}/）下存在模型权重；"
        f"本课程约定直接复用已有权重，不重新下载。"
    )


if __name__ == "__main__":
    # 自检：直接打印解析结果，便于课堂排查「模型找不到」问题
    print(f"{BERT_MODEL_DIR:<18} -> {resolve_model_path(BERT_MODEL_DIR)}")

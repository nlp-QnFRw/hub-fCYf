#region 0. 模块说明
"""
全项目**唯一的路径真源**。禁止在其它脚本里写 `Path(__file__).parent.parent / ...`
或任何绝对路径（规范 §1.6 / §2.11）。

设计重点：
  1. 本地预训练模型目录 `pretrain_models/` 不写死层级，而是**从项目根向上逐级查找**
     并校验目录里真的有权重文件（*.safetensors / pytorch_model.bin）；
  2. 找不到时抛 FileNotFoundError 并列出所有尝试过的路径，**不静默回退联网**
     （静默联网会让"离线能跑"这个前提在某天突然失效，且下载卡住时极难排查）；
  3. 所有路径都是 ROOT 派生的，代码可以整体移动到别的目录树仍然可用。

使用方式：
  from common.paths import ROOT, DATA_PROCESSED, SPLITS, RUNS, bert_local, model_dir
  python -m common.paths     # 自检：打印各模型实际解析到的路径
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import os
from pathlib import Path

# 必须在 import torch 之前设置，避免 Windows 上多份 OpenMP 运行库冲突导致进程直接退出
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 显存碎片治理：8GB 卡上不加这句，反复 alloc/free 后会出现"明明显存够却 OOM"。
# 注意：expandable_segments 只有 Linux 支持，Windows 上会打一条 UserWarning，所以按平台设置。
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
#endregion

#region 2. 项目根与常用目录
ROOT: Path = Path(__file__).resolve().parent.parent  # 本文件在 common/ 下，上跳一级即项目根
DATA: Path = ROOT / "data"
DATA_RAW: Path = DATA / "raw"
DATA_PROCESSED: Path = DATA / "processed"
SPLITS: Path = DATA / "splits"
RUNS: Path = ROOT / "runs"
FIGURES: Path = ROOT / "figures"
DOCS: Path = ROOT / "docs"
CACHE: Path = ROOT / "cache"  # API 响应缓存、tokenize 缓存

ALL_JSONL: Path = DATA_PROCESSED / "all.jsonl"
RUNS_INDEX: Path = RUNS / "runs.json"

# 数据源（官方直链；官网首页已跳 THULAC，但 source/ 下的文件仍可下载）
THUCNEWS_URL: str = "http://thuctc.thunlp.org/source/THUCNews.zip"
THUCNEWS_ZIP_BYTES: int = 1_560_309_796  # 用 Range 请求核实过，用于下载完整性校验

for _d in (DATA_RAW, DATA_PROCESSED, SPLITS, RUNS, FIGURES, DOCS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
#endregion

#region 3. 本地模型：向上查找 + 权重校验
_WEIGHT_SUFFIX = (".safetensors", ".bin")


def _has_weights(d: Path) -> bool:
    """判断目录里是否真有权重文件（只有 config.json 的空壳目录不算）。"""
    try:
        return any(f.suffix in _WEIGHT_SUFFIX and f.stat().st_size > 1024 * 1024 for f in d.iterdir())
    except (FileNotFoundError, PermissionError):
        return False


def find_model_dir(dir_name: str, extra_roots: list[Path] | None = None) -> Path:
    """从项目根向上逐级查找 `<某级目录>/pretrain_models/<dir_name>`，返回校验通过的路径。

    参数：
      dir_name   : 模型子目录名，如 "bert-base-chinese"
      extra_roots: 额外候选根目录（默认无）
    返回：
      Path，已确认目录内存在 >1MB 的权重文件
    异常：
      FileNotFoundError，消息里列出全部尝试过的路径
    """
    tried: list[Path] = []
    roots = [ROOT, *ROOT.parents[:5], *(extra_roots or [])]
    for root in roots:
        cand = root / "pretrain_models" / dir_name
        tried.append(cand)
        if cand.is_dir() and _has_weights(cand):
            return cand
    msg = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(f"找不到可用的本地模型 {dir_name}，已尝试：\n{msg}")


# 短名别名 → 本地目录名。新增主干只需在这里加一行，其它代码不用改。
LOCAL_MODELS: dict[str, str] = {
    "bert-base": "bert-base-chinese",
    "roberta-wwm": "chinese-roberta-wwm-ext",
    "macbert": "chinese-macbert-base",
    "qwen2-0.5b": "Qwen2-0.5B-Instruct",
    "qwen3-0.6b": "Qwen3-0.6B",
}


def model_path(alias: str) -> str:
    """短名 → 本地路径字符串；找不到本地权重时**返回 Hub id 由调用方决定是否联网**。

    返回 str 而不是 Path，是因为 transformers 的 from_pretrained 两种都收，
    但 str 在日志里更好看、也不会被误当成本地目录去 mkdir。

    已知别名见 LOCAL_MODELS；不在表里的按"原样返回"处理（可传 Hub id）。
    """
    local_name = LOCAL_MODELS.get(alias)
    if local_name is None:
        return alias  # 允许直接传 hfl/chinese-roberta-wwm-ext 这类 Hub id
    try:
        return str(find_model_dir(local_name))
    except FileNotFoundError:
        # 本地没有 → 返回 Hub id，由 scripts/04_fetch_backbones.py 负责先下好
        return HUB_FALLBACK.get(alias, local_name)


# 本地缺失时回退的 Hub id（必须可通过 HF_ENDPOINT=https://hf-mirror.com 下载）
HUB_FALLBACK: dict[str, str] = {
    "roberta-wwm": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


def has_local(alias: str) -> bool:
    """判断某个别名是否已经能离线加载（用于批量跑 run 前的可用性检查）。"""
    local_name = LOCAL_MODELS.get(alias)
    if local_name is None:
        return False
    try:
        find_model_dir(local_name)
        return True
    except FileNotFoundError:
        return False
#endregion

#region 4. 数据集相关的固定路径
def split_file(name: str) -> Path:
    """按切分名取文件，如 "train_100k" → data/splits/train_100k.jsonl。"""
    return SPLITS / f"{name}.jsonl"


def run_dir(run_id: str) -> Path:
    """按 run_id 取产物目录，不存在时创建（下游脚本直接写即可）。"""
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
#endregion

#region 5. 自检
if __name__ == "__main__":
    print(f"ROOT = {ROOT}")
    print(f"数据源 = {THUCNEWS_URL}  ({THUCNEWS_ZIP_BYTES / 1024**3:.2f} GB)")
    for alias in LOCAL_MODELS:
        state = "本地 ✓" if has_local(alias) else "本地缺失"
        print(f"  {alias:<12} {state:<8} → {model_path(alias)}")
#endregion

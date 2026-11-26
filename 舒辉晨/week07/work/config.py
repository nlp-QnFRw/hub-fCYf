"""work 项目的全局路径与模型配置。

设计原则：只复用，不复制。week7/code 下的 common/ 是全项目唯一评测与 BIO 编解码实现，
本实验只在 work/ 里新增「数据变体 + 新方法 + 对比分析」，不另起一套指标口径。
"""
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORK_ROOT.parent
WEEK7_CODE = SRC_ROOT / "code"
WEEK6_WORK = SRC_ROOT.parent / "week6-deeplearning" / "work"

WEEK7_DATA = WEEK7_CODE / "data" / "processed"
DATA_ROOT = WORK_ROOT / "data"
OUT = WORK_ROOT / "outputs"
METRIC_DIR = OUT / "metrics"
PRED_DIR = OUT / "preds"
LOG_DIR = OUT / "logs"
CKPT_DIR = OUT / "ckpt"

PRETRAIN_ROOTS = [WEEK6_WORK / "pretrain_models", SRC_ROOT / "pretrain_models"]
MODEL_DIRS = {
    "rbt3": "rbt3",
    "macbert": "chinese-macbert-base",
    "qwen": "Qwen3-0.6B",
}

SEED = 42
LLM_N = 60                 # LLM 单路评测条数：与判别式做同批投影比较
LLM_MAX_NEW_TOKENS = 384
LLM_GEN_BATCH = 8

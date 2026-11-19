"""全局实验配置：改这里，不要在各脚本里散落魔数。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REP_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "pretrain_models"

# 数据规模（每类 5000 条，占 4000 训练 / 800 验证，留 200 条余量）
N_TRAIN = 40000
N_VAL = 8000
MAX_LEN = 256
SEED = 42

# 模型
MACBERT = "chinese-macbert-base"
RBT3 = "rbt3"
QWEN = "Qwen3-0.6B"

# 分流阈值（写进接口契约，不是模型参数）
CONF_THRESHOLD = 0.8

"""复用 week7 的公共模块：数据层、BIO 编解码、模型、唯一评测口径、prompt 解析。

work/ 不重写这些逻辑——否则「更准/更差」可能来自评测实现差异，而不是方法差异。
"""
import os
import sys
from pathlib import Path

from config import WEEK7_CODE

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# common 必须排在 llm_zero_shot 前面：两边都有 event_data.py，
# 若 llm 目录在前会遮蔽 common 的 BIO 编解码，报 ImportError（这是真实踩过的坑）。
for _p in (WEEK7_CODE / "llm_zero_shot", WEEK7_CODE / "common"):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
if str(WEEK7_CODE / "llm_zero_shot") not in sys.path:
    sys.path.append(str(WEEK7_CODE / "llm_zero_shot"))
sys.path.insert(0, str(WEEK7_CODE / "common"))

from event_data import (  # noqa: E402
    IGNORE_INDEX, BioDataset, build_joint_schema, collate_bio, decode_bio_spans,
    load_enums, load_jsonl, load_schema, make_pred_row, row_event, save_jsonl,
    span_to_char_tags, spans_to_event,
)
from crf import CRFClassifier  # noqa: E402
from evaluate_events import evaluate  # noqa: E402
from eval_utils import Timer  # noqa: E402
from models import batch_to_device, build_model, predict_char_tags  # noqa: E402
from model_paths import BERT_MODEL_DIR, resolve_model_path  # noqa: E402
from event_prompt import build_messages, parse_model_output  # noqa: E402
from local_generate import (  # noqa: E402
    batch_messages, generate_batch, load_causal_lm, resolve_torch_dtype,
)

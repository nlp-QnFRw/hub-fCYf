"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比

教学重点：
  1. LLM 做 NER 的 prompt 设计
     - zero-shot：只靠任务描述，无样例
     - few-shot：给 3 个标注示例，引导格式对齐
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. LLM 的 span 级别 F1 计算（与 BERT 保持可比性）
  4. 成本控制：只采样 100 条，不跑完整验证集

使用方式：
  python llm_ner.py
  python llm_ner.py --n_samples 50 --model qwen-max

依赖：
  pip install openai
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
LOG_DIR = ROOT / "outputs" / "logs"
load_dotenv(ROOT / ".env")

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地名",
}

ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH)


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(f"请在环境变量或 {ROOT / '.env'} 中设置 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "LLM_BASE_URL",
            "https://compass.llm.shopee.io/compass-api/v1",
        ),
    )


def record_text(record: dict) -> str:
    """将人民日报数据中的逐字 tokens 拼成原文。"""
    tokens = record.get("tokens")
    tags = record.get("ner_tags")
    if not isinstance(tokens, list) or not isinstance(tags, list):
        raise ValueError("数据记录必须包含 tokens 和 ner_tags 列表")
    if len(tokens) != len(tags):
        raise ValueError(
            f"tokens 与 ner_tags 长度不一致：{len(tokens)} != {len(tags)}"
        )
    if not all(isinstance(token, str) for token in tokens):
        raise ValueError("tokens 中的每一项都必须是字符串")
    return "".join(tokens)


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    """将 BIO 标注还原为 {(实体文本, 类型, 起始字符, 结束字符)}，位置从 0 开始且两端包含。"""
    tokens = record.get("tokens")
    tags = record.get("ner_tags")
    record_text(record)  # 校验字段类型及 tokens/ner_tags 长度

    spans = set()
    entity_type = None
    entity_tokens = []
    entity_start = None
    char_offset = 0

    def flush_entity(end_offset: int):
        nonlocal entity_type, entity_tokens, entity_start
        if entity_type is not None:
            surface = "".join(entity_tokens)
            if surface:
                spans.add((surface, entity_type, entity_start, end_offset - 1))
        entity_type = None
        entity_tokens = []
        entity_start = None

    for token, tag in zip(tokens, tags):
        if not isinstance(tag, str):
            raise ValueError("ner_tags 中的每一项都必须是字符串")

        if tag == "O":
            flush_entity(char_offset)
        else:
            prefix, separator, next_type = tag.partition("-")
            if not separator or prefix not in {"B", "I"} or not next_type:
                raise ValueError(f"无效的 BIO 标签：{tag}")
            if prefix == "B" or entity_type != next_type:
                # 不规范的 I- 开头按新实体处理，与项目的数据探索逻辑保持一致。
                flush_entity(char_offset)
                entity_type = next_type
                entity_tokens = [token]
                entity_start = char_offset
            else:
                entity_tokens.append(token)

        char_offset += len(token)

    flush_entity(char_offset)
    return spans


def pred_spans_from_response(
    text: str,
    response_text: str,
    entity_types: list[str] | None = None,
) -> set[tuple[str, str, int, int]]:
    """从 LLM 输出中解析 span，格式：{(surface, type, start, end)}。"""
    # 提取 JSON 块（兼容带 markdown 代码块的输出）
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return set()

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()

    allowed_types = set(entity_types or ENTITY_TYPES_EN)
    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in allowed_types:
            continue
        # 在原文中查找位置（取第一次出现）
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """计算 span-level 精确率、召回率、F1。"""
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家，专门处理《人民日报》风格的中文文本。
请识别以下3类实体，并以 JSON 格式输出结果：
- PER：人名
- ORG：组织机构，包括公司、团队、政府机构等
- LOC：地名，包括国家、地区、城市等

输出格式（严格遵守，不要包含其他文字）：
{"entities": [{"text": "实体文本", "type": "PER"}, ...]}
type 只能是 PER、ORG 或 LOC。

如果没有实体，输出：{"entities": []}"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "新华社记者李明从上海发来报道。",
        "output": '{"entities": [{"text": "新华社", "type": "ORG"}, {"text": "李明", "type": "PER"}, {"text": "上海", "type": "LOC"}]}'
    },
    {
        "text": "邓小平在北京会见了美国总统卡特。",
        "output": '{"entities": [{"text": "邓小平", "type": "PER"}, {"text": "北京", "type": "LOC"}, {"text": "美国", "type": "LOC"}, {"text": "卡特", "type": "PER"}]}'
    },
    {
        "text": "中国队将在东京参加比赛。",
        "output": '{"entities": [{"text": "中国队", "type": "ORG"}, {"text": "东京", "type": "LOC"}]}'
    },
]


def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages


def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    """调用 LLM API，返回文本输出，带简单重试。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{e}")
                return ""
    return ""


def sample_records(n: int, seed: int = 42) -> list[dict]:
    """从验证集中采样 n 条，尽量覆盖所有实体类型。"""
    if n <= 0:
        return []

    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    rng = random.Random(seed)
    # 按实体类型分层采样
    by_type = defaultdict(list)
    for r in records:
        record_types = {
            tag[2:]
            for tag in (r.get("ner_tags") or [])
            if isinstance(tag, str) and tag.startswith("B-")
        }
        for etype in ENTITY_TYPES_EN:
            if etype in record_types:
                by_type[etype].append(r)

    selected_ids = set()
    selected_list = []

    # 每类先取 n // 类别数条，尽量覆盖 PER、ORG、LOC。
    per_type = max(1, n // len(ENTITY_TYPES_EN))
    for etype in ENTITY_TYPES_EN:
        candidates = [r for r in by_type[etype] if id(r) not in selected_ids]
        chosen = rng.sample(candidates, min(per_type, len(candidates)))
        for r in chosen:
            if len(selected_list) < n and id(r) not in selected_ids:
                selected_ids.add(id(r))
                selected_list.append(r)

    # 补足到 n 条
    remaining = [r for r in records if id(r) not in selected_ids]
    rng.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def main():
    args = parse_args()

    client = build_client()
    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        text = record_text(record)
        gold = gold_spans_from_record(record)

        # Zero-shot
        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        zs_pred = pred_spans_from_response(text, zs_resp)

        # Few-shot
        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        fs_pred = pred_spans_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": "peoples_daily",
        "split": "validation",
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    # 确保数值可 JSON 序列化
    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")
    print("\n下一步：python compare_results.py")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL_ID", "gpt-5.6-sol"))
    return parser.parse_args()


if __name__ == "__main__":
    main()

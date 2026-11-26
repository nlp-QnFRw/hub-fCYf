"""
用 DeepSeek API 替代训练模型做 NER 抽取。

流程：
  1) 从训练集按实体类型采样标注示例，让 LLM 总结每类实体的识别规律
  2) 把规律 + 验证集句子交给 LLM 抽取实体
  3) 和 dev.txt 的真实标注对比，输出 P/R/F1（含每类实体）

直接运行：python llm_ner.py
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# ================= 配置（改这里）=================
API_KEY = "sk-05e262bb90c64bb8ac825d7db1c4625e"      # ← 改成你的 DeepSeek API Key
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"           # deepseek-chat(V3) / deepseek-reasoner(R1)
# ==============================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

VAL_NUM = 200          # 验证句子数
EXAMPLE_NUM = 20       # 归纳示例数
BATCH_SIZE = 10        # 每批验证句数（别太大，单次请求过长容易被断连）
SEED = 42

ENTITY_TYPES = ["PET", "DRUG", "DIS", "CRO", "PER", "CUL", "PART", "ORG",
                "PAOG", "OTH", "COM", "BIS", "FER"]
ENTITY_HINTS = {
    "PET": "害虫/虫害", "DRUG": "农药/药剂", "DIS": "病害", "CRO": "作物",
    "PER": "生育期/发育阶段/虫态", "CUL": "品种", "PART": "植物部位",
    "ORG": "机构/组织", "PAOG": "病原菌", "OTH": "其他", "COM": "公司",
    "BIS": "生物菌株", "FER": "肥料",
}


# ---------------------------------------------------------------------------
# 读数据 & 句子转实体
# ---------------------------------------------------------------------------
def load_bio(path: Path):
    sents, chars, labels = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if chars:
                    sents.append((chars, labels)); chars, labels = [], []
            else:
                p = line.split()
                if len(p) >= 2:
                    chars.append(p[0]); labels.append(p[1])
    if chars:
        sents.append((chars, labels))
    return sents


def sentence_to_entities(chars, labels):
    text = "".join(chars)
    ents, cur_type, cur_start = [], None, None
    for i, lab in enumerate(labels):
        if lab.startswith("B-"):
            if cur_type:
                ents.append((cur_type, text[cur_start:i]))
            cur_type, cur_start = lab[2:], i
        elif lab.startswith("I-"):
            if cur_type != lab[2:]:
                if cur_type:
                    ents.append((cur_type, text[cur_start:i]))
                cur_type, cur_start = None, None
        else:
            if cur_type:
                ents.append((cur_type, text[cur_start:i]))
                cur_type, cur_start = None, None
    if cur_type:
        ents.append((cur_type, text[cur_start:]))
    return text, ents


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def sample_examples(train_sents, n_per_type=2, max_total=20):
    by_type = defaultdict(list)
    for chars, labels in train_sents:
        text, ents = sentence_to_entities(chars, labels)
        for typ, _ in ents:
            by_type[typ].append((text, ents))
    examples = []
    for typ in ENTITY_TYPES:
        for text, ents in by_type[typ][:n_per_type]:
            examples.append((text, ents))
    if len(examples) > max_total:
        examples = random.sample(examples, max_total)
    return examples


def sample_val_sentences(dev_sents, n):
    return dev_sents if n >= len(dev_sents) else random.sample(dev_sents, n)


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
def build_rule_prompt(examples):
    type_desc = "、".join(f"{t}({ENTITY_HINTS.get(t, t)})" for t in ENTITY_TYPES)
    ex_lines = []
    for text, ents in examples:
        by = defaultdict(list)
        for t, e in ents:
            by[t].append(e)
        ex_lines.append(f"- 文本: {text}\n  实体: {json.dumps(by, ensure_ascii=False)}")
    return f"""你是一位中文农业领域命名实体识别（NER）专家。实体类型共 {len(ENTITY_TYPES)} 种：{type_desc}。

下面是从训练集采样的标注示例，请据此总结每种实体类型的识别规律（关键词、上下文、格式特征）。

## 标注示例
{chr(10).join(ex_lines)}

## 任务
输出一份规则手册（Markdown 列表，每类 1~3 行），直接输出规则，不要复述示例。"""


def build_extract_prompt(rules, texts):
    items = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return f"""下面是一份中文农业领域 NER 的实体识别规则手册：

{rules}

## 任务
根据规则手册，从下面 {len(texts)} 句话中抽取实体。实体类型为：{", ".join(ENTITY_TYPES)}。

## 待抽取句子
{items}

## 输出格式（严格只输出 JSON，不要任何解释）
[
  {{"id": 1, "entities": [{{"type": "CRO", "text": "水稻"}}]}},
  {{"id": 2, "entities": []}}
]"""


# ---------------------------------------------------------------------------
# 调 DeepSeek API
# ---------------------------------------------------------------------------
def call_llm(prompt: str, retries: int = 3, delay: int = 2) -> str:
    import time
    import requests

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{BASE_URL.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.0},
                timeout=600,
            )
            if resp.status_code != 200:
                print(f"[API错误] HTTP {resp.status_code}: {resp.text}")
                resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"  调用失败(第 {attempt}/{retries} 次): {e}")
            if attempt < retries:
                time.sleep(delay * attempt)      # 2s, 4s 退避重试
    raise last_err


def parse_entities_json(content, n):
    m = re.search(r"\[.*\]", content, re.S)
    data = json.loads(m.group(0)) if m else []
    preds = [set() for _ in range(n)]
    for item in data:
        i = int(item.get("id", 0)) - 1
        if 0 <= i < n:
            for e in item.get("entities", []):
                t, txt = e.get("type", ""), e.get("text", "")
                if t in ENTITY_TYPES and txt:
                    preds[i].add((t, txt))
    return preds


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate(preds_per_sent, gold_per_sent):
    tp = fp = fn = 0
    per_type = defaultdict(lambda: [0, 0, 0])   # [tp, fp, fn]
    for pred, gold in zip(preds_per_sent, gold_per_sent):
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
        for t, e in pred & gold:
            per_type[t][0] += 1
        for t, e in pred - gold:
            per_type[t][1] += 1
        for t, e in gold - pred:
            per_type[t][2] += 1

    def prf(a, b, c):
        p = a / (a + b) if (a + b) else 0.0
        r = a / (a + c) if (a + c) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    p, r, f1 = prf(tp, fp, fn)
    per = {t: prf(*v) for t, v in per_type.items()}
    return {"P": p, "R": r, "F1": f1, "per_type": per, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    if API_KEY.startswith("sk-在这里"):
        print("[错误] 请先在脚本顶部 API_KEY 处填入你的 DeepSeek API Key。")
        return

    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取数据 ...")
    train_sents = load_bio(DATA_DIR / "train.txt")
    dev_sents = load_bio(DATA_DIR / "dev.txt")
    print(f"train {len(train_sents)} 句，dev {len(dev_sents)} 句")

    # ---- 阶段1：总结规律 ----
    examples = sample_examples(train_sents, max_total=EXAMPLE_NUM)
    rule_prompt = build_rule_prompt(examples)
    print(f"\n[阶段1] 调用 {MODEL} 总结训练集规律 ...")
    rules = call_llm(rule_prompt)
    (OUTPUT_DIR / "rules.md").write_text(rules, encoding="utf-8")
    print(f"  规律已保存: {OUTPUT_DIR / 'rules.md'}")

    # ---- 阶段2：验证集抽取 ----
    val_sents = sample_val_sentences(dev_sents, VAL_NUM)
    golds = [set(sentence_to_entities(c, l)[1]) for c, l in val_sents]
    texts = ["".join(c) for c, _ in val_sents]

    all_preds = []
    print(f"\n[阶段2] 在 {len(val_sents)} 条验证句上抽取（每批 {BATCH_SIZE}）...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        prompt = build_extract_prompt(rules, batch_texts)
        try:
            resp = call_llm(prompt)
        except Exception as e:
            print(f"  批次 {i // BATCH_SIZE + 1} 调用失败: {e}，跳过")
            all_preds.extend([set() for _ in batch_texts])
            continue
        all_preds.extend(parse_entities_json(resp, len(batch_texts)))
        print(f"  已处理 {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    # ---- 评估 ----
    metrics = evaluate(all_preds, golds)
    print("\n" + "=" * 60)
    print(f"验证集结果（{len(val_sents)} 条，实体级 (类型,文本) 匹配）")
    print(f"  Precision = {metrics['P']:.4f}")
    print(f"  Recall    = {metrics['R']:.4f}")
    print(f"  F1        = {metrics['F1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print("=" * 60)

    # 每类 F1
    print("\n每类实体 F1（有 gold 的类型）:")
    print(f"{'类型':<8}{'P':>8}{'R':>8}{'F1':>8}")
    for t in ENTITY_TYPES:
        if t in metrics["per_type"]:
            p, r, f = metrics["per_type"][t]
            print(f"{t:<8}{p:>8.3f}{r:>8.3f}{f:>8.3f}")

    (OUTPUT_DIR / "llm_ner_metrics.json").write_text(
        json.dumps({"val_num": len(val_sents), "model": MODEL,
                    "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n结果已保存: {OUTPUT_DIR / 'llm_ner_metrics.json'}")


if __name__ == "__main__":
    main()

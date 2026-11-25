"""
人民日报 NER 数据集探索与可视化。

控制台摘要覆盖训练集和验证集；图表只统计训练集。

使用方式：
  python explore_data.py

输出：
  outputs/figures/entity_distribution.png
  outputs/figures/text_length_distribution.png
  outputs/figures/entity_length_distribution.png
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs" / "figures"
LABEL_NAMES_PATH = DATA_DIR / "label_names.json"

ENTITY_TYPE_NAMES = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地名",
}


def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.json"
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"{path} 的顶层结构应为列表")
    return records


def load_label_names() -> list[str]:
    with LABEL_NAMES_PATH.open("r", encoding="utf-8") as f:
        label_names = json.load(f)

    if not isinstance(label_names, list) or not all(isinstance(tag, str) for tag in label_names):
        raise ValueError(f"{LABEL_NAMES_PATH} 应为字符串标签列表")
    return label_names


def get_entity_types(label_names: list[str]) -> list[str]:
    """按 label_names.json 中的顺序提取实体类型。"""
    return list(
        dict.fromkeys(
            tag.split("-", 1)[1]
            for tag in label_names
            if tag.startswith(("B-", "I-"))
        )
    )


def extract_entities(tokens: list[str], tags: list[str]) -> list[tuple[str, str, int]]:
    """将逐 token 的 BIO 标签还原为（实体类型、实体文本、token 数）。"""
    entities = []
    current_type = None
    current_tokens = []

    def flush_current_entity():
        nonlocal current_type, current_tokens
        if current_type is not None:
            entities.append((current_type, "".join(current_tokens), len(current_tokens)))
        current_type = None
        current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            flush_current_entity()
            continue

        prefix, separator, entity_type = tag.partition("-")
        if not separator or prefix not in {"B", "I"} or not entity_type:
            raise ValueError(f"无效的 BIO 标签：{tag}")

        if prefix == "B" or current_type != entity_type:
            # 不规范的 I- 开头按新实体处理，避免数据被静默丢弃。
            flush_current_entity()
            current_type = entity_type
            current_tokens = [token]
        else:
            current_tokens.append(token)

    flush_current_entity()
    return entities


def collect_stats(records: list[dict], label_names: list[str], split_name: str) -> dict:
    allowed_tags = set(label_names)
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type: dict[str, list[str]] = {}

    for row_index, row in enumerate(records):
        tokens = row.get("tokens")
        tags = row.get("ner_tags")

        if not isinstance(tokens, list) or not isinstance(tags, list):
            raise ValueError(f"{split_name}第 {row_index + 1} 条缺少 tokens 或 ner_tags 列表")
        if len(tokens) != len(tags):
            raise ValueError(
                f"{split_name}第 {row_index + 1} 条 tokens 与 ner_tags 长度不一致："
                f"{len(tokens)} != {len(tags)}"
            )

        invalid_tags = set(tags) - allowed_tags
        if invalid_tags:
            raise ValueError(
                f"{split_name}第 {row_index + 1} 条包含未知标签：{sorted(invalid_tags)}"
            )

        text_lengths.append(len(tokens))
        entities = extract_entities(tokens, tags)
        entity_per_sentence.append(len(entities))

        for entity_type, surface, entity_length in entities:
            entity_type_counts[entity_type] += 1
            entity_lengths.append(entity_length)
            entities_by_type.setdefault(entity_type, []).append(surface)

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def percentile_95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * 0.95)]


def print_split_summary(split_name: str, stats: dict, entity_types: list[str]):
    text_lengths = stats["text_lengths"]
    entity_lengths = stats["entity_lengths"]
    entity_per_sentence = stats["entity_per_sentence"]

    print(f"\n【{split_name}】")
    print(f"  样本数：{len(text_lengths)} 条")
    if text_lengths:
        ordered_lengths = sorted(text_lengths)
        print(f"  文本平均长度：{sum(text_lengths) / len(text_lengths):.1f} 个 token")
        print(f"  文本最大长度：{max(text_lengths)} 个 token")
        print(f"  文本长度中位数：{ordered_lengths[len(ordered_lengths) // 2]} 个 token")
        print(f"  P95 文本长度：{percentile_95(text_lengths)} 个 token")
        print(f"  平均实体数/句：{sum(entity_per_sentence) / len(entity_per_sentence):.2f}")
    print(f"  实体总数：{sum(stats['entity_type_counts'].values())}")
    if entity_lengths:
        print(f"  平均实体长度：{sum(entity_lengths) / len(entity_lengths):.1f} 个 token")

    print(f"\n【各类实体频次（{split_name}）】")
    for entity_type in entity_types:
        count = stats["entity_type_counts"][entity_type]
        cn_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
        print(f"  {entity_type:5s} ({cn_name:6s}) : {count:5d} 条")

    print(f"\n【各类实体示例（{split_name}，取前5个）】")
    for entity_type in entity_types:
        examples = list(dict.fromkeys(stats["entities_by_type"].get(entity_type, [])))[:5]
        cn_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
        print(f"  {entity_type:5s} ({cn_name}) : {' | '.join(examples) if examples else '无'}")


def print_summary(stats_train: dict, stats_val: dict, entity_types: list[str]):
    print("=" * 70)
    print("人民日报 NER 数据集统计摘要")
    print("=" * 70)
    print_split_summary("训练集", stats_train, entity_types)
    print_split_summary("验证集", stats_val, entity_types)
    print()


def save_figure(fig, filename: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / filename
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  已保存 → {output_path}")


def plot_entity_distribution(stats_train: dict, entity_types: list[str]):
    counts = stats_train["entity_type_counts"]
    labels = [
        f"{entity_type}\n({ENTITY_TYPE_NAMES.get(entity_type, entity_type)})"
        for entity_type in entity_types
    ]
    values = [counts[entity_type] for entity_type in entity_types]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("人民日报 NER 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    fig.tight_layout()
    save_figure(fig, "entity_distribution.png")


def plot_text_length_distribution(stats_train: dict):
    lengths = stats_train["text_lengths"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    p95 = percentile_95(lengths)
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title("人民日报 NER 文本长度分布（训练集）", fontsize=14)
    ax.set_xlabel("文本 token 数")
    ax.set_ylabel("样本数")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "text_length_distribution.png")
    print(f"  训练集 P95 文本长度={p95}")


def plot_entity_length_distribution(stats_train: dict):
    length_counts = Counter(stats_train["entity_lengths"])
    lengths = sorted(length_counts)[:20]
    counts = [length_counts[length] for length in lengths]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        [str(length) for length in lengths],
        counts,
        color="#55A868",
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_title("人民日报 NER 实体长度分布（训练集，前20种长度）", fontsize=14)
    ax.set_xlabel("实体 token 数")
    ax.set_ylabel("出现次数")
    fig.tight_layout()
    save_figure(fig, "entity_length_distribution.png")

    if stats_train["entity_lengths"]:
        average_length = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
        print(f"  训练集实体平均长度={average_length:.1f} 个 token")


def parse_args():
    parser = argparse.ArgumentParser(description="探索人民日报 NER 训练集和验证集")
    return parser.parse_args()


def main():
    parse_args()

    label_names = load_label_names()
    entity_types = get_entity_types(label_names)
    train_records = load_split("train")
    val_records = load_split("validation")

    stats_train = collect_stats(train_records, label_names, "训练集")
    stats_val = collect_stats(val_records, label_names, "验证集")

    print_summary(stats_train, stats_val, entity_types)

    print("正在生成训练集可视化图表...")
    plot_entity_distribution(stats_train, entity_types)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)

    print("\n探索完成！图表已保存到 outputs/figures/")


if __name__ == "__main__":
    main()

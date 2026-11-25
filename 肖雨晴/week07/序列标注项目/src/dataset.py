"""
人民日报 NER 数据集类：BIO 标签编码 + BERT 子词对齐。

数据格式：
  {
    "tokens": ["北", "京", ...],
    "ner_tags": ["B-LOC", "I-LOC", ...]
  }

处理流程：
  1. 从 label_names.json 构建标签体系
  2. 将 ner_tags 转为标签 id
  3. 对每个原始 token 做 BERT 子词切分
  4. 首个子词保留标签，其余子词、特殊 token 和 PAD 标记为 -100

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
LABEL_NAMES_FILE = "label_names.json"


def build_label_schema(
    data_dir: Optional[Path] = None,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """从 label_names.json 构建标签体系。"""
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    label_path = directory / LABEL_NAMES_FILE
    with label_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{label_path} 应为非空标签列表")
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError(f"{label_path} 中的标签必须是非空字符串")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{label_path} 中存在重复标签")
    if "O" not in labels:
        raise ValueError(f"{label_path} 缺少 O 标签")

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return labels, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 数据集的 PyTorch Dataset。"""

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens = row.get("tokens")
        tags = row.get("ner_tags")

        if not isinstance(tokens, list) or not isinstance(tags, list):
            raise ValueError(f"第 {idx + 1} 条数据缺少 tokens 或 ner_tags 列表")
        if len(tokens) != len(tags):
            raise ValueError(
                f"第 {idx + 1} 条数据的 tokens 与 ner_tags 长度不一致："
                f"{len(tokens)} != {len(tags)}"
            )

        unknown_tags = set(tags) - self.label2id.keys()
        if unknown_tags:
            raise ValueError(f"第 {idx + 1} 条数据包含未知标签：{sorted(unknown_tags)}")

        piece_ids = []
        piece_labels = []
        for token, tag in zip(tokens, tags):
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                token_ids = [self.tokenizer.unk_token_id]

            piece_ids.extend(token_ids)
            piece_labels.append(self.label2id[tag])
            piece_labels.extend([-100] * (len(token_ids) - 1))

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        if cls_token_id is None or sep_token_id is None:
            raise ValueError("tokenizer 未配置 cls_token_id 或 sep_token_id")

        special_token_count = 2
        content_length = self.max_length - special_token_count
        if content_length < 0:
            raise ValueError(
                f"max_length={self.max_length} 小于特殊 token 数量 {special_token_count}"
            )

        piece_ids = piece_ids[:content_length]
        piece_labels = piece_labels[:content_length]

        input_ids = [cls_token_id, *piece_ids, sep_token_id]
        token_type_ids = [0] * len(input_ids)
        aligned_labels = [-100, *piece_labels, -100]

        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if padding_length and pad_token_id is None:
            raise ValueError("tokenizer 未配置 pad_token_id")

        padding_ids = [pad_token_id] * padding_length
        padding_token_types = [self.tokenizer.pad_token_type_id] * padding_length
        padding_attention = [0] * padding_length
        padding_labels = [-100] * padding_length

        if self.tokenizer.padding_side == "left":
            input_ids = padding_ids + input_ids
            token_type_ids = padding_token_types + token_type_ids
            attention_mask = padding_attention + attention_mask
            aligned_labels = padding_labels + aligned_labels
        else:
            input_ids += padding_ids
            token_type_ids += padding_token_types
            attention_mask += padding_attention
            aligned_labels += padding_labels

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list[dict]:
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    path = directory / f"{split}.json"
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"{path} 的顶层结构应为列表")
    return records


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

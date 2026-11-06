"""第四周：本地 BERT（双向编码器预训练模型）的掩码训练基础。

参考老师的 language_model.py、tokenizer.py 与 export_bert_viz.py，
保留“数据、模型、检查入口”的分块结构，不修改原始材料。
实现 MLM（掩码语言建模）的数据、损失检查及小规模、正式训练。

在 py312 中运行：python -B train_bert_mlm.py --check-only
此入口不写模型、数据或报告文件，不执行反向传播或参数更新。
任务 3 使用 --smoke-test，最多 32 条训练、16 条验证和 4 次优化器更新。
任务 4 使用 --train：最多 20000 条训练候选、2000 条验证、完整一轮。
默认前 5% 更新预热、随后线性衰减，每 500 次更新验证并保存新最佳模型。
训练与验证默认显示终端进度条；--no-progress 可关闭进度条。
--train --plan-only 只检查数据和训练计划，不加载模型、不训练、不写文件。
--train --model-dir <已有模型目录> 从本周已保存权重开始新一轮，重新创建优化器与调度器。
--verify-run <结果目录> 独立验收已保存模型，新增恢复报告，不重新训练。
实际运行结果保存在 outputs/ 下的新目录，不覆盖原始模型或历史结果。
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PY312 = Path("D:/Anaconda3/envs/py312")
DEFAULT_MODEL = HERE / "bert-base-chinese"
DEFAULT_CORPUS = HERE / "week4-语言模型" / "循环神经网络语言模型" / "corpus_general.txt"
IGNORE_INDEX = -100
RELOAD_RTOL = 1e-4
RELOAD_ATOL = 1e-5
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")
SMOKE_CONFIG = {
    "train_candidate_cap": 32, "validation_candidate_cap": 16,
    "epochs": 1, "batch_size": 1, "gradient_accumulation": 8,
    "optimizer_updates_cap": 4, "learning_rate": 2e-5,
    "weight_decay": 0.01, "max_grad_norm": 1.0,
    "precision_policy": "bfloat16_if_supported_else_float32",
    "gradient_checkpointing": True,
    "console_log_every": 1,
    "lr_scheduler": "constant", "warmup_ratio": 0.0, "eval_every": 2,
}
FORMAL_CONFIG = {
    **SMOKE_CONFIG, "train_candidate_cap": 20000, "validation_candidate_cap": 2000,
    "optimizer_updates_cap": None, "console_log_every": 50,
    "lr_scheduler": "linear", "warmup_ratio": 0.05, "eval_every": 500,
}


# ─────────────────────────── 数据 ───────────────────────────

def load_and_split_corpus(path, seed=42, val_ratio=0.05):
    """保持首次出现顺序精确去重，再按行打乱并划分，避免重复文本跨集合。"""
    if not 0 < val_ratio < 1:
        raise ValueError("验证集比例必须在 0 与 1 之间。")
    path = Path(path)
    raw = path.read_bytes()
    # 与普通文本读取一致，将文件中的回车换行统一为换行。
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    nonblank = [line.strip() for line in lines if line.strip()]
    unique = list(dict.fromkeys(nonblank))
    if len(unique) < 2:
        raise ValueError("去重后至少需要两条非空文本，才能划分训练和验证集。")
    random.Random(seed).shuffle(unique)
    split = int(len(unique) * (1 - val_ratio))
    train, validation = unique[:split], unique[split:]
    if not train or not validation:
        raise ValueError("当前比例产生了空训练集或空验证集，请调整划分比例。")
    overlap = len(set(train).intersection(validation))
    if overlap:
        raise ValueError("训练与验证文本存在精确重叠。")
    info = {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "characters": len(text), "raw_lines": len(lines),
        "nonblank_lines": len(nonblank), "unique_lines": len(unique),
        "duplicates_removed": len(nonblank) - len(unique),
        "split_seed": seed, "validation_ratio": val_ratio,
        "train_rows": len(train), "validation_rows": len(validation),
        "exact_overlap": overlap,
    }
    return train, validation, info


def load_local_tokenizer(model_dir):
    """只读取本地分词资源，不下载模型，不更改原配置。"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("本地分词器必须提供遮盖标记和补齐标记。")
    return tokenizer


def make_masked_sample(encoded, tokenizer, seed, mask_probability=0.15):
    """输入一条分词结果，返回同位置预测的输入和标签，不做下一字符错位。

    15% 独立选中；选中位置 80% 遮盖、10% 随机替换、10% 保持原样。
    三类位置均以原编号为标签；未选中位置的 -100 表示不计入损失。
    """
    if not 0 < mask_probability <= 1:
        raise ValueError("选中概率必须大于 0 且不超过 1。")
    original = list(encoded["input_ids"])
    attention = list(encoded["attention_mask"])
    special_mask = list(encoded["special_tokens_mask"])
    segments = list(encoded["token_type_ids"])
    if len({len(original), len(attention), len(special_mask), len(segments)}) != 1:
        raise ValueError("输入编号、注意力屏蔽、特殊标记和句段编号长度必须一致。")
    special_ids = set(tokenizer.all_special_ids)
    eligible = [
        i for i, token_id in enumerate(original)
        if attention[i] and not special_mask[i] and token_id not in special_ids
    ]
    if not eligible:
        raise ValueError("当前样本分词后没有可预测的普通词元，不能产生有效损失。")
    rng = random.Random(seed)
    selected = [i for i in eligible if rng.random() < mask_probability]
    if not selected:
        selected = [rng.choice(eligible)]  # 短句兜底，因此实际比例可能高于 15%。
    masked = original.copy()
    labels = [IGNORE_INDEX] * len(original)
    for position in selected:
        labels[position] = original[position]
        draw = rng.random()
        if draw < 0.8:
            masked[position] = tokenizer.mask_token_id
        elif draw < 0.9:
            replacement = rng.randrange(len(tokenizer))
            while replacement in special_ids:
                replacement = rng.randrange(len(tokenizer))
            masked[position] = replacement
        # 剩余 10% 保持原词元，标签仍有效，不等同于未选中位置。
    return {
        "input_ids": torch.tensor(masked, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "token_type_ids": torch.tensor(segments, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class MLMDataset(Dataset):
    """原始编号为 [样本数, 长度]，每次取样构造同样长度的掩码输入与标签。

    验证集只按种子和索引决定遮盖；训练集额外加入轮次，避免每轮固定遮盖。
    使用独立随机数生成器，不依赖数据加载顺序，也不修改原始分词结果。
    """

    def __init__(self, texts, tokenizer, max_length=64, seed=42,
                 fixed_mask=False, mask_probability=0.15):
        if not texts:
            raise ValueError("数据集不能为空。")
        if max_length < 3:
            raise ValueError("输入长度至少为 3，需容纳起始、内容和分隔标记。")
        if not 0 < mask_probability <= 1:
            raise ValueError("选中概率必须大于 0 且不超过 1。")
        self.tokenizer = tokenizer
        self.seed = seed
        self.fixed_mask = fixed_mask
        self.mask_probability = mask_probability
        self.epoch = 0
        self.encoded = tokenizer(
            texts, max_length=max_length, truncation=True, padding="max_length",
            return_special_tokens_mask=True, return_tensors="pt",
        )
        eligible = (self.encoded["attention_mask"].bool()
                    & ~self.encoded["special_tokens_mask"].bool())
        for special_id in tokenizer.all_special_ids:
            eligible &= self.encoded["input_ids"] != special_id
        usable = eligible.any(dim=1)
        self.source_indices = usable.nonzero().flatten().tolist()
        self.excluded_indices = (~usable).nonzero().flatten().tolist()
        if not self.source_indices:
            raise ValueError("所有样本分词后均无可预测的普通词元，请检查语料或分词配置。")

    def __len__(self):
        return len(self.source_indices)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("轮次不能为负数。")
        self.epoch = epoch

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # 排除无目标样本后仍保留原索引，避免其他样本的固定遮盖种子发生漂移。
        source_index = self.source_indices[index]
        source_count = self.encoded["input_ids"].size(0)
        epoch_offset = 0 if self.fixed_mask else self.epoch * source_count
        encoded = {key: value[source_index].tolist() for key, value in self.encoded.items()}
        try:
            return make_masked_sample(
                encoded, self.tokenizer, self.seed + epoch_offset + source_index,
                self.mask_probability,
            )
        except ValueError as exc:
            raise ValueError(f"原始样本索引 {source_index}：{exc}") from exc


# ─────────────────────── 本地模型与损失 ───────────────────────

def load_local_model(model_dir, device):
    """复用老师的本地加载方式，使用带掩码预测头的完整预训练模型。"""
    from transformers import BertForMaskedLM

    model, loading = BertForMaskedLM.from_pretrained(
        str(model_dir), local_files_only=True, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"预训练权重加载不完整：{loading}")
    # 新版工具库可能返回集合，先转成可稳定序列化的列表。
    info = {
        "missing_keys": sorted(loading.get("missing_keys") or []),
        "unexpected_keys": sorted(loading.get("unexpected_keys") or []),
    }
    return model.to(device), info


# ─────────────────────── 数据与前向检查 ───────────────────────

def check_sample(dataset, index):
    """以未遮盖编号独立核对标签与排除位置，返回有效预测位置数量。"""
    sample = dataset[index]
    source_index = dataset.source_indices[index]
    original = dataset.encoded["input_ids"][source_index]
    selected = sample["labels"] != IGNORE_INDEX
    if not selected.any():
        raise AssertionError("样本没有任何预测目标。")
    excluded = (dataset.encoded["special_tokens_mask"][source_index].bool()
                | ~sample["attention_mask"].bool())
    for special_id in dataset.tokenizer.all_special_ids:
        excluded |= original == special_id
    if selected[excluded].any():
        raise AssertionError("特殊标记或补齐位置被错误地计入损失。")
    if not torch.equal(sample["labels"][selected], original[selected]):
        raise AssertionError("预测标签必须保持选中位置的原始编号。")
    if not torch.equal(sample["input_ids"][~selected], original[~selected]):
        raise AssertionError("未选中的词元不应改变。")
    if not all(value.shape == original.shape for value in sample.values()):
        raise AssertionError("输入、屏蔽与标签维度不一致。")
    return int(selected.sum())


def run_checks(args):
    if Path(sys.prefix).resolve() != PY312.resolve():
        raise RuntimeError("本项目必须使用 D:/Anaconda3/envs/py312/python.exe。")
    train_rows, val_rows, corpus_info = load_and_split_corpus(
        args.corpus, args.seed, args.val_ratio,
    )
    tokenizer = load_local_tokenizer(args.model_dir)
    # 先读本地配置校验长度，避免截断到模型不支持的位置数。
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if args.max_length > config["max_position_embeddings"]:
        raise ValueError("输入长度超过本地模型的位置数上限。")
    train_ds = MLMDataset(
        train_rows[:args.train_limit], tokenizer, args.max_length,
        seed=args.seed, mask_probability=args.mask_probability,
    )
    val_ds = MLMDataset(
        val_rows[:args.val_limit], tokenizer, args.max_length,
        seed=args.validation_mask_seed, fixed_mask=True,
        mask_probability=args.mask_probability,
    )
    counts = {
        "train": sum(check_sample(train_ds, i) for i in range(len(train_ds))),
        "validation": sum(check_sample(val_ds, i) for i in range(len(val_ds))),
    }
    # 检查每个验证样本，包含替换内容和标签，而不仅是检查选中位置。
    for i in range(len(val_ds)):
        first, repeated = val_ds[i], val_ds[i]
        if not all(torch.equal(first[key], repeated[key]) for key in first):
            raise AssertionError("验证遮盖重复生成不一致。")
    train_examples = [train_ds[i] for i in range(min(32, len(train_ds)))]
    val_example = val_ds[0]
    train_ds.set_epoch(1)
    val_ds.set_epoch(1)
    changed = sum(
        any(not torch.equal(previous[key], train_ds[i][key]) for key in previous)
        for i, previous in enumerate(train_examples)
    )
    # 不同随机种子仍可能恰好给出相同遮盖，尤其是只有一个可预测词元的短句；
    # 记录观察到的变化数，不将偶然相同误判为错误。
    if not all(torch.equal(val_example[key], val_ds[0][key]) for key in val_example):
        raise AssertionError("验证遮盖不应随训练轮次变化。")
    train_ds.set_epoch(0)
    val_ds.set_epoch(0)
    if not all(
        torch.equal(previous[key], train_ds[i][key])
        for i, previous in enumerate(train_examples) for key in previous
    ):
        raise AssertionError("恢复相同轮次后，训练遮盖应可复现。")
    print("数据与遮盖检查通过，准备执行一次模型前向计算。", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loading = load_local_model(args.model_dir, device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("模型与分词器的词表大小不一致。")
    model.eval()
    batch = next(iter(DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)))
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        # 原配置 return_dict=false，在调用处显式要求带字段名的返回结果。
        output = model(**batch, return_dict=True)
        selected = batch["labels"] != IGNORE_INDEX
        manual_loss = F.cross_entropy(output.logits[selected], batch["labels"][selected])
        expected_shape = (*batch["input_ids"].shape, len(tokenizer))
        if tuple(output.logits.shape) != expected_shape:
            raise AssertionError("模型输出维度不符合 [批次数, 长度, 词表大小]。")
        if not torch.isfinite(output.loss).item():
            raise AssertionError("模型损失不是有限值。")
        torch.testing.assert_close(output.loss, manual_loss, rtol=1e-5, atol=1e-6)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("检查入口不应生成参数梯度。")
    first = val_ds[0]
    valid_length = int(first["attention_mask"].sum())
    report = {
        "status": "passed", "python": sys.executable, "corpus": corpus_info,
        "selected_rows": {"train": len(train_ds), "validation": len(val_ds)},
        "excluded_no_target_rows": {
            "train": len(train_ds.excluded_indices),
            "validation": len(val_ds.excluded_indices),
        },
        "excluded_source_indices": {
            "train": train_ds.excluded_indices, "validation": val_ds.excluded_indices,
        },
        "max_length": args.max_length, "mask_probability": args.mask_probability,
        "validation_mask_seed": args.validation_mask_seed,
        "prediction_positions": counts,
        "validation_masks_repeatable": True, "validation_masks_independent_of_epoch": True,
        "training_mask_changes_in_first_32": changed, "training_masks_repeatable": True,
        "device": str(device), "model_loading": loading,
        "input_shape": list(batch["input_ids"].shape),
        "labels_shape": list(batch["labels"].shape),
        "logits_shape": list(output.logits.shape),
        "model_loss": output.loss.item(), "selected_only_cross_entropy": manual_loss.item(),
        "loss_absolute_difference": abs(output.loss.item() - manual_loss.item()),
        "first_validation_example": {
            "source_index": val_ds.source_indices[0],
            "original": tokenizer.convert_ids_to_tokens(val_ds.encoded["input_ids"][val_ds.source_indices[0], :valid_length].tolist()),
            "masked": tokenizer.convert_ids_to_tokens(first["input_ids"][:valid_length].tolist()),
            "labels": first["labels"][:valid_length].tolist(),
        },
        "training_executed": False,
        "note": "仅检查数据和一次前向计算，不是训练结果或完整验证集模型基线。",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


# ─────────────────────── 训练、验证与保存 ───────────────────────

def write_json(path, value):
    """先完成标准格式转换再写文件，避免无法序列化时留下半份报告。"""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def log_event(run_dir, event, console=True, **values):
    record = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **values}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    if console:
        tqdm.write(line)


def fixed_sample_records(samples, source_indices):
    return [
        {"source_index": source_index,
         **{key: tensor.tolist() for key, tensor in sample.items()}}
        for sample, source_index in zip(samples, source_indices)
    ]


def sample_signature(samples, source_indices):
    text = json.dumps(fixed_sample_records(samples, source_indices), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_fingerprint(tensor):
    """记录张量的形状、类型与原始字节指纹，不以浮点容差比较模型权重。"""
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def model_state_fingerprint(model):
    """覆盖全部持久化参数与缓冲区，包括共享权重在状态字典中的别名。"""
    tensors = {key: tensor_fingerprint(value) for key, value in sorted(model.state_dict().items())}
    canonical = json.dumps(tensors, sort_keys=True, separators=(",", ":"))
    return {"algorithm": "sha256", "scope": "all_state_dict_tensors", "tensors": tensors,
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


@torch.inference_mode()
def reference_logits(model, batch, device):
    """固定评估模式，关闭自动混合精度，返回内存中的单精度输出用于比较。"""
    model.eval()
    inputs = {key: value.to(device) for key, value in batch.items()}
    with torch.autocast(device_type=device.type, enabled=False):
        return model(**inputs, return_dict=True).logits.detach().float().cpu().clone()


def compare_logits(reference, candidate, attention_mask):
    """保留原有容差，只验收非补齐位置；完整输出差异作为诊断信息。"""
    reference, candidate = reference.cpu(), candidate.cpu()
    valid = attention_mask.cpu().bool()
    if (reference.shape != candidate.shape or reference.ndim != 3
            or tuple(valid.shape) != tuple(reference.shape[:2]) or not valid.any()):
        raise ValueError("参考输出、重载输出及非补齐位置的形状不匹配，或没有有效位置。")
    if not torch.isfinite(reference).all().item() or not torch.isfinite(candidate).all().item():
        raise RuntimeError("模型输出包含非有限值。")
    difference = (candidate - reference).abs()
    mismatched = ~torch.isclose(candidate, reference, rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
    valid_mismatches = int(mismatched[valid].sum().item())
    return {
        "passed": valid_mismatches == 0, "scope": "nonpadding_positions",
        "rtol": RELOAD_RTOL, "atol": RELOAD_ATOL,
        "nonpadding_positions": int(valid.sum().item()),
        "nonpadding_mismatched_elements": valid_mismatches,
        "padding_mismatched_elements": int(mismatched[~valid].sum().item()),
        "nonpadding_max_absolute_difference": float(difference[valid].max().item()),
        "nonpadding_mean_absolute_difference": float(difference[valid].mean().item()),
        "all_positions_max_absolute_difference": float(difference.max().item()),
    }


def require_matching_metrics(expected, actual):
    """验证样本、目标数与正确数须相同，损失允许现有单精度浮点容差。"""
    for key in ("samples", "prediction_positions", "correct"):
        if actual[key] != expected[key]:
            raise RuntimeError(f"重新加载后的验证统计 {key} 不一致。")
    if not math.isclose(actual["loss"], expected["loss"], rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError("重新加载模型后的固定验证损失不一致。")


@torch.inference_mode()
def evaluate(model, samples, device, *, progress=False, desc="验证进度", position=0):
    """仅对有效预测位置汇总损失与准确率；不平均各批次的平均损失。"""
    started = time.perf_counter()
    model.eval()
    loss_sum, correct, targets = 0.0, 0, 0
    loader = DataLoader(samples, batch_size=1, shuffle=False, num_workers=0)
    for batch in tqdm(loader, desc=desc, unit="条", dynamic_ncols=True, mininterval=0.5,
                      disable=not progress, position=position, leave=position == 0):
        batch = {key: value.to(device) for key, value in batch.items()}
        selected = batch["labels"] != IGNORE_INDEX
        count = int(selected.sum().item())
        if count == 0:
            raise ValueError("验证样本没有可预测位置。")
        with torch.autocast(device_type=device.type, enabled=False):
            output = model(**batch, return_dict=True)
        loss = float(output.loss.item())
        if not math.isfinite(loss):
            raise RuntimeError("验证损失不是有限值。")
        loss_sum += loss * count
        correct += int((output.logits.argmax(dim=-1)[selected]
                        == batch["labels"][selected]).sum().item())
        targets += count
    if targets == 0:
        raise ValueError("验证集没有可预测位置。")
    return {"loss": loss_sum / targets, "accuracy": correct / targets,
            "correct": correct, "prediction_positions": targets,
            "samples": len(samples), "seconds": time.perf_counter() - started}


def build_lr_scheduler(optimizer, config, planned_updates):
    """按优化器更新次数调度，首段线性预热，随后从峰值线性降至零。"""
    warmup = config["warmup_updates"]
    def factor(step):
        if config["lr_scheduler"] == "constant":
            return 1.0
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (planned_updates - step) / max(1, planned_updates - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, dataset, optimizer, device, run_dir, config, *, scheduler=None,
                    on_validation=None):
    """复用老师的清零、前向、反向、更新流程，补充累积与显存控制。

    先统计累积窗口内的有效目标总数，再按其归一化各批损失后反向传播。
    这与按目标数平均梯度等价，但不先放大损失，避免半精度中间梯度溢出。
    支持脑浮点半精度或单精度；不使用普通半精度和梯度缩放器。
    """
    model.train()
    dataset.set_epoch(config.get("training_mask_epoch", 0))
    generator = torch.Generator().manual_seed(config.get("shuffle_seed", config["seed"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        generator=generator)
    precision = config["mixed_precision"]
    if precision not in {"bfloat16", "float32"}:
        raise ValueError("本训练流程只支持脑浮点半精度或单精度。")
    if precision == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("当前设备不支持配置中的脑浮点半精度。")
    planned_updates = min(config["optimizer_updates_cap"],
                          math.ceil(len(loader) / config["gradient_accumulation"]))
    optimizer.zero_grad(set_to_none=True)
    loss_sum, targets, correct, updates, microbatches = 0.0, 0, 0, 0, 0
    gradient_norms = []
    started = time.perf_counter()
    batches = iter(loader)
    with tqdm(total=planned_updates, desc="训练进度", unit="次更新", dynamic_ncols=True,
              mininterval=0.5, disable=not config.get("progress_bar", True)) as progress:
        for _ in range(planned_updates):
            # 只缓存编号与标签，最多 8 条留在内存；不同时保留多批模型计算图。
            window = list(islice(batches, config["gradient_accumulation"]))
            if not window:
                break
            counts = [int((batch["labels"] != IGNORE_INDEX).sum().item()) for batch in window]
            if any(count == 0 for count in counts):
                raise ValueError("训练样本没有可预测位置。")
            window_targets = sum(counts)
            for batch, count in zip(window, counts):
                batch = {key: value.to(device) for key, value in batch.items()}
                selected = batch["labels"] != IGNORE_INDEX
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=precision == "bfloat16"):
                    output = model(**batch, return_dict=True)
                    loss = output.loss
                    # 原损失是本批目标的均值，乘目标占比后即为整个窗口均值的贡献。
                    normalized_loss = loss.float() * (count / window_targets)
                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"第 {updates + 1} 次更新的训练损失不是有限值。")
                loss_sum += loss_value * count
                correct += int((output.logits.detach().argmax(dim=-1)[selected]
                                == batch["labels"][selected]).sum().item())
                targets += count
                microbatches += 1
                normalized_loss.backward()
                del output, loss, normalized_loss
            try:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"], error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"第 {updates + 1} 次更新、已处理 {microbatches} 条训练样本，"
                    f"梯度裁剪检查失败；计算格式为 {precision}，未执行本次参数更新。"
                ) from exc
            norm_value = float(norm.item())
            if not math.isfinite(norm_value) or norm_value <= 0:
                raise RuntimeError("梯度范数不是有限正数，不能确认有效参数更新。")
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            updates += 1
            gradient_norms.append(norm_value)
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.perf_counter() - started
            log_event(run_dir, "train_update", update=updates, microbatches=microbatches,
                      console=(not config.get("progress_bar", True) and
                               (updates == 1 or updates % config["console_log_every"] == 0
                                or updates == planned_updates)),
                      planned_updates=planned_updates, total_microbatches=len(loader),
                      progress_percent=100 * updates / planned_updates,
                      elapsed_seconds=elapsed,
                      estimated_remaining_seconds=elapsed / updates * (planned_updates - updates),
                      window_prediction_positions=window_targets,
                      cumulative_loss=loss_sum / targets, gradient_norm_before_clip=norm_value,
                      mixed_precision=precision, learning_rate=learning_rate,
                      next_learning_rate=float(optimizer.param_groups[0]["lr"]))
            progress.set_postfix_str(f"损失={loss_sum / targets:.4f}, 学习率={learning_rate:.2e}", refresh=False)
            progress.update(1)
            if on_validation is not None and (updates % config["eval_every"] == 0 or updates == planned_updates):
                on_validation(updates)
    if updates != planned_updates:
        raise RuntimeError(f"实际更新 {updates} 次，预期 {planned_updates} 次。")
    return {"loss": loss_sum / targets, "accuracy": correct / targets,
            "prediction_positions": targets, "microbatches": microbatches,
            "optimizer_updates": updates, "planned_updates": planned_updates,
            "gradient_norms_before_clip": gradient_norms,
            "seconds": time.perf_counter() - started,
            "time_includes_periodic_validation": on_validation is not None}


def resolve_training_source(model_dir, formal):
    """识别原始权重与本周保存的检查点，只读取来源记录，不恢复优化器状态。"""
    model_dir = model_dir.resolve()
    initial = {"training_mode": "pretrained_start", "training_mask_epoch": 0,
               "optimizer_state_restored": False, "scheduler_state_restored": False}
    if model_dir == DEFAULT_MODEL.resolve():
        return initial, None
    if not formal:
        return {**initial, "training_mode": "custom_model_check"}, None
    output_root = (HERE / "outputs").resolve()
    if not model_dir.is_relative_to(output_root) or model_dir.name != "model":
        raise ValueError("续训起点必须是本周 outputs/ 下保存的 model 目录。")
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (model_dir / name).is_file():
            raise ValueError(f"续训模型缺少文件：{name}，不会回退加载原始模型。")
    snapshot = model_dir.parent
    is_best = snapshot.parent.name == "best_checkpoints"
    source_run = snapshot.parent.parent if is_best else snapshot
    config_path, manifest_path = source_run / "config.json", source_run / "data_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise ValueError("续训起点缺少对应运行的配置与数据记录。")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_epoch = source_config.get("training_mask_epoch", 0)
    if is_best and snapshot.name == "update_000000" and source_config.get("training_mode") == "continue_saved_weights":
        # 起点快照没有经历该阶段的更新，继承其实际来源的遮盖轮次。
        previous_epoch = source_config.get("source_training_mask_epoch", 0)
    if not isinstance(previous_epoch, int) or previous_epoch < 0 or source_config.get("epochs") != 1:
        raise ValueError("来源运行的遮盖轮次或训练轮数不受当前续训入口支持。")
    if not source_manifest.get("validation_signature"):
        raise ValueError("来源运行缺少固定验证输入指纹。")
    return {
        **initial, "training_mode": "continue_saved_weights",
        "training_mask_epoch": previous_epoch + 1, "source_training_mask_epoch": previous_epoch,
        "source_run_directory": str(source_run), "source_checkpoint_kind": "best" if is_best else "final",
        "source_run_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_validation_signature": source_manifest["validation_signature"],
        "source_optimizer_state_available": False,
        "continuation_note": "加载已有权重开启新训练阶段；优化器、学习率调度和训练随机状态重新初始化。",
    }, source_manifest


def prepare_training_data(args, formal):
    """复用数据处理，支持从原始权重或已保存权重开始一轮训练。"""
    source_info, source_manifest = resolve_training_source(args.model_dir, formal)
    settings = FORMAL_CONFIG if formal else SMOKE_CONFIG
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = {
        **settings, **source_info, "phase": "task04_train" if formal else "task03_smoke",
        "python": sys.executable,
        "mixed_precision": "bfloat16" if bf16_available else "float32",
        "gradient_scaling": False,
        "seed": args.seed, "validation_mask_seed": args.validation_mask_seed,
        "max_length": args.max_length, "mask_probability": args.mask_probability,
        "validation_ratio": args.val_ratio,
        "train_candidates": min(args.train_limit, settings["train_candidate_cap"]),
        "validation_candidates": min(args.val_limit, settings["validation_candidate_cap"]),
        "source_model": str(args.model_dir.resolve()),
        "corpus_path": str(args.corpus.resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {name: metadata.version(name) for name in ["torch", "transformers", "tqdm"]},
        "progress_bar": not getattr(args, "no_progress", False),
        "shuffle_seed": args.seed + source_info["training_mask_epoch"],
    }
    for key in ("learning_rate", "lr_scheduler", "warmup_ratio", "eval_every"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("学习率必须为有限正数。")
    if not math.isfinite(config["warmup_ratio"]) or not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("预热比例必须大于等于 0 且小于 1。")
    if config["lr_scheduler"] == "constant":
        if getattr(args, "warmup_ratio", None) not in (None, 0):
            raise ValueError("固定学习率模式的预热比例应为 0。")
        config["warmup_ratio"] = 0.0
    train_rows, val_rows, corpus_info = load_and_split_corpus(
        args.corpus, args.seed, args.val_ratio,
    )
    if source_manifest is not None and corpus_info["sha256"] != source_manifest["corpus"]["sha256"]:
        raise ValueError("续训语料与来源运行不一致，请保留原语料。")
    tokenizer = load_local_tokenizer(args.model_dir)
    model_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if args.max_length > model_config["max_position_embeddings"]:
        raise ValueError("输入长度超过本地模型的位置数上限。")
    train_ds = MLMDataset(train_rows[:config["train_candidates"]], tokenizer,
                          args.max_length, seed=args.seed, mask_probability=args.mask_probability)
    val_ds = MLMDataset(val_rows[:config["validation_candidates"]], tokenizer,
                        args.max_length, seed=args.validation_mask_seed,
                        fixed_mask=True, mask_probability=args.mask_probability)
    train_ds.set_epoch(config["training_mask_epoch"])
    if source_manifest is not None:
        fixed = [val_ds[i] for i in range(len(val_ds))]
        if sample_signature(fixed, val_ds.source_indices) != source_manifest["validation_signature"]:
            raise ValueError("续训的固定验证输入或标签与来源运行不一致，请保留原划分、验证规模和遮盖配置。")
    if formal:
        config["optimizer_updates_cap"] = math.ceil(len(train_ds) / config["gradient_accumulation"])
    config["planned_updates"] = min(config["optimizer_updates_cap"],
                                    math.ceil(len(train_ds) / config["gradient_accumulation"]))
    config["warmup_updates"] = min(math.ceil(config["planned_updates"] * config["warmup_ratio"]),
                                   max(0, config["planned_updates"] - 1))
    return config, tokenizer, train_ds, val_ds, val_rows, corpus_info


def run_training(args, formal=False):
    """只在明确选择训练模式时写入新结果目录；计划模式完全只读。"""
    if Path(sys.prefix).resolve() != PY312.resolve():
        raise RuntimeError("本项目必须使用 D:/Anaconda3/envs/py312/python.exe。")
    if args.batch_size != 1 or args.max_length > 64:
        raise ValueError("当前训练方案要求每批 1 条、输入长度不超过 64。")
    if args.plan_only:
        config, _, train_ds, val_ds, _, corpus_info = prepare_training_data(args, formal)
        report = {
            "status": "plan_checked", "config": config, "corpus": corpus_info,
            "train_samples": len(train_ds), "validation_samples": len(val_ds),
            "excluded_train_indices": train_ds.excluded_indices,
            "excluded_validation_indices": val_ds.excluded_indices,
            "validation_prediction_positions": sum(check_sample(val_ds, i) for i in range(len(val_ds))),
            "last_accumulation_window_samples": (len(train_ds) - 1) % config["gradient_accumulation"] + 1,
            "model_loaded": False, "training_executed": False, "files_written": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report
    if not torch.cuda.is_available():
        raise RuntimeError("当前训练方案需要显卡；不会自动切换训练设备或方案。")
    output_root = HERE / "outputs"
    output_root.mkdir(exist_ok=True)
    prefix = "task04_train_" if formal else "task03_smoke_"
    run_dir = output_root / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(exist_ok=False)
    print(f"本次结果目录：{run_dir}", flush=True)
    try:
        return _run_training(args, run_dir, formal)
    except (Exception, KeyboardInterrupt):
        write_json(run_dir / "failure.json", {
            "status": "failed", "traceback": traceback.format_exc(),
            "note": "失败或中断不代表训练和保存加载检查已通过。",
        })
        raise


def save_model_checkpoint(model, tokenizer, reference_batch, snapshot_dir, device):
    """保存独立检查点及验收参考，不覆盖先前模型，不改变梯度检查点开关。"""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = snapshot_dir / "model"
    checkpoint_dir.mkdir(exist_ok=False)
    before_logits = reference_logits(model, reference_batch, device)
    write_json(snapshot_dir / "pre_save_state.json", model_state_fingerprint(model))
    from safetensors.torch import save_file
    save_file({"logits": before_logits,
               **{key: value.detach().cpu().contiguous().clone()
                  for key, value in reference_batch.items()}},
              str(snapshot_dir / "reload_reference.safetensors"))
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def verify_model_checkpoint(snapshot_dir, args, val_rows, expected_signature, expected_metrics,
                            device, *, progress=False, desc="模型验收"):
    """顺序加载检查点、复现分词与完整验证，返回结果，不改写保存文件。"""
    from safetensors.torch import load_file
    tokenizer = load_local_tokenizer(snapshot_dir / "model")
    dataset = MLMDataset(val_rows, tokenizer, args.max_length,
                         seed=args.validation_mask_seed, fixed_mask=True,
                         mask_probability=args.mask_probability)
    samples = [dataset[i] for i in range(len(dataset))]
    if sample_signature(samples, dataset.source_indices) != expected_signature:
        raise RuntimeError("重新加载分词器后，固定验证输入或标签发生变化。")
    model, loading = load_local_model(snapshot_dir / "model", device)
    if loading["unexpected_keys"]:
        raise RuntimeError("保存模型中出现额外权重。")
    expected_state = json.loads((snapshot_dir / "pre_save_state.json").read_text(encoding="utf-8"))
    if model_state_fingerprint(model) != expected_state:
        raise RuntimeError("保存前后完整模型状态的指纹不一致。")
    reference = load_file(str(snapshot_dir / "reload_reference.safetensors"))
    batch = {key: value.unsqueeze(0) for key, value in samples[0].items()}
    if any(not torch.equal(reference[key], value) for key, value in batch.items()):
        raise RuntimeError("保存前参考输入与固定验证集不一致。")
    comparison = compare_logits(reference["logits"], reference_logits(model, batch, device), batch["attention_mask"])
    if not comparison["passed"]:
        raise RuntimeError(f"非补齐位置的输出一致性检查失败：{comparison}")
    metrics = evaluate(model, samples, device, progress=progress, desc=desc)
    require_matching_metrics(expected_metrics, metrics)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"model_loading": loading, "validation": metrics, "comparison": comparison,
            "all_state_tensors_identical_after_reload": True}


def _run_training(args, run_dir, formal):
    started = time.perf_counter()
    print("正在准备语料与固定验证集。", flush=True)
    config, tokenizer, train_ds, val_ds, val_rows, corpus_info = prepare_training_data(args, formal)
    write_json(run_dir / "config.json", config)
    torch.manual_seed(config["shuffle_seed"])
    torch.cuda.manual_seed_all(config["shuffle_seed"])
    print(f"本次起始模型：{config['source_model']}", flush=True)
    if config["training_mode"] == "continue_saved_weights":
        print(f"加载已有权重开启新训练阶段，训练遮盖轮次 {config['training_mask_epoch']}，优化器和调度器重新初始化。", flush=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    fixed_samples = [val_ds[i] for i in range(len(val_ds))]
    validation_signature = sample_signature(fixed_samples, val_ds.source_indices)
    write_json(run_dir / "data_manifest.json", {
        "corpus": corpus_info, "train_source_indices": train_ds.source_indices,
        "validation_source_indices": val_ds.source_indices,
        "excluded_train_indices": train_ds.excluded_indices,
        "excluded_validation_indices": val_ds.excluded_indices,
        "validation_signature": validation_signature,
    })
    write_json(run_dir / "fixed_validation.json",
               fixed_sample_records(fixed_samples, val_ds.source_indices))
    log_event(run_dir, "data_ready", train_samples=len(train_ds),
              validation_samples=len(val_ds),
              mixed_precision=config["mixed_precision"],
              training_mode=config["training_mode"], training_mask_epoch=config["training_mask_epoch"],
              planned_updates=min(config["optimizer_updates_cap"],
                                  math.ceil(len(train_ds) / config["gradient_accumulation"])))
    model, loading = load_local_model(args.model_dir, device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("模型与分词器的词表大小不一致。")
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("训练方案要求更新全部模型参数，发现被冻结参数。")
    print("正在计算训练前的完整固定验证集指标，此阶段不更新参数。", flush=True)
    baseline = evaluate(model, fixed_samples, device, progress=config["progress_bar"], desc="训练前验证")
    write_json(run_dir / "baseline.json", baseline)
    log_event(run_dir, "baseline", **baseline)

    # 记录一个有直接预测梯度的权重矩阵，用实测变化证明更新确实发生。
    tracked_name = "cls.predictions.transform.dense.weight"
    tracked_before = model.get_parameter(tracked_name).detach().cpu().clone()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], foreach=False)
    scheduler = build_lr_scheduler(optimizer, config, config["planned_updates"])
    reference_batch = {key: value.unsqueeze(0) for key, value in fixed_samples[0].items()}
    validation_history, best = [], None
    if config["training_mode"] == "continue_saved_weights":
        # 起点也是有效候选；若追加训练没有改善，推荐模型保留起点状态。
        initial_snapshot = run_dir / "best_checkpoints" / "update_000000"
        save_model_checkpoint(model, tokenizer, reference_batch, initial_snapshot, device)
        write_json(initial_snapshot / "validation.json", baseline)
        best = {"update": 0, "metrics": baseline, "candidate_kind": "continuation_start",
                "snapshot_directory": str(initial_snapshot.resolve()),
                "directory": str((initial_snapshot / "model").resolve())}
        write_json(run_dir / "best_checkpoint.json", best)
        log_event(run_dir, "continuation_start_checkpoint", loss=baseline["loss"], directory=best["directory"])

    def periodic_validation(update):
        nonlocal best
        was_training = model.training
        try:
            metrics = evaluate(model, fixed_samples, device, progress=config["progress_bar"],
                               desc=f"第 {update} 次更新验证", position=1)
            validation_history.append({"update": update, "metrics": metrics})
            log_event(run_dir, "periodic_validation", update=update, **metrics)
            if best is None or metrics["loss"] < best["metrics"]["loss"]:
                snapshot_dir = run_dir / "best_checkpoints" / f"update_{update:06d}"
                save_model_checkpoint(model, tokenizer, reference_batch, snapshot_dir, device)
                write_json(snapshot_dir / "validation.json", metrics)
                best = {"update": update, "metrics": metrics, "candidate_kind": "trained_checkpoint",
                        "snapshot_directory": str(snapshot_dir.resolve()),
                        "directory": str((snapshot_dir / "model").resolve())}
                write_json(run_dir / "best_checkpoint.json", best)
                log_event(run_dir, "new_best_checkpoint", update=update, loss=metrics["loss"],
                          directory=best["directory"])
        finally:
            # 评估与参考输出计算会进入评估模式，必须恢复训练以启用随机失活与梯度检查点。
            model.train(was_training)

    train_metrics = train_one_epoch(model, train_ds, optimizer, device, run_dir, config,
                                    scheduler=scheduler, on_validation=periodic_validation)
    if formal and train_metrics["microbatches"] != len(train_ds):
        raise RuntimeError("正式训练没有遍历全部选定训练样本。")
    tracked_after = model.get_parameter(tracked_name).detach().cpu().clone()
    parameter_delta = float((tracked_after - tracked_before).abs().max().item())
    if parameter_delta <= 0:
        raise RuntimeError("被跟踪的预测权重未发生变化。")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("训练后存在非有限模型参数。")
    print("训练轮次结束，已完成最后一次完整验证，正在保存和验收模型。", flush=True)
    after = validation_history[-1]["metrics"]
    if sample_signature(fixed_samples, val_ds.source_indices) != validation_signature:
        raise RuntimeError("训练前后固定验证输入发生变化。")
    if baseline["prediction_positions"] != after["prediction_positions"]:
        raise RuntimeError("训练前后验证统计位置数不一致。")
    log_event(run_dir, "validation_after_training", **after)
    model.gradient_checkpointing_disable()
    save_model_checkpoint(model, tokenizer, reference_batch, run_dir, device)
    write_json(run_dir / "validation.json", after)
    checkpoint_dir = run_dir / "model"
    log_event(run_dir, "checkpoint_saved", directory=str(checkpoint_dir))

    # 先释放训练时的优化器与模型，避免在 4 GiB 显存中同时驻留两套模型。
    del scheduler, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    final_verification = verify_model_checkpoint(run_dir, args, val_rows[:config["validation_candidates"]], validation_signature,
                                                after, device, progress=config["progress_bar"], desc="最终模型验收")
    reload_info = final_verification["model_loading"]
    comparison = final_verification["comparison"]
    write_json(run_dir / "reload_comparison.json", comparison)
    reload_max_difference = comparison["all_positions_max_absolute_difference"]
    reloaded_metrics = final_verification["validation"]
    log_event(run_dir, "reload_verified", max_logit_difference=reload_max_difference,
              validation_loss=reloaded_metrics["loss"])
    best_root = Path(best["snapshot_directory"])
    best_verification = verify_model_checkpoint(best_root, args, val_rows[:config["validation_candidates"]], validation_signature,
                                               best["metrics"], device, progress=config["progress_bar"], desc="最佳模型验收")
    write_json(best_root / "reload_comparison.json", best_verification["comparison"])
    log_event(run_dir, "best_checkpoint_verified", update=best["update"], directory=best["directory"])
    summary = {
        "status": "passed", "run_directory": str(run_dir), "config": config,
        "source_model_loading": loading, "reload_model_loading": reload_info,
        "train_samples": len(train_ds), "validation_samples": len(val_ds),
        "validation_signature": validation_signature,
        "baseline": baseline, "training": train_metrics, "after_training": after,
        "reloaded_validation": reloaded_metrics,
        "periodic_validation": validation_history, "best_checkpoint": best,
        "best_checkpoint_verification": best_verification, "recommended_checkpoint": best["directory"],
        "tracked_parameter": tracked_name, "tracked_parameter_max_change": parameter_delta,
        "parameters_updated": True, "all_parameters_finite": True,
        "reload_max_logit_difference": reload_max_difference,
        "all_state_tensors_identical_after_reload": True, "reload_comparison": comparison,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20,
        "total_seconds": time.perf_counter() - started,
        "training_executed": True, "formal_training_executed": formal,
        "note": ("完成所选数据的一轮正式训练及保存加载验证；指标变化应据实分析。"
                 if formal else "仅验证小规模流程；不能代替任务 4 的正式训练。"),
    }
    write_json(run_dir / "summary.json", summary)
    # 完整逐步梯度记录保存在文件，终端只显示汇总，避免正式运行后刷屏。
    display_summary = {
        **summary,
        "training": {key: value for key, value in train_metrics.items()
                     if key != "gradient_norms_before_clip"},
    }
    print(json.dumps(display_summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


def verify_saved_run(args):
    """只验收已有训练成果，新增唯一报告；不更新参数或改写原始运行文件。"""
    if Path(sys.prefix).resolve() != PY312.resolve():
        raise RuntimeError("本项目必须使用 D:/Anaconda3/envs/py312/python.exe。")
    run_dir = args.verify_run.resolve()
    if not run_dir.is_relative_to((HERE / "outputs").resolve()) or not (run_dir / "model").is_dir():
        raise ValueError("只允许验收本周 outputs/ 下已有模型的运行目录。")
    report_path = run_dir / ("recovery_verification_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    print(f"独立验收目录：{run_dir}，不重新训练。", flush=True)
    try:
        report = _verify_saved_run(run_dir)
    except (Exception, KeyboardInterrupt):
        write_json(report_path, {"status": "failed", "verification_type": "post_training_recovery",
                                 "training_executed": False, "traceback": traceback.format_exc()})
        print(f"独立验收失败，新增报告：{report_path}", flush=True)
        raise
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "training_executed": False, "reloaded_validation": report["reloaded_validation"],
                      "limitations": report["limitations"]}, ensure_ascii=False, indent=2))
    return report


def _verify_saved_run(run_dir):
    started = time.perf_counter()
    read = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))
    config, manifest, records = read("config.json"), read("data_manifest.json"), read("fixed_validation.json")
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    updates = [event for event in events if event["event"] == "train_update"]
    after = next(event for event in reversed(events) if event["event"] == "validation_after_training")
    if not any(event["event"] == "checkpoint_saved" for event in events):
        raise RuntimeError("没有模型保存完成记录，不能恢复验收。")
    if config["epochs"] != 1:
        raise ValueError("当前恢复验收只支持已有的一轮训练方案。")
    train_count = len(manifest["train_source_indices"])
    expected_updates = min(config["optimizer_updates_cap"],
                           math.ceil(train_count / config["gradient_accumulation"]))
    if [event["update"] for event in updates] != list(range(1, expected_updates + 1)):
        raise RuntimeError("训练日志未记录全部计划更新，不能判定训练完成。")
    expected_rows = min(train_count, expected_updates * config["gradient_accumulation"])
    if updates[-1]["microbatches"] != expected_rows:
        raise RuntimeError("日志中的已处理训练样本数量与计划不一致。")
    if not all(math.isfinite(event["gradient_norm_before_clip"])
               and math.isfinite(event["cumulative_loss"]) for event in updates):
        raise RuntimeError("训练日志存在非有限损失或梯度。")

    tokenizer = load_local_tokenizer(run_dir / "model")
    corpus = Path(config["corpus_path"]).resolve()
    if not corpus.is_relative_to(HERE):
        raise ValueError("只读取本周原始语料。")
    _, val_rows, corpus_info = load_and_split_corpus(corpus, config["seed"], config["validation_ratio"])
    if corpus_info["sha256"] != manifest["corpus"]["sha256"]:
        raise RuntimeError("原始语料指纹与训练记录不一致。")
    dataset = MLMDataset(val_rows[:config["validation_candidates"]], tokenizer, config["max_length"],
                         seed=config["validation_mask_seed"], fixed_mask=True,
                         mask_probability=config["mask_probability"])
    regenerated = [dataset[i] for i in range(len(dataset))]
    samples = [{key: torch.tensor(record[key], dtype=torch.long)
                for key in (*MODEL_INPUT_KEYS, "labels")} for record in records]
    indices = [record["source_index"] for record in records]
    signature = sample_signature(samples, indices)
    if (signature != manifest["validation_signature"]
            or signature != sample_signature(regenerated, dataset.source_indices)
            or indices != manifest["validation_source_indices"]):
        raise RuntimeError("保存的固定验证输入、标签或分词结果无法复现。")
    target_count = sum(check_sample(dataset, i) for i in range(len(dataset)))
    if after["samples"] != len(samples) or after["prediction_positions"] != target_count:
        raise RuntimeError("原始评估记录与固定验证集规模不一致。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loading = load_local_model(run_dir / "model", device)
    if loading["unexpected_keys"] or model.config.vocab_size != len(tokenizer):
        raise RuntimeError("保存模型的结构或词表不匹配。")
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise RuntimeError("保存模型包含非有限参数。")
    state = model_state_fingerprint(model)
    from safetensors import safe_open
    # 直接核对文件中的全部张量；共享权重别名由加载后的完整状态指纹覆盖。
    legacy_key_mapping = {}
    checked_model_keys = set()
    with safe_open(str(run_dir / "model" / "model.safetensors"), framework="pt", device="cpu") as saved:
        saved_keys = list(saved.keys())
        for key in saved_keys:
            # 与本机模型库 conversion_mapping.py 中的旧版层归一化命名转换一致。
            model_key = key.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")
            if model_key in checked_model_keys:
                raise RuntimeError(f"保存文件中存在映射到同一模型参数的重复名称：{key}")
            checked_model_keys.add(model_key)
            if model_key != key:
                legacy_key_mapping[key] = model_key
            if state["tensors"].get(model_key) != tensor_fingerprint(saved.get_tensor(key)):
                raise RuntimeError(f"保存文件中的张量 {key} 与加载后内容不一致。")
    old_state_path = run_dir / "pre_save_state.json"
    old_reference_path = run_dir / "reload_reference.safetensors"
    if old_state_path.exists() and state != read("pre_save_state.json"):
        raise RuntimeError("保存前完整模型状态的指纹与加载后不一致。")
    batch = {key: value.unsqueeze(0) for key, value in samples[0].items()}
    if old_reference_path.exists():
        from safetensors.torch import load_file
        old_reference = load_file(str(old_reference_path))
        for key, value in batch.items():
            if not torch.equal(old_reference[key], value):
                raise RuntimeError("保存前参考输入与固定验证集不一致。")
        logits = old_reference["logits"]
        reference_source = "persisted_pre_save_logits"
    else:
        logits = reference_logits(model, batch, device)
        reference_source = "first_load_of_saved_checkpoint"
    print("模型和文件张量核对通过，正在重新评估完整固定验证集。", flush=True)
    metrics = evaluate(model, samples, device)
    require_matching_metrics(after, metrics)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("独立验收不应生成参数梯度。")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    second, second_loading = load_local_model(run_dir / "model", device)
    if second_loading["unexpected_keys"] or model_state_fingerprint(second) != state:
        raise RuntimeError("再次加载后的完整模型状态不一致。")
    comparison = compare_logits(logits, reference_logits(second, batch, device), batch["attention_mask"])
    if not comparison["passed"]:
        raise RuntimeError(f"非补齐位置的输出一致性检查失败：{comparison}")
    del second
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_verification = None
    if (run_dir / "best_checkpoint.json").exists():
        best = read("best_checkpoint.json")
        best_root = Path(best["snapshot_directory"]).resolve()
        if not best_root.is_relative_to(run_dir):
            raise ValueError("最佳检查点必须位于本次运行目录内。")
        verify_args = argparse.Namespace(max_length=config["max_length"],
                                         validation_mask_seed=config["validation_mask_seed"],
                                         mask_probability=config["mask_probability"])
        best_verification = verify_model_checkpoint(best_root, verify_args,
                            val_rows[:config["validation_candidates"]], signature, best["metrics"], device,
                            progress=config.get("progress_bar", True), desc="最佳模型独立验收")
    limitations = []
    if not old_state_path.exists():
        limitations.append("原运行没有保存训练结束时全部权重的指纹，不能事后补证保存前后全部权重逐元素一致。")
    if not old_reference_path.exists():
        limitations.append("原运行没有保存保存前参考输出；本次输出比较验证的是已保存模型两次独立加载的一致性。")
    original_summary = read("summary.json") if (run_dir / "summary.json").exists() else None
    if original_summary is None:
        limitations.append("原运行的在线训练准确率、权重变化量与显存峰值未落盘，无法从现有日志精确恢复。")
    return {
        "status": "passed", "verification_type": "post_training_recovery",
        "verified_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
        "run_directory": str(run_dir), "checkpoint_directory": str(run_dir / "model"),
        "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_config": config, "training_executed": False,
        "original_training_completed_approved_plan": True,
        "original_formal_training_completed": config["phase"] == "task04_train",
        "original_training": {"optimizer_updates": len(updates), "microbatches": expected_rows,
                              "prediction_positions": sum(event["window_prediction_positions"] for event in updates),
                              "loss": updates[-1]["cumulative_loss"],
                              "seconds": updates[-1]["elapsed_seconds"],
                              "accuracy": original_summary["training"]["accuracy"] if original_summary else None},
        "original_summary_available": original_summary is not None,
        "baseline": read("baseline.json"), "original_after_training": after,
        "reloaded_validation": metrics, "validation_signature": signature,
        "fixed_validation_reproduced": True, "all_current_parameters_finite": True,
        "loading": loading, "saved_tensor_count_verified": len(saved_keys),
        "legacy_saved_key_mapping": legacy_key_mapping,
        "current_loaded_state_fingerprint": state, "state_identical_between_independent_loads": True,
        "pre_save_state_fingerprint_available": old_state_path.exists(),
        "pre_save_logits_available": old_reference_path.exists(),
        "comparison_reference_source": reference_source, "reload_comparison": comparison,
        "best_checkpoint_verification": best_verification,
        "original_failure_record_preserved": (run_dir / "failure.json").exists(),
        "limitations": limitations, "verification_seconds": time.perf_counter() - started,
        "note": "本报告恢复已保存成果的验收，不改写原失败记录，不声称旧运行已通过后来新增的保存前检查。",
    }


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("必须为正整数。")
    return value


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="只检查数据与前向计算，不训练")
    mode.add_argument("--smoke-test", action="store_true", help="受限小规模训练，最多 4 次更新")
    mode.add_argument("--train", action="store_true", help="正式训练一轮，最多 20000 条训练候选")
    mode.add_argument("--verify-run", type=Path, help="独立验收已有结果目录，新增报告，不训练")
    parser.add_argument("--plan-only", action="store_true", help="配合 --train，只读检查正式训练计划")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL,
                        help="起始权重目录，可指定本周已保存的 model 目录进行新阶段续训")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-mask-seed", type=int, default=43)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=positive_int, default=64)
    parser.add_argument("--train-limit", type=positive_int, default=20000)
    parser.add_argument("--val-limit", type=positive_int, default=2000)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, help="峰值学习率，默认 0.00002")
    parser.add_argument("--lr-scheduler", choices=["linear", "constant"], help="线性预热衰减或固定学习率")
    parser.add_argument("--warmup-ratio", type=float, help="预热更新比例，正式训练默认 0.05")
    parser.add_argument("--eval-every", type=positive_int, help="每隔多少次更新验证，正式训练默认 500")
    parser.add_argument("--no-progress", action="store_true", help="关闭终端进度条，保留日志输出")
    args = parser.parse_args()
    if args.plan_only and not args.train:
        parser.error("--plan-only 必须与 --train 一起使用。")
    if args.verify_run is not None:
        verify_saved_run(args)
    elif args.check_only:
        run_checks(args)
    else:
        run_training(args, formal=args.train)


if __name__ == "__main__":
    main()

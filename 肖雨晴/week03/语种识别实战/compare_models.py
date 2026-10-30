# -*- coding: utf-8 -*-
"""训练并比较四种语种识别模型在 v12 测试集上的效果。

默认先用相同数据和训练参数分别训练四个模型，再汇总测试集指标：
accuracy、macro-F1、最常混淆的语种组合和各语种准确率。

    python compare_models.py
    python compare_models.py --reuse-checkpoints
"""

import argparse
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

from train_langid_cls import (
    DATA_DIR,
    DROPOUT,
    EMBED_DIM,
    EPOCHS,
    HIDDEN_DIM,
    LABELS,
    MODEL_CONFIGS,
    NUM_THREADS,
    VOCAB_PATH,
    TextDataset,
    build_model,
    confusion,
    load_split,
    macro_f1,
    predict,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "langid_v12")


def checkpoint_path(checkpoint_dir, model_type):
    return os.path.join(checkpoint_dir, f"{model_type}_v12.pt")


def train_all(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    script = os.path.join(HERE, "train_langid_cls.py")
    for model_type in MODEL_CONFIGS:
        save_path = checkpoint_path(args.checkpoint_dir, model_type)
        command = [
            sys.executable,
            script,
            "--model", model_type,
            "--data-dir", args.data_dir,
            "--vocab", args.vocab,
            "--epochs", str(args.epochs),
            "--threads", str(args.threads),
            "--save", save_path,
        ]
        if args.per_class:
            command.extend(["--per-class", str(args.per_class)])
        print(f"\n{'=' * 72}\n训练模型：{model_type}\n{'=' * 72}", flush=True)
        subprocess.run(command, check=True)


def evaluate_checkpoint(path, expected_model_type, test_rows, batch_size, threads):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "rnn_layernorm_dropout")
    if model_type != expected_model_type:
        raise ValueError(
            f"{path} 保存的是 {model_type}，预期为 {expected_model_type}"
        )
    labels = checkpoint["labels"]
    vocab = checkpoint["vocab"]

    if labels != LABELS:
        raise ValueError(
            f"{path} 的语种标签顺序与当前 v12 配置不一致：{labels}"
        )
    label2id = {label: i for i, label in enumerate(labels)}
    unknown = sorted({label for _, label in test_rows if label not in label2id})
    if unknown:
        raise ValueError(f"测试集包含模型未配置的标签：{unknown}")

    model = build_model(
        len(vocab),
        len(labels),
        model_type=model_type,
        embed_dim=checkpoint.get("embed_dim", EMBED_DIM),
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        dropout=checkpoint.get("dropout", DROPOUT),
    )
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(
        TextDataset(test_rows, vocab, label2id),
        batch_size=batch_size,
        shuffle=False,
    )
    torch.set_num_threads(threads)
    preds, golds = predict(model, loader)
    accuracy = sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
    return {
        "model_type": model_type,
        "name": MODEL_CONFIGS[model_type]["name"],
        "accuracy": accuracy,
        "macro_f1": macro_f1(preds, golds, len(labels)),
        "preds": preds,
        "golds": golds,
        "matrix": confusion(preds, golds, labels),
        "labels": labels,
    }


def print_report(result, top_confusions):
    labels = result["labels"]
    print(f"\n--- {result['model_type']} ---")
    print(result["name"])
    print(f"accuracy = {result['accuracy']:.4f}    macro-F1 = {result['macro_f1']:.4f}")

    matrix = result["matrix"]
    pairs = [
        (matrix[gold][pred], labels[gold], labels[pred])
        for gold in range(len(labels))
        for pred in range(len(labels))
        if gold != pred and matrix[gold][pred] > 0
    ]
    pairs.sort(reverse=True)
    print(f"最常混淆的组合（真实 → 预测，前 {top_confusions}）：")
    if pairs:
        for count, gold, pred in pairs[:top_confusions]:
            print(f"  {gold} → {pred}: {count} 条")
    else:
        print("  无误分类")

    print("各语种准确率：")
    for class_id, label in enumerate(labels):
        indices = [i for i, gold in enumerate(result["golds"]) if gold == class_id]
        hits = sum(result["preds"][i] == class_id for i in indices)
        print(f"  {label}: {hits}/{len(indices)} = {hits / max(1, len(indices)):.4f}")


def main():
    parser = argparse.ArgumentParser(description="比较四种语种识别模型")
    parser.add_argument("--data-dir", default=DATA_DIR, help="v12 数据目录")
    parser.add_argument("--vocab", default=VOCAB_PATH, help="v12 字符表")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR,
                        help="四个模型检查点目录")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="自动训练时使用的 epoch 数")
    parser.add_argument("--threads", type=int, default=NUM_THREADS)
    parser.add_argument("--per-class", type=int, default=0,
                        help="自动训练时每类最多使用的训练样本数")
    parser.add_argument("--top-confusions", type=int, default=5,
                        help="每个模型展示的混淆组合数量")
    parser.add_argument("--reuse-checkpoints", action="store_true",
                        help="不重新训练，直接读取 checkpoint-dir 中已有的模型")
    args = parser.parse_args()

    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not args.reuse_checkpoints:
        train_all(args)

    missing = [
        checkpoint_path(args.checkpoint_dir, model_type)
        for model_type in MODEL_CONFIGS
        if not os.path.isfile(checkpoint_path(args.checkpoint_dir, model_type))
    ]
    if missing:
        raise SystemExit("缺少模型文件：\n  " + "\n  ".join(missing))

    test_rows = load_split(args.data_dir, "test")
    print(f"\n测试集：{len(test_rows)} 条；类别：{len(LABELS)} 个")
    results = []
    for model_type in MODEL_CONFIGS:
        path = checkpoint_path(args.checkpoint_dir, model_type)
        result = evaluate_checkpoint(
            path, model_type, test_rows, batch_size=128, threads=args.threads
        )
        results.append(result)

    print("\n=== 模型指标汇总 ===")
    print(f"{'模型结构':<34}{'accuracy':>12}{'macro-F1':>12}")
    for result in results:
        print(f"{result['model_type']:<34}{result['accuracy']:>12.4f}{result['macro_f1']:>12.4f}")
    for result in results:
        print_report(result, args.top_confusions)


if __name__ == "__main__":
    main()

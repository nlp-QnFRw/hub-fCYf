# -*- coding: utf-8 -*-
"""
metrics.py —— 两个基座共用的**同一个**指标函数

【为什么必须共用一个函数】
    四路方法对比时最常见的 bug：这边用 weighted F1、那边用 macro F1；
    这边 sklearn 默认 binary、那边 average='macro'。
    数字长得都像「0.55」，但含义不同，横向比较直接失效。
    本文件是唯一实现，训练脚本和对比脚本都调它。

【主指标为什么是 Macro-F1】
    15 类、最大类 5,955 条 vs 最小类 257 条。
    Accuracy 由头部类主导——把尾部三类全预测错，Accuracy 只掉 3 个点；
    而这三类恰恰是证券/农业这种业务上最值钱的。
    Macro-F1 每类等权，尾部类塌了立刻在数字上现形。
    Accuracy 仍然报，但只作为「大众口径」，两者差距本身就是诊断信号。

【per-class 指标的用途】
    两个基座的总体 Macro-F1 可能只差 1~2 个点，接口看起来「差不多」；
    但 per-class F1 往往此消彼长（RoBERTa 在财经类涨 3 点、在故事类掉 4 点）。
    只看总分就做选型，等于把「涨在哪、跌在哪」的信息全丢了。
"""

import numpy as np

from labels import LABELS, NUM_LABELS


def compute_metrics(y_true, y_pred, probs=None, labels=None) -> dict:
    """
    统一口径的多分类指标。

    y_true / y_pred: 等长整数序列（或 np 数组）
    probs: 可选 [N, C] 概率矩阵，给了才算 ECE
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    labels = list(range(NUM_LABELS)) if labels is None else list(labels)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = []
    for i, lab in enumerate(labels):
        per_class.append({
            "id": int(lab),
            "name": LABELS[lab] if lab < len(LABELS) else f"类别{lab}",
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        })

    # macro_f1 用 sklearn 再算一次，和 per-class 的均值互为校验
    metrics = {
        "n": int(len(y_true)),
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(np.mean(p)), 4),
        "macro_recall": round(float(np.mean(r)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if probs is not None:
        metrics["ece"] = expected_calibration_error(y_true, probs)
    return metrics


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """
    ECE：把置信度分桶，比「平均置信度」和「实际准确率」差多少。
    >0.1 基本不能拿置信度当阈值做「低置信转人工」。
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / len(conf)
    return round(float(ece), 4)


def delta_table(m_base: dict, m_other: dict, other_name: str = "RoBERTa") -> list:
    """
    per-class F1 变化表：按变化量从大到小排。

    为什么不直接比 F1 而是比「变化量(pp)」：
    F1 的绝对值受 support 影响，尾部类的 F1 天然波动大；
    看差值 + 样本量一起读，才知道一个 +4 点是真实提升还是小类噪声。
    """
    rows = []
    idx = {c["id"]: c for c in m_other["per_class"]}
    for c in m_base["per_class"]:
        o = idx[c["id"]]
        rows.append({
            "id": c["id"],
            "name": c["name"],
            "support": c["support"],
            "f1_base": c["f1"],
            "f1_other": o["f1"],
            "delta_pp": round((o["f1"] - c["f1"]) * 100, 2),
            "recall_base": c["recall"],
            "recall_other": o["recall"],
            "delta_recall_pp": round((o["recall"] - c["recall"]) * 100, 2),
            "precision_base": c["precision"],
            "precision_other": o["precision"],
            "delta_precision_pp": round((o["precision"] - c["precision"]) * 100, 2),
            "delta_label": f"Δ{other_name} - 基准",
        })
    rows.sort(key=lambda x: x["delta_pp"], reverse=True)
    return rows

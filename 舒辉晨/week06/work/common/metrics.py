"""
统一评测指标

和原项目同一套口径，改动只有一处：类别数从 15 变成 10。
主指标仍是 Macro-F1（每类等权），并额外给出「最容易被混淆的类别对」，
因为 10 类新闻里 财经/房产、科技/家居 本来就重叠，看总分会掩盖这一点。
"""

import numpy as np


def classification_report(y_true, y_pred, num_classes=10):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_pred >= 0                      # 生成式方法解析失败记 -1
    unparsed = int((~valid).sum())

    rep = {"n": int(len(y_true)), "unparsed": unparsed,
           "unparsed_rate": round(unparsed / len(y_true), 4) if len(y_true) else 0.0}
    if valid.sum() == 0:
        return rep

    yt, yp = y_true[valid], y_pred[valid]
    rep["acc"] = round(float(accuracy_score(yt, yp)), 4)
    # 解析失败按「答错」计入分母：生成式方法不能靠丢样本虚高
    rep["acc_incl_unparsed"] = round(float((yt == yp).sum()) / len(y_true), 4)
    rep["macro_f1"] = round(float(f1_score(yt, yp, average="macro")), 4)
    rep["weighted_f1"] = round(float(f1_score(yt, yp, average="weighted")), 4)

    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=list(range(num_classes)),
                                                 zero_division=0)
    rep["per_class"] = {
        int(c): {"precision": round(float(p[c]), 4), "recall": round(float(r[c]), 4),
                 "f1": round(float(f[c]), 4), "support": int(s[c])}
        for c in range(num_classes)
    }
    from sklearn.metrics import confusion_matrix
    rep["confusion_matrix"] = confusion_matrix(yt, yp, labels=list(range(num_classes))).tolist()
    return rep


def top_confusions(cm, classes, k=8):
    """
    从混淆矩阵里挑出「最常把 A 判成 B」的几对（不含对角线）。
    比看 10×10 矩阵快，直接告诉人「错误长什么样」。
    """
    pairs = []
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            if i != j and v:
                pairs.append((v, classes[i], classes[j]))
    pairs.sort(reverse=True)
    return [{"count": v, "true": a, "pred": b} for v, a, b in pairs[:k]]


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE：>0.1 说明置信度不能直接当阈值用"""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / len(conf)
    return round(float(ece), 4)


def pred_distribution(preds, num_classes=10):
    counts = {}
    for p in preds:
        counts[int(p)] = counts.get(int(p), 0) + 1
    return {"used_classes": sum(1 for k in counts if k >= 0), "counts": counts}


def to_markdown_table(results: dict):
    lines = ["| 方法 | 准确率(含失败) | Macro-F1 | 解析失败 |", "|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r.get('acc_incl_unparsed', r.get('acc', '-'))} | "
                     f"{r.get('macro_f1', '-')} | {r.get('unparsed', 0)} |")
    return "\n".join(lines)

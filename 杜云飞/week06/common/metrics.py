#region 0. 模块说明
"""
指标计算的**唯一实现**。所有方法都 import 这里，不允许各自写一份
`sklearn.classification_report`（否则"这脚本算的是 macro、那脚本算的是 weighted"，
对比时才暴露，且无法回溯）。

为什么不用 sklearn 的 classification_report：
  生成式方法会出现 `pred=None`（解析失败 / 拒答）。sklearn 的 `labels=range(14)`
  会把 -1 的预测**整体忽略**，等于把失败样本从分母里悄悄删掉，指标虚高。
  本实现对"失败"的处理方式是明确的一条：**计入分母、算该真值类别的 FN、不算任何类的 FP**。

三档口径（生成式必备，规范 §3.10.3）：
  acc_incl_unparsed : 全样本计入分母（主表口径，对外唯一可信）
  acc_parsed_only   : 只算解析成功的（辅助诊断，**不能单独汇报**）
  parse_fail_rate   : 失败率单列

使用方式：
  from common.metrics import compute_metrics
  m = compute_metrics(golds, preds)   # golds/preds 都是类名字符串，pred 可为 None
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from common.labels import ID2LABEL, LABEL2ID, LABELS, NUM_LABELS

#endregion

#region 2. 主函数：分类指标
def compute_metrics(golds: Sequence[str], preds: Sequence[str | None],
                    probs: np.ndarray | None = None,
                    labels: Sequence[str] | None = None) -> dict:
    """计算全部指标。

    参数：
      golds  : 真值类名序列（长度 n）
      preds  : 预测类名序列（长度 n，**允许 None 表示拒答/解析失败**）
      probs  : 可选，形状 (n, len(labels)) 的概率矩阵（用于 ECE；判别式可用 softmax 得到）
      labels : **标签空间**，默认 14 个细分类。算"主题层（6 组）"指标时必须传 GROUPS，
               否则组名会全部映射成 -1，acc 虚高成 1.0（本项目踩过：`教育` 既是细分类
               又是组名，混在一起还会出现 -1 与 id=5 的碰撞）
    返回：
      dict，字段固定（下游 compare.py 依赖字段名，不可随意改）：
        n / acc / acc_parsed_only / macro_f1 / weighted_f1 / parse_fail_rate
        per_class: {类名: {precision, recall, f1, support}}
        confusion_matrix: K×K 的 list（仅统计解析成功的样本）
        n_abstain / ece
    """
    if len(golds) != len(preds):
        raise ValueError(f"长度不一致：golds={len(golds)} preds={len(preds)}")

    names: list[str] = list(labels) if labels is not None else list(LABELS)
    name2id = {name: i for i, name in enumerate(names)}
    K = len(names)

    n = len(golds)
    y_true = np.array([name2id.get(g, -1) for g in golds], dtype=np.int64)
    # 预测侧：None 或非法类名统一记为 -1（拒答），参与 acc 分母，不参与任何类的 FP
    y_pred = np.array([name2id.get(p, -1) if p is not None else -1 for p in preds], dtype=np.int64)

    correct = int((y_pred == y_true).sum())
    n_abstain = int((y_pred < 0).sum())
    n_valid_pred = n - n_abstain

    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    supports: list[int] = []
    for c in range(K):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[names[c]] = {
            "precision": round(prec, 6), "recall": round(rec, 6),
            "f1": round(f1, 6), "support": support,
        }
        if support > 0:  # 训练集/验证集缺失某类时该类的 F1 无定义，不计入 macro
            f1s.append(f1)
        supports.append(support)

    # weighted F1 只按出现过的类加权
    w_num = sum(f * s for f, s in zip(f1s, [s for s in supports if s > 0]))
    w_den = sum(s for s in supports if s > 0)

    # 混淆矩阵只统计"真值与预测都合法"的样本，拒答另计
    mask = (y_pred >= 0) & (y_true >= 0)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true[mask], y_pred[mask]):
        cm[t, p] += 1

    out = {
        "n": n,
        "acc": round(correct / n, 6) if n else 0.0,
        "acc_parsed_only": round(correct / n_valid_pred, 6) if n_valid_pred else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 6) if f1s else 0.0,
        "weighted_f1": round(float(w_num / w_den), 6) if w_den else 0.0,
        "parse_fail_rate": round(n_abstain / n, 6) if n else 0.0,
        "n_abstain": n_abstain,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "ece": round(expected_calibration_error(y_true, y_pred, probs), 6) if probs is not None else None,
    }
    return out
#endregion

#region 3. 校准误差 ECE（置信度是否可信）
def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                               probs: np.ndarray | None, n_bins: int = 15) -> float:
    """Expected Calibration Error。

    定义：把样本按预测置信度分 n_bins 桶，ECE = Σ (|bucket| / n) · |acc(bucket) − conf(bucket)|。
    **拒答样本不参与**（它们没有置信度）。返回 0~1，越小越可信。

    为什么关心它：下游若要用"conf < 0.8 转人工"，则 conf 必须是真概率；
    softmax 输出通常偏自信（ECE 常在 0.05~0.15），过热/欠热时需要温度缩放。
    """
    if probs is None:
        return 0.0
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (y_pred >= 0) & (pred == y_pred)  # 只统计与最终预测一致的部分，避免两套口径打架
    if ok.sum() == 0:
        return 0.0
    conf, hit = conf[ok], (y_true[ok] == pred[ok]).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(ece)
#endregion

#region 4. 置信区间（别拿小数点当结论）
def bootstrap_ci(golds: Sequence[str], preds: Sequence[str | None],
                 n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """对 macro-F1 做 bootstrap，返回 (点估计, 下界, 上界)。

    用途：当两个方法的差只有 0.3pp 时，先看区间是否重叠；重叠就说明"这个差
    在 4000 条测试集上不可分辨"，需要配对检验（eval/paired_test.py）才能下结论。
    """
    rng = random.Random(seed)
    n = len(golds)
    point = compute_metrics(golds, preds)["macro_f1"]
    vals: list[float] = []
    idx = list(range(n))
    for _ in range(n_boot):
        pick = [idx[rng.randrange(n)] for _ in range(n)]
        vals.append(compute_metrics([golds[i] for i in pick], [preds[i] for i in pick])["macro_f1"])
    vals.sort()
    lo = vals[int(n_boot * alpha / 2)]
    hi = vals[int(n_boot * (1 - alpha / 2)) - 1]
    return point, lo, hi
#endregion

#region 5. markdown 表格（报告与终端共用）
def to_markdown_table(rows: list[dict], columns: list[str],
                      headers: list[str] | None = None) -> str:
    """把 list[dict] 渲染成 markdown 表。

    参数：
      rows    : 每行一个 dict
      columns : 取哪些键、按什么顺序
      headers : 表头显示名（默认用键名）
    缺失字段显示 "N/A"，浮点统一 4 位小数。
    """
    heads = headers or columns
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, None)
            if v is None:
                cells.append("N/A")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
#endregion

#region 6. 自检
if __name__ == "__main__":
    g = ["财经", "体育", "时尚", "财经", "彩票"]
    p = ["财经", "体育", None, "股票", "彩票"]
    m = compute_metrics(g, p)
    print(f"n={m['n']} acc={m['acc']} macro_f1={m['macro_f1']} 拒答={m['n_abstain']}")
    assert m["acc"] == 0.6, m["acc"]                       # 3/5 正确，None 计入分母
    assert m["n_abstain"] == 1
    assert m["per_class"]["时尚"]["recall"] == 0.0           # 拒答 = 该真值类的 FN
    print(to_markdown_table([{"method": "bert", "macro_f1": 0.9123, "acc": None}],
                            ["method", "macro_f1", "acc"]))

    # 回归测试：自定义标签空间（主题层）。用细分类名去算组指标时必须算"错"，
    # 而不是因为映射不到就静默变成 -1 与 -1 相等（这会虚高成 acc=1.0）
    from common.labels import GROUPS

    gm = compute_metrics(["财经投资", "文体娱乐"], ["财经投资", "财经"], labels=GROUPS)
    assert gm["acc"] == 0.5, f"主题层口径错误：{gm['acc']}"
    pure = compute_metrics(["财经投资", "文体娱乐"], ["财经投资", "文体娱乐"], labels=GROUPS)
    assert pure["acc"] == 1.0 and pure["macro_f1"] > 0.99, "主题层全对时 macro 应为 1"
    print("指标自检通过 ✓（含自定义标签空间）")
#endregion

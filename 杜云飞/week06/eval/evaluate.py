#region 0. 模块说明
"""
**唯一评测入口**：读 runs/<run_id>/pred_*.jsonl → 算指标 → 写 metrics.json。

为什么指标必须由这一个脚本算：
  六条方法线各自算指标，迟早出现"这个用 macro、那个用 weighted""这个把解析失败
  剔除、那个没剔除"。汇总时才发现口径不一致，前面的跑全白跑。

两个测试集的分工（固定约定）：
  test_bal      每类 ≥300 条  → **macro_f1 主指标**（长尾类的 F1 才有统计意义）
  test_nat_10k  按自然分布    → **acc 辅指标**（贴近线上真实分布）
  两者都算、都写进 metrics.json，报告里分开列，不混用。

ECE 的来源（按优先级）：
  runs/<run_id>/probs_test_bal.npy     BERT 保存的概率矩阵
  runs/<run_id>/probs.npz              其它方法（rule / tfidf / llm）保存的字典
  都没有 → ece 记 None（生成式方法本来就没有概率）

使用方式：
  python eval/evaluate.py --run t1__bert__xxx
  python eval/evaluate.py --all
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import list_runs, load_preds  # noqa: E402
from common.labels import ID2LABEL, LABELS, NUM_LABELS  # noqa: E402
from common.metrics import compute_metrics, to_markdown_table  # noqa: E402
from common.paths import RUNS  # noqa: E402
from common.text import ELLIPSIS  # noqa: F401,E402  （仅为了确认依赖可导入，无实际用途）

#endregion

#region 2. 全局配置
SPLIT_MAIN: str = "test_bal"  # 主指标口径
SPLIT_NAT: str = "test_nat_10k"  # 辅指标口径
#endregion

#region 3. 函数：加载概率
def load_probs(run_id: str, split: str, n_expected: int) -> np.ndarray | None:
    """尝试读概率矩阵，形状必须与预测条数对齐；对不上就返回 None（宁缺勿错）。"""
    d = RUNS / run_id
    p_npy = d / f"probs_{split}.npy"
    if p_npy.exists():
        arr = np.load(p_npy)
        if arr.ndim == 2 and arr.shape[0] == n_expected and arr.shape[1] == NUM_LABELS:
            return arr
    p_npz = d / "probs.npz"
    if p_npz.exists():
        try:
            with np.load(p_npz) as z:
                if split in z:
                    arr = z[split]
                    if arr.ndim == 2 and arr.shape[0] == n_expected and arr.shape[1] == NUM_LABELS:
                        return arr
        except (ValueError, OSError):
            return None
    return None
#endregion

#region 4. 函数：评测单个 run
def evaluate_run(run_id: str, verbose: bool = True) -> dict | None:
    """读 pred 文件并写 metrics.json；缺文件返回 None（不抛异常）。"""
    out: dict = {"run_id": run_id}
    got_any = False

    for split, role in ((SPLIT_MAIN, "main"), (SPLIT_NAT, "nat")):
        preds = load_preds(run_id, split)
        if preds is None:
            if verbose:
                print(f"  [跳过] {run_id} 无 pred_{split}.jsonl")
            continue
        got_any = True
        golds = [p["gold"] for p in preds]
        pr = [p["pred"] for p in preds]
        probs = load_probs(run_id, split, len(preds))
        m = compute_metrics(golds, pr, probs)

        # 延延迟：pred 里每行都写了 latency_ms（批处理时同一个值），取中位数
        lat = float(np.median([p.get("latency_ms", 0.0) for p in preds])) if preds else 0.0
        m["ms_per_sample"] = round(lat, 3)

        # 业务主题层的指标（14 类 → 6 组），回答"细分类错了但主题对了没"。
        # **必须显式传 labels=GROUPS**：否则组名会被当作未知标签映射成 -1，
        # 而 `教育` 恰好既是细分类又是组名，会与 id=5 发生碰撞，指标完全失真。
        from common.labels import GROUPS, to_group

        gm = compute_metrics([to_group(g) for g in golds],
                             [to_group(p) if p else None for p in pr],
                             labels=GROUPS)
        m["group_macro_f1"] = gm["macro_f1"]
        m["group_acc"] = gm["acc"]

        if role == "main":
            out.update({
                "macro_f1": m["macro_f1"], "acc_bal": m["acc"],
                "weighted_f1": m["weighted_f1"], "n_eval": m["n"],
                "parse_fail_rate": m["parse_fail_rate"], "n_abstain": m["n_abstain"],
                "ece": m["ece"], "group_macro_f1": m["group_macro_f1"],
                "confusion_matrix": m["confusion_matrix"],
                "per_class": m["per_class"],
                "ms_per_sample": m["ms_per_sample"],
            })
        else:
            out.update({
                "acc": m["acc"], "macro_f1_nat": m["macro_f1"],
                "n_eval_nat": m["n"], "group_acc_nat": m["group_acc"],
            })
        out[f"detail_{split}"] = m

    if not got_any:
        return None

    # 兼容层：老 run 可能只有 f1 字段，统一暴露成 macro_f1（规范 §8）
    mf = out.get("macro_f1", out.get("f1"))
    out["macro_f1"] = mf

    p = RUNS / run_id / "metrics.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"  [写入] {p}")
    return out
#endregion

#region 5. 打印与结论
def print_report(run_id: str, m: dict) -> None:
    """打印单 run 的对比表 + 逐类明细 + 关键结论 / 思考题。"""
    print(f"\n{'=' * 72}\n{run_id}\n{'=' * 72}")
    row = {"方法": run_id, "macro-F1": m.get("macro_f1"), "acc(nat)": m.get("acc"),
           "加权F1": m.get("weighted_f1"), "主题F1": m.get("group_macro_f1"),
           "拒答率": m.get("parse_fail_rate"), "ECE": m.get("ece"),
           "ms/条": m.get("ms_per_sample")}
    print(to_markdown_table([row], list(row.keys())))

    pc = m.get("per_class") or {}
    if pc:
        rows = [{"类别": k, **v} for k, v in pc.items()]
        print("\n逐类指标（按 support 降序）：")
        print(to_markdown_table(sorted(rows, key=lambda r: -r["support"]),
                                ["类别", "precision", "recall", "f1", "support"]))

    mf = m.get("macro_f1") or 0.0
    acc = m.get("acc") or 0.0
    print("\n【关键结论】")
    print(f"- 主指标 macro-F1 = {mf:.4f}（test_bal，每类≥300 条），自然分布 acc = {acc:.4f}（test_nat）。")
    print(f"- 两者差 {abs(acc - mf) * 100:.1f}pp：差距越大，说明模型越偏向头部大类"
          f"（acc 被大类拉高、macro-F1 被小类拖低）。")
    if m.get("parse_fail_rate"):
        print(f"- 解析失败率 {m['parse_fail_rate'] * 100:.1f}%：这是**独立指标**，"
              f"不能只汇报 acc(parsed_only)。")
    if pc:
        worst = min(pc.items(), key=lambda kv: kv[1]["f1"])
        print(f"- 最差类别「{worst[0]}」F1={worst[1]['f1']:.3f}，support={worst[1]['support']}："
              f"这是下一个该看的错例来源。")

    print("\n【思考题】")
    print("1. macro-F1 与 acc 的差值里，有多少来自类别不均衡、有多少来自模型本身？"
          "（对照 class_weight 消融那一格）")
    print("2. 主题层 F1 明显高于细分类 F1 时，业务上是否可以只做 6 分类？代价是什么？")
    print("3. 最差类别的错误里，是「真错」还是「标注体系本身模糊」？（抽 20 条看错例）")
#endregion

#region 6. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="统一评测入口")
    ap.add_argument("--run", default=None, help="run_id")
    ap.add_argument("--all", action="store_true", help="评测全部 run")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    targets = [args.run] if args.run else (list_runs() if args.all else [])
    if not targets:
        sys.exit("请指定 --run <run_id> 或 --all")

    ok = 0
    for rid in targets:
        m = evaluate_run(rid, verbose=not args.quiet)
        if m is None:
            continue
        ok += 1
        if not args.quiet:
            print_report(rid, m)
    print(f"\n完成 {ok}/{len(targets)} 个 run")
    print("下一步：python eval/compare.py          # 汇总成表 + summary.md")
    print("        python eval/paired_test.py --run <A> --vs <B>   # 配对检验")


if __name__ == "__main__":
    main()
#endregion

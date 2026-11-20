#region 0. 模块说明
"""
McNemar 配对检验：判断两个方法在**同一批样本**上的差异是否显著。

为什么不能看小数点：
  在 4200 条测试集上，macro-F1 差 0.3pp 完全可能是噪声。而"两条独立抽样各估一次
  ±9.7pp"的错误做法（用两批不同样本估计）更会给出荒谬的置信区间。
  正确做法是配对：同一批样本，逐条比较两个模型的对错。

统计量：
  b = A 对 B 错；c = A 错 B 对
  H0：b 与 c 服从 p=0.5 的二项分布（两者没有系统差异）
  精确二项检验（样本量小时比卡方更稳）：p = 2 · Σ_{k=0}^{min(b,c)} C(n,k) 0.5^n

口径：
  拒答（pred=None）一律算"错"，与 evaluate.py 的主表口径一致。

使用方式：
  python eval/paired_test.py --run t1__bert__xxx --vs t4__lora__yyy
  python eval/paired_test.py --all-pairs          # 所有两两组合（输出较长）
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import list_runs, load_preds  # noqa: E402
from common.paths import RUNS  # noqa: E402
from common.metrics import to_markdown_table  # noqa: E402

#endregion

#region 2. 函数：McNemar 精确检验
def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确二项检验的 p 值。b/c 为两种"单边不一致"的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare_pair(run_a: str, run_b: str, split: str = "test_bal") -> dict | None:
    """在同一 split 上配对比较两个 run。

    返回：
      {run_a, run_b, n, acc_a, acc_b, n_a_only, n_b_only, n_both_right, n_both_wrong, p_value, verdict}
    任一方缺预测文件则返回 None（不抛异常，方便批量对比时跳过未跑的方案）。
    """
    pa, pb = load_preds(run_a, split), load_preds(run_b, split)
    if pa is None or pb is None:
        return None
    da = {r["id"]: r for r in pa}
    db = {r["id"]: r for r in pb}
    common = sorted(set(da) & set(db))
    if not common:
        return None

    ok_a = [da[i]["pred"] == da[i]["gold"] for i in common]
    ok_b = [db[i]["pred"] == db[i]["gold"] for i in common]
    b = sum(1 for x, y in zip(ok_a, ok_b) if x and not y)  # A 对 B 错
    c = sum(1 for x, y in zip(ok_a, ok_b) if not x and y)  # A 错 B 对
    p = mcnemar_exact(b, c)
    acc_a = sum(ok_a) / len(common)
    acc_b = sum(ok_b) / len(common)
    diff = acc_a - acc_b

    if p < 0.01:
        verdict = "高度显著"
    elif p < 0.05:
        verdict = "显著"
    else:
        verdict = "不可分辨（差异在噪声内）"
    return {
        "run_a": run_a, "run_b": run_b, "split": split, "n": len(common),
        "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4), "diff": round(diff, 4),
        "n_a_only": b, "n_b_only": c,
        "n_both_right": sum(1 for x, y in zip(ok_a, ok_b) if x and y),
        "n_both_wrong": sum(1 for x, y in zip(ok_a, ok_b) if not x and not y),
        "p_value": round(p, 6), "verdict": verdict,
    }
#endregion

#region 3. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="McNemar 配对检验")
    ap.add_argument("--run", default=None, help="方法 A 的 run_id")
    ap.add_argument("--vs", default=None, help="方法 B 的 run_id")
    ap.add_argument("--all-pairs", action="store_true", help="所有已评测 run 的两两组合")
    ap.add_argument("--split", default="test_bal")
    ap.add_argument("--out", default=str(RUNS / "paired_tests.json"))
    args = ap.parse_args()

    pairs: list[tuple[str, str]] = []
    if args.run and args.vs:
        pairs = [(args.run, args.vs)]
    elif args.all_pairs:
        runs = [r for r in list_runs() if (RUNS / r / f"metrics.json").exists()]
        pairs = list(itertools.combinations(runs, 2))
    else:
        sys.exit("请用 --run A --vs B，或 --all-pairs")

    rows, results = [], []
    for a, b in pairs:
        r = compare_pair(a, b, args.split)
        if r is None:
            print(f"[跳过] {a} vs {b}：缺 pred_{args.split}.jsonl")
            continue
        results.append(r)
        rows.append({k: r[k] for k in ("run_a", "run_b", "n", "acc_a", "acc_b",
                                       "diff", "p_value", "verdict")})

    if not rows:
        sys.exit("没有可比较的 pair")

    print(f"\nMcNemar 配对检验（{args.split}，同一批 {rows[0]['n']} 条样本）\n")
    print(to_markdown_table(rows, ["run_a", "run_b", "n", "acc_a", "acc_b", "diff", "p_value", "verdict"],
                            ["方法 A", "方法 B", "n", "A acc", "B acc", "差值", "p 值", "结论"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.out}")

    print("\n【关键结论】")
    sig = [r for r in results if r["p_value"] < 0.05]
    print(f"- {len(sig)}/{len(results)} 组对比达到 p<0.05；其余组**不能**说「谁更好」，"
          f"只能说「在这批样本上不可分辨」。")
    print("- 判定口径：|acc 差| < 1pp 且 p>0.05 时，优先选成本更低的那条路线。")
    print("\n【思考题】")
    print("1. 两模型都错的样本（n_both_wrong）占比多少？它们是标注问题还是能力问题？")
    print("2. 若 A 只在长尾类上赢，acc 上却看不出差异，应该用哪个指标做选型？")


if __name__ == "__main__":
    main()
#endregion

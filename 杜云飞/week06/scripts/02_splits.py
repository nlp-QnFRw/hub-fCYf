#region 0. 模块说明
"""
从 all.jsonl 生成全部分层切分。

产物（全部固定 seed=42，互不重叠）：
  train_100k   100k  按自然分布比例   —— BERT / LoRA 主训练集
  sub_1k/5k/10k/20k/30k/50k   嵌套子集（严格满足 1k ⊂ 5k ⊂ 10k ⊂ 20k ⊂ 30k ⊂ 50k ⊂ 100k）
                              —— 数据量曲线；30k 用于超参消融（10~12 min/格）
  val_10k      10k   按自然分布比例   —— 早停 / 选超参
  test_nat_10k 10k   按自然分布比例   —— **acc**（贴近真实部署的类别分布）
  test_bal     4200  每类 300 保底    —— **macro-F1**（主指标；长尾类的 F1 才有统计意义）

为什么要有两个测试集：
  按比例抽 10k 时，"时尚"只有约 177 条，其 F1 的 95% 置信区间半宽接近 ±7pp，
  任何"这个类提升了"的结论都不可信；而按比例算 acc 才贴近线上真实分布。
  所以：**macro-F1 在 test_bal 上算，acc 在 test_nat 上算**，两个都汇报，不混用。

实现要点：
  两遍流式读文件 + 每类蓄水池抽样（reservoir sampling）。74 万条全读进内存约需
  1.5GB，普通笔记本会顶到交换分区；蓄水池把内存压到 ~12 万条（约 200MB）。

使用方式：
  python scripts/02_splits.py
  python scripts/02_splits.py --train 10000 --smoke   # 小规模冒烟
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import nested_subset  # noqa: E402
from common.labels import LABELS  # noqa: E402
from common.paths import ALL_JSONL, SPLITS, split_file  # noqa: E402

#endregion

#region 2. 全局配置
SEED: int = 42
N_TRAIN: int = 100_000
N_VAL: int = 10_000
N_TEST_NAT: int = 10_000
BAL_PER_CLASS: int = 300  # test_bal 每类保底条数（300 时单类 F1 的 95%CI 半宽约 ±5pp）
SUB_SIZES: tuple[int, ...] = (1_000, 5_000, 10_000, 20_000, 30_000, 50_000)
#endregion

#region 3. 第一遍：统计每类总量
def count_labels(path: Path) -> tuple[Counter[str], int]:
    """流式统计每类条数（只解析 label 字段所在的行，但仍然整体 json.loads 简单可靠）。"""
    cnt: Counter[str] = Counter()
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cnt[json.loads(line)["label"]] += 1
            n += 1
    return cnt, n
#endregion

#region 4. 第二遍：每类蓄水池抽样
def reservoir_per_class(path: Path, caps: dict[str, int], seed: int = SEED) -> dict[str, list[dict]]:
    """对每个类别做蓄水池抽样，各保留 caps[label] 条。

    蓄水池算法：第 i 条样本以 cap/i 的概率替换池内随机一条，最终池内是
    全体样本的均匀随机子集——不需要知道总量，单遍流式即可。
    """
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    seen: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            lab = rec["label"]
            cap = caps.get(lab, 0)
            if cap <= 0:
                continue
            seen[lab] += 1
            if len(pools[lab]) < cap:
                pools[lab].append(rec)
            else:
                j = rng.randrange(seen[lab])
                if j < cap:  # 以 cap/seen 的概率替换
                    pools[lab][j] = rec
    for lab in pools:
        pools[lab].sort(key=lambda r: r["id"])  # 排序保证可复现
    return pools
#endregion

#region 5. 写文件
def dump(name: str, recs: list[dict]) -> Path:
    p = split_file(name)
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def summarize(name: str, recs: list[dict]) -> str:
    """返回一行摘要：总数 + 最大类/最小类。"""
    c = Counter(r["label"] for r in recs)
    head = c.most_common(1)[0] if c else ("-", 0)
    tail = min(((k, v) for k, v in c.items()), key=lambda x: x[1]) if c else ("-", 0)
    return f"{name:<14} {len(recs):>7} 条  最大 {head[0]}{head[1]} / 最小 {tail[0]}{tail[1]}"
#endregion

#region 6. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="生成分层切分（含两个测试集）")
    ap.add_argument("--train", type=int, default=N_TRAIN)
    ap.add_argument("--val", type=int, default=N_VAL)
    ap.add_argument("--test_nat", type=int, default=N_TEST_NAT)
    ap.add_argument("--bal_per_class", type=int, default=BAL_PER_CLASS)
    ap.add_argument("--smoke", action="store_true", help="冒烟：规模自动缩小 20 倍")
    args = ap.parse_args()

    if args.smoke:  # 冒烟只为验证流程，不产出可用于汇报的数字
        args.train, args.val, args.test_nat, args.bal_per_class = 2000, 400, 400, 30
        print("[冒烟模式] 规模已缩小，产物不可用于正式对比")

    if not ALL_JSONL.exists():
        sys.exit(f"缺少 {ALL_JSONL}，请先运行 python scripts/01_to_jsonl.py")

    total_cnt, total_n = count_labels(ALL_JSONL)
    print(f"all.jsonl 共 {total_n} 条，{len(total_cnt)} 类")

    # 每类需要的样本数 = 三个集合的比例份额 + test_bal 保底
    caps: dict[str, int] = {}
    for lab in LABELS:
        p = total_cnt.get(lab, 0) / max(1, total_n)
        caps[lab] = (int(args.train * p) + int(args.val * p) + int(args.test_nat * p)
                     + args.bal_per_class + 3)  # +3 余量，吸收取整误差

    pools = reservoir_per_class(ALL_JSONL, caps)

    train: list[dict] = []
    val: list[dict] = []
    test_nat: list[dict] = []
    test_bal: list[dict] = []
    rng = random.Random(SEED)

    for lab in LABELS:
        pool = pools.get(lab, [])
        rng.shuffle(pool)  # 蓄水池内顺序本身随机，但显式 shuffle 让切分逻辑一目了然
        p = total_cnt.get(lab, 0) / max(1, total_n)
        n_tr = int(args.train * p)
        n_va = int(args.val * p)
        n_te = int(args.test_nat * p)
        n_ba = args.bal_per_class
        need = n_tr + n_va + n_te + n_ba
        if len(pool) < need:
            print(f"[警告] {lab} 池内只有 {len(pool)} 条，少于需要的 {need} 条，按比例缩减")
            scale = len(pool) / max(1, need)
            n_tr, n_va, n_te, n_ba = (int(n_tr * scale), int(n_va * scale),
                                      int(n_te * scale), int(n_ba * scale))
        train += pool[:n_tr]
        val += pool[n_tr: n_tr + n_va]
        test_nat += pool[n_tr + n_va: n_tr + n_va + n_te]
        test_bal += pool[n_tr + n_va + n_te: n_tr + n_va + n_te + n_ba]

    train.sort(key=lambda r: (r["label"], r["id"]))
    val.sort(key=lambda r: (r["label"], r["id"]))
    test_nat.sort(key=lambda r: (r["label"], r["id"]))
    test_bal.sort(key=lambda r: (r["label"], r["id"]))

    files = [dump("train_100k", train), dump("val_10k", val),
             dump("test_nat_10k", test_nat), dump("test_bal", test_bal)]

    # 嵌套子集：从 train_100k 里按 (label, id) 取前缀，保证 5k ⊂ 20k ⊂ 50k ⊂ 100k
    for n in SUB_SIZES:
        if len(train) >= n:
            files.append(dump(f"sub_{n // 1000}k", nested_subset(train, n, seed=SEED)))

    print("\n" + "=" * 70)
    for name, recs in [("train_100k", train), ("val_10k", val),
                       ("test_nat_10k", test_nat), ("test_bal", test_bal)]:
        print(summarize(name, recs))

    # 重叠检查：三个集合必须完全不相交，否则指标会因"见过测试样本"而虚高
    ids = {"train": {r["id"] for r in train}, "val": {r["id"] for r in val},
           "test_nat": {r["id"] for r in test_nat}, "test_bal": {r["id"] for r in test_bal}}
    assert not (ids["train"] & ids["val"]), "train 与 val 重叠"
    assert not (ids["train"] & ids["test_nat"]), "train 与 test_nat 重叠"
    assert not (ids["val"] & ids["test_nat"]), "val 与 test_nat 重叠"
    print(f"\n重叠检查通过 ✓（test_nat 与 test_bal 也可能重叠 "
          f"{len(ids['test_nat'] & ids['test_bal'])} 条——它们口径不同，允许）")

    print("\ntest_bal 每类条数：")
    c = Counter(r["label"] for r in test_bal)
    print("  " + "  ".join(f"{k}:{c.get(k, 0)}" for k in LABELS))
    print(f"\n产物目录：{SPLITS}")
    print("下一步：python scripts/03_len_stats.py")


if __name__ == "__main__":
    main()
#endregion

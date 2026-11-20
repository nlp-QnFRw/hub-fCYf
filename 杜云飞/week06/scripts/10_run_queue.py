#region 0. 模块说明
"""
实验队列：把消融矩阵写成一张清单，串行跑、跳过已完成的、失败不中断，并记录耗时。

为什么需要它：
  消融矩阵有 30+ 个 run，逐个手敲命令既容易漏、也容易把参数敲错（比如把 --lr 3e-5
  敲成 3e-4，跑 40 分钟后才发现）。清单集中在一处，等于给实验做了版本管理。

行为约定：
  1. `--list` 只打印清单与预计耗时，不执行；
  2. 已有 `runs/<run_id>/metrics.json` 的 run **自动跳过**（可断点续跑）；
  3. 单个 run 失败会打印错误并继续下一个（一个坏参数不该毁掉整夜）；
  4. 全部结束后打印"已完成 / 跳过 / 失败"三栏与总耗时。

使用方式：
  python scripts/10_run_queue.py --list
  python scripts/10_run_queue.py --tier t1
  python scripts/10_run_queue.py --only t1__bert__bert-base-l256-100k-3ep
  python scripts/10_run_queue.py --tier t2 --start 3          # 从第 3 个开始
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.contract import new_run_id  # noqa: E402
from common.paths import RUNS  # noqa: E402

#endregion

#region 2. 实验清单
# 每条：(tier, 预计分钟, [命令参数...])
# 预计耗时来自本机实测（docs/PROBE.md），用于排期与砍单，不是拍脑袋
QUEUE: list[tuple[str, int, list[str]]] = [
    # ---------- Tier-1 主干对比（必做） ----------
    ("t1", 2, ["rule/classify_rule.py"]),
    ("t1", 6, ["tfidf/train_tfidf.py", "--feature", "char", "--clf", "lr"]),
    ("t1", 6, ["tfidf/train_tfidf.py", "--feature", "char", "--clf", "svm"]),
    ("t1", 9, ["tfidf/train_tfidf.py", "--feature", "word", "--clf", "lr"]),
    ("t1", 45, ["bert/train_bert.py", "--tier", "t1", "--train_split", "train_100k",
                "--epochs", "3", "--variant", "bert-base-l256-100k-3ep"]),
    ("t1", 12, ["bert/train_bert.py", "--tier", "t1", "--train_split", "sub_30k", "--epochs", "2",
                "--variant", "bert-base-l256-30k-2ep"]),
    ("t1", 12, ["bert/train_bert.py", "--tier", "t1", "--backbone", "roberta-wwm",
                "--train_split", "sub_30k", "--epochs", "2", "--variant", "roberta-l256-30k-2ep"]),
    ("t1", 12, ["bert/train_bert.py", "--tier", "t1", "--backbone", "macbert",
                "--train_split", "sub_30k", "--epochs", "2", "--variant", "macbert-l256-30k-2ep"]),
    # 数据量曲线：5k/20k/50k（100k 就是上面那条基线），子集严格嵌套，曲线才干净
    ("t1", 4, ["bert/train_bert.py", "--tier", "t1", "--train_split", "sub_5k",
               "--epochs", "3", "--variant", "bert-base-l256-5k-3ep"]),
    ("t1", 10, ["bert/train_bert.py", "--tier", "t1", "--train_split", "sub_20k",
                "--epochs", "3", "--variant", "bert-base-l256-20k-3ep"]),
    ("t1", 24, ["bert/train_bert.py", "--tier", "t1", "--train_split", "sub_50k",
                "--epochs", "3", "--variant", "bert-base-l256-50k-3ep"]),
    ("t1", 8, ["llm_local/classify_local.py", "--shots", "0", "--variant", "qwen2-0shot"]),
    ("t1", 8, ["llm_local/classify_local.py", "--shots", "1", "--variant", "qwen2-1shot"]),
    ("t1", 8, ["llm_local/classify_local.py", "--shots", "3", "--variant", "qwen2-3shot"]),
    ("t1", 8, ["llm_local/classify_local.py", "--shots", "5", "--variant", "qwen2-5shot"]),
    ("t1", 45, ["llm_lora/train_lora.py", "--train_split", "sub_10k", "--epochs", "3",
                "--r", "8", "--variant", "10k-r8-3ep-qkvo"]),
    # DeepSeek-V4.1-Flash 正式对比：n 给足 10000，test_bal 只有 4200 条会自动全取，
    # test_nat_10k 取满 10000 —— 这样才与其他方法严格同口径
    ("t1", 32, ["llm_api/classify_deepseek.py", "--shots", "0", "--thinking", "off",
                "--splits", "test_bal", "test_nat_10k", "--n", "10000",
                "--variant", "ds-v4flash-0shot"]),

    # ---------- Tier-2 单因素消融（30k×2ep，10~12 min/格） ----------
    ("t2", 6, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
               "--max_len", "128", "--variant", "l128-30k-2ep"]),
    ("t2", 18, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--max_len", "384", "--variant", "l384-30k-2ep"]),
    ("t2", 26, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--max_len", "512", "--batch_size", "16", "--variant", "l512-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--truncation", "head", "--variant", "trunc-head-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--truncation", "title_only", "--variant", "trunc-title-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--pool", "mean", "--variant", "pool-mean-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--freeze_layers", "6", "--variant", "freeze6-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--lr", "3e-5", "--variant", "lr3e5-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--lr", "1e-5", "--variant", "lr1e5-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--warmup_ratio", "0", "--variant", "warmup0-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--dropout", "0.3", "--variant", "dropout3-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--class_weight", "balanced", "--variant", "cweight-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--batch_size", "16", "--variant", "bs16-30k-2ep"]),
    # seed 复现性：同一配置换 2 个种子，用来判断上面那些 0.3pp 的差异是真实效应还是噪声
    # （没有它，"lr3e5 比默认高 0.4pp" 这类结论无法成立）
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--seed", "123", "--variant", "seed123-30k-2ep"]),
    ("t2", 12, ["bert/train_bert.py", "--train_split", "sub_30k", "--epochs", "2",
                "--seed", "2024", "--variant", "seed2024-30k-2ep"]),

    # ---------- Tier-3 生成式与鲁棒性（有时间再跑） ----------
    # [待充值] DeepSeek 3-shot：0-shot 已花掉账户余额（¥3.02 / 14200 条），
    # 余额不足后并发额度被压到 5~9，并发 12 直接返回 402。
    # 充值后把下面这条取消注释即可（约 ¥8~10、40 分钟），命令如下：
    # ("t3", 42, ["llm_api/classify_deepseek.py", "--shots", "3", "--thinking", "off",
    #             "--splits", "test_bal", "test_nat_10k", "--n", "10000",
    #             "--concurrency", "4", "--variant", "ds-v4flash-3shot"]),
    ("t3", 8, ["llm_local/classify_local.py", "--model", "qwen3-0.6b", "--shots", "0",
               "--variant", "qwen3-0shot"]),
    # few-shot 的格式对照：同样是 3 个示例，但不走多轮、压进同一条 user 消息。
    # 用来分清"few-shot 没用"和"多轮结构不适合 0.5B"
    ("t3", 18, ["llm_local/classify_local.py", "--shots", "3", "--shot_style", "inline",
                "--variant", "qwen2-3shot-inline"]),
    ("t3", 14, ["llm_local/classify_local.py", "--model", "qwen3-0.6b", "--shots", "2",
                "--variant", "qwen3-2shot"]),
    ("t3", 22, ["llm_lora/train_lora.py", "--train_split", "sub_5k", "--epochs", "3",
                "--r", "8", "--variant", "5k-r8-3ep-qkvo"]),
    ("t3", 22, ["llm_lora/train_lora.py", "--train_split", "sub_5k", "--epochs", "3",
                "--r", "16", "--variant", "5k-r16-3ep-qkvo"]),
    ("t3", 10, ["llm_lora/train_lora.py", "--train_split", "sub_1k", "--epochs", "3",
                "--r", "8", "--variant", "1k-r8-3ep-qkvo"]),
    ("t3", 22, ["llm_lora/train_lora.py", "--train_split", "sub_5k", "--epochs", "3",
                "--r", "8", "--targets", "qv", "--variant", "5k-r8-3ep-qv"]),
    ("t3", 90, ["llm_lora/train_lora.py", "--train_split", "sub_20k", "--epochs", "3",
                "--r", "8", "--variant", "20k-r8-3ep-qkvo"]),
    ("t3", 45, ["bert/train_bert.py", "--tier", "t3", "--train_split", "train_100k",
                "--epochs", "3", "--lr", "1e-5", "--variant", "bert-base-l256-100k-3ep-lr1e5"]),
]
#endregion

#region 3. 工具函数
def guess_run_id(cmd: list[str]) -> str | None:
    """从命令里推 run_id（用来判断是否已完成）。

    这里的规则必须与各脚本内部的 `new_run_id(...)` **完全一致**，否则：
    队列以为没跑过 → 重复跑一遍（白烧 45 分钟）；或者以为跑过了 → 实际没跑。
    因此各脚本都支持显式 `--tier`，队列优先用它，推不出来才回退默认规则。
    """
    script = cmd[0]
    args = cmd[1:]
    variant = args[args.index("--variant") + 1] if "--variant" in args else None
    tier_arg = args[args.index("--tier") + 1] if "--tier" in args else None
    if "--run_id" in args:
        return args[args.index("--run_id") + 1]

    # rule / tfidf 的命令行里带 --variant 是多余的（标签本身就是固定的），
    # 所以这两支**必须在 variant is None 判断之前**处理，否则会被判成"未完成"而每次重跑。
    if script.startswith("rule/"):
        return new_run_id("t1", "rule", "keyword")
    if script.startswith("tfidf/"):
        feat = args[args.index("--feature") + 1] if "--feature" in args else "char"
        clf = args[args.index("--clf") + 1] if "--clf" in args else "lr"
        n = "10k" if "--n_train" in args and args[args.index("--n_train") + 1] == "10000" else "100k"
        return new_run_id("t1", "tfidf", f"{feat}-{clf}-{n}")

    if variant is None:
        return None

    if script.startswith("bert/"):
        return new_run_id(tier_arg or "t2", "bert", variant)
    if script.startswith("llm_local/"):
        shots = int(args[args.index("--shots") + 1]) if "--shots" in args else 0
        return new_run_id(tier_arg or ("t1" if shots == 0 else "t3"), "llm-local", variant)
    if script.startswith("llm_lora/"):
        return new_run_id(tier_arg or "t3", "lora", variant)
    if script.startswith("llm_api/"):
        shots = int(args[args.index("--shots") + 1]) if "--shots" in args else 0
        return new_run_id(tier_arg or ("t1" if shots == 0 else "t3"), "llm-api", variant)
    return None


def is_done(run_id: str | None) -> bool:
    return bool(run_id) and (RUNS / run_id / "metrics.json").exists()


def run_one(cmd: list[str], log_path: Path) -> tuple[bool, float]:
    """执行一条命令，输出同时写日志。返回 (是否成功, 耗时秒)。"""
    t0 = time.time()
    full = [sys.executable, str(ROOT / cmd[0]), *cmd[1:]]
    print("  $ " + " ".join([cmd[0], *cmd[1:]]), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n\n{'=' * 90}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(cmd)}\n{'=' * 90}\n")
        log.flush()
        p = subprocess.run(full, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    ok = p.returncode == 0
    if not ok:
        print(f"  [失败] 退出码 {p.returncode}，详见 {log_path}")
    return ok, time.time() - t0
#endregion

#region 4. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="实验队列（串行、可续跑）")
    ap.add_argument("--tier", default=None, help="只跑某层：t1 / t2 / t3")
    ap.add_argument("--only", default=None, help="只跑某个 run_id")
    ap.add_argument("--start", type=int, default=0, help="从第 N 条开始（跳过前面）")
    ap.add_argument("--list", action="store_true", help="只列清单，不执行")
    ap.add_argument("--force", action="store_true", help="忽略已完成状态，强制重跑")
    ap.add_argument("--log", default=str(RUNS / "_queue.log"))
    args = ap.parse_args()

    items = [(t, m, c) for (t, m, c) in QUEUE if (args.tier is None or t == args.tier)]
    if args.only:
        items = [(t, m, c) for (t, m, c) in items
                 if guess_run_id(c) == args.only or args.only in " ".join(c)]
    items = items[args.start:]
    if not items:
        sys.exit("没有匹配的实验条目")

    total_min = sum(m for _, m, _ in items)
    print(f"共 {len(items)} 个 run，预计 {total_min / 60:.1f} 小时\n")
    for i, (tier, mins, cmd) in enumerate(items, 1):
        rid = guess_run_id(cmd)
        state = "已完成" if is_done(rid) and not args.force else "待跑"
        print(f"{i:>2}. [{tier}] {mins:>3}min  {state:<4} {rid or ' '.join(cmd)}")
    if args.list:
        return

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    t_all = time.time()
    for i, (tier, mins, cmd) in enumerate(items, 1):
        rid = guess_run_id(cmd)
        if is_done(rid) and not args.force:
            print(f"\n[{i}/{len(items)}] 跳过（已完成）：{rid}")
            skipped += 1
            continue
        print(f"\n[{i}/{len(items)}] 开始 {rid or cmd[0]}，预计 {mins} min")
        ok, sec = run_one(cmd, log_path)
        if ok:
            done += 1
            print(f"  ✓ 完成，用时 {sec / 60:.1f} min")
        else:
            failed += 1
            print(f"  ✗ 失败，用时 {sec / 60:.1f} min（继续下一个）")

    print("\n" + "=" * 70)
    print(f"完成 {done} / 跳过 {skipped} / 失败 {failed}，总耗时 {(time.time() - t_all) / 60:.1f} min")
    print(f"日志：{log_path}")
    print("下一步：python eval/evaluate.py --all && python eval/compare.py")


if __name__ == "__main__":
    main()
#endregion

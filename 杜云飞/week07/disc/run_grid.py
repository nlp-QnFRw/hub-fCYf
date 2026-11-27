# -*- coding: utf-8 -*-
"""run_grid.py — A 组判别式网格串行执行器（抗慢窗口 / 抗单点失败版）

今晚（11-26）的教训直接写进设计：
  1. **不 fail-fast**：单个配置失败或超时只记录，继续跑后面的。上一版 fail-fast 让
     第一个配置挂死就带走了整张网格，这是"跑了 4 小时一个数都没有"的直接原因之一。
  2. **每配置硬超时**（默认 900s）：到点 taskkill /F /T 杀进程树。一个卡住的评测
     不能拖死整晚。
  3. **逐个追加落盘**：每跑完一个配置就刷新 outputs/logs/disc_grid_results.json，
     任何时刻都能回答"现在到底完成了什么"。
  4. **--deadline HH:MM**：到点不再启动新配置；主循环跑完后若有剩余时间，对失败
     配置再重试一轮（慢窗口过去后往往一次就过）。
  5. **统一预算 epochs=4**：所有配置同一预算，层数/编码器/解码器消融才可比。
     （2 epoch 实测 underfit：CRF 的 NLL 还停在 38、实体角色召回只有 0.1–0.17，
     第 3–4 轮的收益明显还没吃完，故统一抬到 4。）

用法：
  python disc/run_grid.py                 # 全部 13 个配置
  python disc/run_grid.py --skip-done     # 断点续跑（已有指标 JSON 的跳过）
  python disc/run_grid.py --only bert12L_crf,bert4L_crf --timeout 600
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"D:\Programs\Dev\Miniconda3\envs\py312\python.exe"
TRAIN = ROOT / "disc" / "train.py"
METRICS = ROOT / "outputs" / "metrics" / "disc"
RESULT = ROOT / "outputs" / "logs" / "disc_grid_results.json"

# 顺序 = 优先级：先跑报告主结论需要的格子（层数消融 + 解码器对比），macbert 放最后
GRID = [
    ("bert12L_crf",     ["--encoder", "bert", "--decoder", "crf"]),
    ("bert12L_linear",  ["--encoder", "bert", "--decoder", "linear"]),
    ("bert4L_crf",      ["--encoder", "bert", "--decoder", "crf", "--bert_layers", "4"]),
    ("bert4L_linear",   ["--encoder", "bert", "--decoder", "linear", "--bert_layers", "4"]),
    ("bert4Lrand_crf",  ["--encoder", "bert", "--decoder", "crf", "--bert_layers", "4",
                         "--init", "random"]),
    ("roberta12L_crf",  ["--encoder", "roberta", "--decoder", "crf"]),
    ("roberta4L_crf",   ["--encoder", "roberta", "--decoder", "crf", "--bert_layers", "4"]),
    ("roberta12L_linear", ["--encoder", "roberta", "--decoder", "linear"]),
    ("roberta4L_linear", ["--encoder", "roberta", "--decoder", "linear", "--bert_layers", "4"]),
    ("macbert12L_crf",  ["--encoder", "macbert", "--decoder", "crf"]),
    ("macbert4L_crf",   ["--encoder", "macbert", "--decoder", "crf", "--bert_layers", "4"]),
    ("macbert12L_linear", ["--encoder", "macbert", "--decoder", "linear"]),
    ("macbert4L_linear", ["--encoder", "macbert", "--decoder", "linear", "--bert_layers", "4"]),
]

COMMON = ["--epochs", "4", "--batch_size", "32", "--max_len", "256",
          "--log_every", "50", "--dev_monitor", "0"]


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def deadline_ts(hhmm: str) -> float:
    """HH:MM → 时间戳；若早于当前时刻则视为明天（跨夜任务）。"""
    h, m = [int(x) for x in hhmm.split(":")]
    d = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= datetime.now():
        d = d + timedelta(days=1)
    return d.timestamp()


def run_one(name: str, extra: list, timeout: int, inprocess: bool = False) -> dict:
    if inprocess:
        # 进程内执行：本机实测"新起子进程"有概率掉进 5–7s/步的慢态（同一份代码、同一时刻
        # 同进程内是 0.7s/步），跨配置复用同一个 CUDA 上下文可以绕开。
        # 代价：单配置没有硬超时，靠外部监控 + --skip-done 续跑兜底。
        import traceback
        sys.argv = [str(TRAIN), *extra, *COMMON, "--run_name", name]
        t0 = time.perf_counter()
        rc = 0
        try:
            import train
            train.main()
        except SystemExit as e:
            rc = int(e.code or 0)
        except Exception:
            traceback.print_exc()
            rc = 1
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {"run": name, "rc": rc, "timed_out": False,
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "epochs": 4, "finished_at": now(), "inprocess": True,
                "metric": str((METRICS / f"{name}.json").relative_to(ROOT)),
                "has_metric": (METRICS / f"{name}.json").exists()}

    cmd = [PY, str(TRAIN), *extra, *COMMON, "--run_name", name]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=str(ROOT))
    timed_out = False
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                       capture_output=True)
        try:
            p.wait(timeout=30)
        except Exception:
            pass
        rc = -9
    return {"run": name, "rc": rc, "timed_out": timed_out,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "epochs": 4, "finished_at": now(),
            "metric": str((METRICS / f"{name}.json").relative_to(ROOT)),
            "has_metric": (METRICS / f"{name}.json").exists()}


def save(results: list) -> None:
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=900, help="单配置硬超时（秒）")
    ap.add_argument("--deadline", default="07:00", help="到点不再启动新配置")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--inprocess", action="store_true",
                    help="同一进程内顺序跑（绕开子进程慢态；无单配置硬超时）")
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    dl = deadline_ts(args.deadline)
    results = []

    queue = [(n, a) for n, a in GRID if not only or n in only]
    if args.skip_done:
        queue = [(n, a) for n, a in queue if not (METRICS / f"{n}.json").exists()]
    print(f"[{now()}] 待跑 {len(queue)} 个配置；单配置超时 {args.timeout}s；"
          f"deadline {args.deadline}", flush=True)

    for name, extra in queue:
        if time.time() > dl:
            print(f"[{now()}] 已过 deadline，停止启动新配置", flush=True)
            break
        print(f"\n===== [{now()}] {name} =====", flush=True)
        rec = run_one(name, extra, args.timeout, args.inprocess)
        results.append(rec)
        save(results)
        print(f"[{now()}] {name} rc={rec['rc']} 超时={rec['timed_out']} "
              f"用时 {rec['elapsed_s'] / 60:.1f} min 指标={rec['has_metric']}", flush=True)

    # 失败/超时重试一轮（同一 deadline 内）
    if not args.no_retry:
        failed = [(n, a) for n, a in queue if not (METRICS / f"{n}.json").exists()]
        failed = [x for x in failed if x[0] not in {r["run"] for r in results if r["rc"] == 0}]
        if failed:
            print(f"\n[{now()}] 重试 {len(failed)} 个失败配置：{[n for n, _ in failed]}", flush=True)
        for name, extra in failed:
            if time.time() > dl:
                print(f"[{now()}] 已过 deadline，放弃重试", flush=True)
                break
            print(f"\n===== [{now()}] retry {name} =====", flush=True)
            rec = run_one(name, extra, args.timeout, args.inprocess)
            rec["retry"] = True
            results.append(rec)
            save(results)
            print(f"[{now()}] retry {name} rc={rec['rc']} 指标={rec['has_metric']}", flush=True)

    ok = [r["run"] for r in results if (METRICS / f"{r['run']}.json").exists()]
    print(f"\n[{now()}] 完成 {len(ok)}/{len(queue)}：{ok}", flush=True)
    print(f"缺：{[n for n, _ in queue if n not in ok]}", flush=True)
    print(f"结果 → {RESULT}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""run_night_chain.py — B 组夜间串行链（本地 LLM 提示路线 + LoRA 微调）

顺序 = 报告价值排序，前面的先落地：
  1. Qwen3-0.6B few-shot        （补完今晚被时间截断的那一格，支持断点续跑）
  2. Qwen3-0.6B LoRA SFT 训练   （用户明确要求的"本地微调"，zero-shot 0.087 的对照）
  3. LoRA 适配器评测            （走同一条 n200 口径）
  4. Qwen2-0.5B-Instruct zero   （同量级旧一代）
  5. Qwen3-1.7B zero            （参数量斜率）
  6. Qwen3-4B zero，device_map=auto（8GB 显存上限，慢则超时放弃）

工程纪律：每个任务独立子进程 + 硬超时 + 失败继续 + 结果逐个落盘；
--deadline 到点不再启动新任务。任何时候杀掉整个链都不丢已完成任务的产物。
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"D:\Programs\Dev\Miniconda3\envs\py312\python.exe"
EXTRACT = ROOT / "llm" / "extract_local.py"
SFT = ROOT / "llm" / "sft_lora.py"
M = ROOT / "outputs" / "metrics" / "disc"
STATUS = ROOT / "outputs" / "logs" / "night_chain_results.json"

TASKS = [
    dict(name="local_Qwen3-0.6B_few", timeout=5400,
         cmd=[PY, str(EXTRACT), "--model", "Qwen3-0.6B", "--device_map", "cuda",
              "--few_shot", "--run_name", "local_Qwen3-0.6B_few"],
         out=str(M / "local_Qwen3-0.6B_few.json")),
    # 1 epoch：本机实测 0.6B SFT 约 750–1000 tok/s（小 batch + checkpointing），
    # 全量 3548 句 1 轮约 45 min；2 轮会把后面三个模型的时间挤掉，收益边际递减。
    # bs1 + max_len 1024：max_len 1536 时全词表 logits（2×1536×151936）会把 8GB 显存吃满，
    # WDDM 一分页步速从 2s 掉到 16s/微批（实测）。1024 覆盖 p95=927 的样本，代价是 8 条超长样本截断。
    dict(name="sft_qwen3_0.6b", timeout=4200,
         cmd=[PY, str(SFT), "--epochs", "1", "--batch_size", "1", "--grad_accum", "8",
              "--max_len", "1024"],
         out=str(ROOT / "outputs" / "sft" / "qwen3_0.6b_lora" / "adapter_model.safetensors")),
    dict(name="local_Qwen3-0.6B_lora", timeout=4200,
         cmd=[PY, str(EXTRACT), "--model", "Qwen3-0.6B", "--device_map", "cuda",
              "--adapter", str(ROOT / "outputs" / "sft" / "qwen3_0.6b_lora"),
              "--run_name", "local_Qwen3-0.6B_lora"],
         out=str(M / "local_Qwen3-0.6B_lora.json")),
    dict(name="local_Qwen2-0.5B-Instruct_zero", timeout=3600,
         cmd=[PY, str(EXTRACT), "--model", "Qwen2-0.5B-Instruct", "--device_map", "cuda",
              "--run_name", "local_Qwen2-0.5B-Instruct_zero"],
         out=str(M / "local_Qwen2-0.5B-Instruct_zero.json")),
    dict(name="local_Qwen3-1.7B_zero", timeout=5400,
         cmd=[PY, str(EXTRACT), "--model", "Qwen3-1.7B", "--device_map", "cuda",
              "--gen_batch", "4", "--run_name", "local_Qwen3-1.7B_zero"],
         out=str(M / "local_Qwen3-1.7B_zero.json")),
    dict(name="local_Qwen3-4B_zero", timeout=5400,
         cmd=[PY, str(EXTRACT), "--model", "Qwen3-4B", "--device_map", "auto",
              "--gen_batch", "4", "--max_new_tokens", "800",
              "--run_name", "local_Qwen3-4B_zero"],
         out=str(M / "local_Qwen3-4B_zero.json")),
]


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def deadline_ts(hhmm: str) -> float:
    h, m = [int(x) for x in hhmm.split(":")]
    d = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= datetime.now():
        d = d + timedelta(days=1)
    return d.timestamp()


def save(results: list) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="07:00")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    dl = deadline_ts(args.deadline)
    tasks = [t for t in TASKS if not only or t["name"] in only]

    results = []
    for t in tasks:
        if Path(t["out"]).exists() and not only:
            print(f"[{now()}] [skip] {t['name']} 产物已存在", flush=True)
            results.append({"name": t["name"], "skipped": True})
            save(results)
            continue
        if time.time() > dl:
            print(f"[{now()}] 已过 deadline，停止", flush=True)
            break
        print(f"\n===== [{now()}] {t['name']} =====", flush=True)
        t0 = time.perf_counter()
        p = subprocess.Popen(t["cmd"], cwd=str(ROOT))
        timed_out = False
        try:
            rc = p.wait(timeout=t["timeout"])
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            try:
                p.wait(timeout=30)
            except Exception:
                pass
            rc = -9
        rec = {"name": t["name"], "rc": rc, "timed_out": timed_out,
               "elapsed_s": round(time.perf_counter() - t0, 1),
               "finished_at": now(), "has_out": Path(t["out"]).exists()}
        results.append(rec)
        save(results)
        print(f"[{now()}] {t['name']} rc={rc} 超时={timed_out} "
              f"用时 {rec['elapsed_s']/60:.1f} min 产物={rec['has_out']}", flush=True)

    print(f"\n[{now()}] 链结束", flush=True)
    for r in results:
        print(f"  {r['name']:<34} rc={r.get('rc')} {r.get('elapsed_s', 0)/60:.1f} min "
              f"产物={r.get('has_out', 'skip')}", flush=True)


if __name__ == "__main__":
    main()

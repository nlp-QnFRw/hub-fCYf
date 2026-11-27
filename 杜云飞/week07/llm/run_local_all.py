# -*- coding: utf-8 -*-
"""run_local_all.py — B 组本地离线大模型顺序执行驱动。

4 个本地模型 zero-shot 逐个跑（Qwen3-0.6B → Qwen2-0.5B-Instruct → Qwen3-1.7B → Qwen3-4B），
每个模型独立子进程（一个挂了不影响后面的），输出追加到 outputs/logs/local_all.log。
4B 走 device_map=auto（bf16 8GB 放不下，部分层卸到 32GB 内存，速度慢属预期）。
支持 --skip-done：已有指标 JSON 的跳过。
用法：python llm/run_local_all.py [--skip-done] [--few-shot]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"D:\Programs\Dev\Miniconda3\envs\py312\python.exe"
EXTRACT = ROOT / "llm" / "extract_local.py"
LOG = ROOT / "outputs" / "logs" / "local_all.log"
METRICS = ROOT / "outputs" / "metrics" / "disc"

MODELS = [
    ("Qwen3-0.6B", ["--device_map", "cuda"]),
    ("Qwen2-0.5B-Instruct", ["--device_map", "cuda"]),
    ("Qwen3-1.7B", ["--device_map", "cuda"]),
    ("Qwen3-4B", ["--device_map", "auto"]),
]


def main():
    skip_done = "--skip-done" in sys.argv
    few = "--few-shot" in sys.argv
    LOG.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for name, dm in MODELS:
        suffix = "few" if few else "zero"
        run_name = f"local_{name}_{suffix}"
        if skip_done and (METRICS / f"{run_name}.json").exists():
            print(f"[skip] {run_name} 指标已存在", flush=True)
            continue
        cmd = [PY, str(EXTRACT), "--model", name, *dm, "--run_name", run_name]
        if few:
            cmd.append("--few_shot")
        print(f"\n===== [{time.strftime('%H:%M:%S')}] {run_name} =====", flush=True)
        print(" ".join(cmd), flush=True)
        t0 = time.perf_counter()
        with open(LOG, "a", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
        el = time.perf_counter() - t0
        print(f"[{'OK' if rc == 0 else 'FAIL'}] {run_name} 用时 {el / 60:.1f} min", flush=True)
        results.append({"run": run_name, "rc": rc, "elapsed_s": round(el, 1)})
        if rc != 0:
            print("继续下一个模型（独立进程互不干扰）", flush=True)
    out = ROOT / "outputs" / "logs" / "local_all_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n汇总 → {out}")
    for r in results:
        print(f"  {r['run']:<40} rc={r['rc']}  {r['elapsed_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()

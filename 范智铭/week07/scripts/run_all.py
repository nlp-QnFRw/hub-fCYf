"""端到端跑完所有实验（可断点续跑：已存在的产物会跳过）。

    python scripts/run_all.py --steps prepare bilstm bert llm compare
    python scripts/run_all.py --steps llm --datasets cluener      # 只补某一格

每一步都是独立子进程，失败会停下并打印命令，方便单独重跑。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 每个数据集的训练预算不同：简历样本少、句子长；CLUENER/MSRA 句子短、样本多
TRAIN_CFG = {
    "resume": {"bilstm": ["--epochs", "20", "--patience", "5", "--max_len", "512"],
               "bert": ["--epochs", "6", "--max_len", "512", "--batch", "16", "--lr", "3e-5"]},
    "cluener": {"bilstm": ["--epochs", "15", "--patience", "4", "--max_len", "160"],
                "bert": ["--epochs", "5", "--max_len", "128", "--batch", "32", "--lr", "2e-5"]},
    "msra": {"bilstm": ["--epochs", "12", "--patience", "4", "--max_len", "256"],
             "bert": ["--epochs", "4", "--max_len", "256", "--batch", "16", "--lr", "2e-5"]},
}
LLM_N = {"resume": 0, "cluener": 300, "msra": 300}


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    print(f"  -> exit={r.returncode}  {time.time()-t0:.0f}s", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"命令失败: {' '.join(cmd)}")


def exists(*parts: str) -> bool:
    return (ROOT.joinpath(*parts)).exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["prepare", "bilstm", "bert", "llm", "compare"])
    ap.add_argument("--datasets", nargs="+", default=["resume", "cluener", "msra"])
    ap.add_argument("--force", action="store_true", help="已有产物也重跑")
    args = ap.parse_args()

    if "prepare" in args.steps:
        run([PY, "scripts/download_data.py"])
        run([PY, "scripts/prepare_data.py", "--datasets", *args.datasets])
        run([PY, "scripts/selftest.py"])

    if "models" in args.steps:
        run([PY, "scripts/download_models.py"])

    if "bilstm" in args.steps:
        for ds in args.datasets:
            if not args.force and exists("outputs", ds, "bilstm", "metrics.json"):
                print(f"[skip] bilstm/{ds}")
                continue
            run([PY, "-m", "src.train_bilstm", "--dataset", ds, *TRAIN_CFG[ds]["bilstm"]])

    if "bert" in args.steps:
        for ds in args.datasets:
            for model in ("bert", "roberta"):
                if not args.force and exists("outputs", ds, model, "metrics.json"):
                    print(f"[skip] {model}/{ds}")
                    continue
                run([PY, "-m", "src.train_transformer", "--dataset", ds, "--model", model,
                     *TRAIN_CFG[ds]["bert"]])

    if "llm" in args.steps:
        for ds in args.datasets:
            for mode in ("zero", "few"):
                tag = f"{mode}_n{LLM_N[ds] or 'all'}"
                if not args.force and exists("outputs", ds, "llm", f"metrics_{tag}.json"):
                    print(f"[skip] llm_{mode}/{ds}")
                    continue
                cmd = [PY, "-m", "src.llm_extract", "--dataset", ds, "--mode", mode,
                       "--n", str(LLM_N[ds])]
                run(cmd)

    if "compare" in args.steps:
        run([PY, "-m", "src.compare"])


if __name__ == "__main__":
    main()

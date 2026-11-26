"""一键复现 work 实验（可单步）。统一使用 py312 解释器：sys.executable 就是启动本脚本的 python。

    python scripts/run_all.py                  # 全流程（GPU 训练约 20 分钟）
    python scripts/run_all.py --step rule      # 只跑规则基线
    python scripts/run_all.py --step compare   # 只重算汇总表

API 一路需要环境变量 LLM_API_KEY；没给就记 NOT_RUN，不阻塞其它步骤。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = {
    "data": ["data_prep/build_variants.py"],
    "rule": ["methods/rule_baseline.py --variant sent",
             "methods/rule_baseline.py --variant heldout",
             "methods/rule_baseline.py --variant doc"],
    "disc": ["methods/train_bio.py --model rbt3 --variant sent --epochs 2 --batch_size 16",
             "methods/train_bio.py --model macbert --variant sent --epochs 2 --batch_size 16",
             "methods/train_bio.py --model rbt3 --variant heldout --epochs 2 --batch_size 16",
             "methods/train_bio.py --model macbert --variant heldout --epochs 2 --batch_size 16",
             "methods/train_bio.py --model rbt3 --variant doc --predict_only --src_tag rbt3_crf_sent --tag rbt3_crf_doc",
             "methods/train_bio.py --model macbert --variant doc --predict_only --src_tag macbert_crf_sent --tag macbert_crf_doc"],
    "llm": ["methods/qwen_zero.py --variant sent --mode zero --n 60 --max_length 2048",
            "methods/qwen_zero.py --variant sent --mode few --n 60 --max_length 2048",
            "methods/qwen_zero.py --variant heldout --mode zero --n 60 --max_length 2048",
            "methods/qwen_zero.py --variant doc --mode zero --n 60 --max_length 2048",
            "methods/api_zero.py --variant sent --mode zero --n 60",
            "methods/api_zero.py --variant heldout --mode zero --n 60"],
    "extra": ["methods/doc_gate.py --src macbert_crf_doc"],
    "compare": ["eval/compare.py"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS), default="")
    args = ap.parse_args()
    steps = [args.step] if args.step else list(STEPS)
    for step in steps:
        for cmd in STEPS[step]:
            t0 = time.time()
            print(f"\n===== [{step}] {cmd} =====", flush=True)
            r = subprocess.run([PY, *cmd.split()], cwd=ROOT)
            if r.returncode != 0:
                print(f"[FAILED] {cmd} (exit={r.returncode})", file=sys.stderr)
                return r.returncode
            print(f"[done] {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

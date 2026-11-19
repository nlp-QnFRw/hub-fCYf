"""
一键复现：按依赖顺序跑完所有实验

  python scripts/run_all.py            # 全流程（约 60~90 分钟，取决于显卡）
  python scripts/run_all.py --fast     # 跳过零样本与数据量扫描（约 25 分钟）
  python scripts/run_all.py --only bert
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STAGES = {
    "rule": [["rule/classify_rule.py"]],
    "tfidf": [["tfidf/classify_tfidf.py"]],
    "textcnn": [["textcnn/train_textcnn.py", "--epochs", "4"]],
    "macbert": [["bert/train_bert.py", "--model", "macbert", "--epochs", "3"]],
    "rbt3": [["bert/train_bert.py", "--model", "rbt3", "--epochs", "3"]],
    "zero_shot": [["zero_shot/classify_zero_shot.py", "--n", "8000"]],
    "scan": [["eval/scan_scale.py", "--model", "rbt3"]],
    "summary": [["eval/eval_all.py"], ["eval/make_report.py"]],
}
ORDER = ["rule", "tfidf", "textcnn", "rbt3", "macbert", "zero_shot", "scan", "summary"]
FAST = {"rule", "tfidf", "textcnn", "rbt3", "macbert", "summary"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", default=None, help="只跑某一阶段，逗号分隔")
    args = ap.parse_args()

    todo = [s.strip() for s in args.only.split(",")] if args.only else (
        [s for s in ORDER if s in FAST] if args.fast else ORDER)

    t_all = time.time()
    for name in todo:
        for cmd in STAGES[name]:
            full = [PY, str(ROOT / cmd[0])] + cmd[1:]
            print(f"\n{'='*70}\n>>> {name}: {' '.join(cmd)}\n{'='*70}", flush=True)
            t0 = time.time()
            r = subprocess.run(full, cwd=str(ROOT))
            if r.returncode != 0:
                print(f"[fail] {name} 退出码 {r.returncode}，中止", flush=True)
                return
            print(f"[ok] {name} 用时 {time.time()-t0:.0f}s", flush=True)
    print(f"\n全部完成，总用时 {(time.time()-t_all)/60:.1f} 分钟")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# 一键复现：下模型 → 训 BERT 基线 → 训 RoBERTa → 出对比报告
# 用法：bash run_all.sh          （两个基座都用 config.py 里的同一套超参）
#       bash run_all.sh --smoke  （冒烟：2000 条 + 20 步，30 秒验证代码通路）
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
# 8 个物理核；不设的话 torch 会用 16 个逻辑核，实测反而更慢
THREADS="${THREADS:-8}"
# 国内直连 huggingface.co 常超时；两步下载都走镜像（config.py 里也会兜底设一次）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# macOS 上 ~/.matplotlib 可能不可写，导致字体缓存建不起来、import 变慢
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.cache/mpl}"
mkdir -p "$MPLCONFIGDIR"

SMOKE=""
TAG=""
if [ "${1:-}" = "--smoke" ]; then SMOKE="--smoke"; TAG="--tag smoke"; fi

echo "════════ 1/3  BERT 基线 ════════"
"$PY" src/train.py --backbone bert-base-chinese --threads "$THREADS" $SMOKE

echo "════════ 2/3  RoBERTa ════════"
"$PY" src/train.py --backbone chinese-roberta-wwm-ext --threads "$THREADS" $SMOKE

echo "════════ 3/3  对比分析 ════════"
"$PY" src/compare.py $TAG

echo
echo "完成。报告：outputs/comparison_report.md    图：outputs/figures/"

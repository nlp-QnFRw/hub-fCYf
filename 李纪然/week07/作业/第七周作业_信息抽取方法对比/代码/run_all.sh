#!/usr/bin/env bash
# 本周作业全部实验的一键复现脚本（与 附录/运行记录.md 一一对应）
#
# 前置：
#   1) 课程项目 week7-信息抽取/code 已解压（data/processed 随包发）
#   2) python 3.12 + torch 2.6 + transformers 5.5.3（requirements.txt）
#   3) 本地 bert-base-chinese 权重（用 BERT_PATH 指定，不联网下载）
#   4) API 链路需要 LLM_API_KEY（OpenAI 兼容接口，默认 https://api.deepseek.com）
#
# 用法：
#   bash 代码/run_all.sh                # 全量（本次在本机约 1.5~2 小时）
#   API_ONLY=1 bash 代码/run_all.sh     # 只跑 LLM/API 对比（约 15 分钟）
#   SKIP_LLM=1 bash 代码/run_all.sh     # 只跑判别式方法
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SUB_ROOT="$(dirname "$HERE")"
PROJECT="${PROJECT:-$SUB_ROOT/../week7-信息抽取/code}"
PY="${PY:-python}"
BERT_PATH="${BERT_PATH:-$HOME/Downloads/第四周 语言模型/bert-base-chinese}"
DEVICE="${DEVICE:-cpu}"          # 本机实测：BERT 跑 CPU 比 MPS 快；BiLSTM 用 MPS 快
cd "$PROJECT"

echo "== 项目目录：$PROJECT"
echo "== python   ：$($PY -V 2>&1)"
echo "== bert 权重 ：$BERT_PATH"

# ── 0) 数据集变体 ───────────────────────────────────────────────────────────
[ -f "$HERE/data_holdout/train_extract.jsonl" ] || \
  $PY "$HERE/build_holdout.py" --project "$PROJECT" --out "$HERE/data_holdout"
[ -f "$HERE/data_frac25/train_extract.jsonl" ] || \
  $PY "$HERE/build_frac25.py" --project "$PROJECT" --out "$HERE/data_frac25"

# ── 1) L0 规则基线（秒级，纯 CPU）───────────────────────────────────────────
if [ -z "${API_ONLY:-}" ]; then
  $PY rule/run_rule.py --name hw_rule_full
  $PY rule/run_rule.py --data_dir data/sample            --name hw_rule_sample
  $PY rule/run_rule.py --data_dir "$HERE/data_holdout"    --name hw_rule_holdout
  $PY rule/run_rule.py --data_dir "$HERE/data_frac25"     --name hw_rule_frac25
fi

# ── 2) L1 BiLSTM-CRF（无预训练；MPS 上很快）─────────────────────────────────
if [ -z "${API_ONLY:-}" ]; then
  $PY bilstm_crf/train.py --device "${BILSTM_DEVICE:-mps}" --epochs 6 --max_len 256 \
      --run_name hw_bilstm_crf
  $PY bilstm_crf/train.py --device "${BILSTM_DEVICE:-mps}" --epochs 6 --max_len 256 \
      --data_dir data/sample        --run_name hw_bilstm_crf_sample
  $PY bilstm_crf/train.py --device "${BILSTM_DEVICE:-mps}" --epochs 6 --max_len 256 \
      --data_dir "$HERE/data_frac25" --run_name hw_bilstm_crf_frac25
fi

# ── 3) L2 BERT+Linear（CPU；预算受限，见报告第 7 节）────────────────────────
if [ -z "${API_ONLY:-}" ]; then
  $PY bert/train.py --decoder linear --device "$DEVICE" --epochs 3 --max_len 128 \
      --bert_path "$BERT_PATH" --run_name hw_bert_linear
  $PY bert/train.py --decoder linear --device "$DEVICE" --epochs 3 --max_len 128 \
      --data_dir data/sample --bert_path "$BERT_PATH" --run_name hw_bert_linear_sample
  # 预算对照：1 epoch @256（约 28 分钟，可选）
  # $PY bert/train.py --decoder linear --device "$DEVICE" --epochs 1 --max_len 256 \
  #     --bert_path "$BERT_PATH" --run_name hw_bert_linear_e1_256
fi

# ── 4) L3 LLM：换 API 模型 / zero vs few-shot / 换数据集 ────────────────────
if [ -z "${SKIP_LLM:-}" ]; then
  : "${LLM_API_KEY:?需要先 export LLM_API_KEY=sk-xxx}"
  $PY llm_zero_shot/llm_extract.py --backend api --mode zero --n 200 \
      --model deepseek-flash --tag hw_api_flash_zero_n200
  $PY llm_zero_shot/llm_extract.py --backend api --mode few  --n 200 \
      --model deepseek-flash --tag hw_api_flash_few_n200
  $PY llm_zero_shot/llm_extract.py --backend api --mode zero --n 200 \
      --model deepseek-v4-pro --tag hw_api_pro_zero_n200
  $PY llm_zero_shot/llm_extract.py --backend api --mode zero --n 116 \
      --model deepseek-flash --data "$HERE/data_holdout/dev_extract.jsonl" \
      --tag hw_api_flash_zero_holdout_n116
fi

# ── 5) 汇总 / 切片分析 / 出图 / 收拢产物 ──────────────────────────────────
$PY "$HERE/summarize.py" --project "$PROJECT"
$PY "$HERE/analyze_slices.py" --project "$PROJECT"
MPLCONFIGDIR=/tmp/mpl-week7 $PY "$HERE/make_figures.py" --project "$PROJECT"
bash "$HERE/collect_results.sh"
echo "全部完成 → $SUB_ROOT/报告/实验报告.md"

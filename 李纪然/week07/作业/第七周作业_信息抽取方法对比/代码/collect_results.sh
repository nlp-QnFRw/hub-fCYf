#!/usr/bin/env bash
# 把课程项目 outputs/ 里本次作业的产物（hw_ 前缀）收拢到提交目录，保证"报告里的数字可反查"。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB_ROOT="$(dirname "$HERE")"
PROJECT="${PROJECT:-$SUB_ROOT/../week7-信息抽取/code}"

mkdir -p "$SUB_ROOT/结果/metrics" "$SUB_ROOT/结果/predictions" "$SUB_ROOT/结果/reports" "$SUB_ROOT/附录/logs"

cp "$PROJECT"/outputs/metrics/disc/hw_*.json       "$SUB_ROOT/结果/metrics/" 2>/dev/null || true
cp "$PROJECT"/outputs/metrics/llm/hw_*.json        "$SUB_ROOT/结果/metrics/" 2>/dev/null || true
cp "$PROJECT"/outputs/disc/pred_hw_*.jsonl         "$SUB_ROOT/结果/predictions/" 2>/dev/null || true
cp "$PROJECT"/outputs/llm/pred_hw_*.jsonl          "$SUB_ROOT/结果/predictions/" 2>/dev/null || true
cp "$PROJECT"/outputs/llm/report_hw_*.json         "$SUB_ROOT/结果/reports/" 2>/dev/null || true
cp "$PROJECT"/outputs/logs/hw_*_train.json         "$SUB_ROOT/附录/logs/" 2>/dev/null || true
cp "$PROJECT"/outputs/logs/hw_*_steps.jsonl        "$SUB_ROOT/附录/logs/" 2>/dev/null || true
cp "$PROJECT"/outputs/logs/hw_*_eval.json          "$SUB_ROOT/附录/logs/" 2>/dev/null || true

echo "已收集："
find "$SUB_ROOT/结果" "$SUB_ROOT/附录/logs" -type f | sed "s|$SUB_ROOT/||" | sort

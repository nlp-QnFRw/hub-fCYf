#!/usr/bin/env bash
# 手动下载两个基座（自动脚本失败时的兜底）
# 用法：bash models/download_models.sh
# 说明：hf-mirror 偶发截断，脚本用 -C - 断点续传，中断后重跑同一条命令即可
set -euo pipefail
cd "$(dirname "$0")"

W="$(pwd)"
HF="${HF_ENDPOINT:-https://hf-mirror.com}"

fetch() {  # repo_dir  file  [alt_url]
  local dir="$1" f="$2" alt="${3:-}"
  mkdir -p "$dir"
  local url="$HF/$dir/resolve/main/$f"
  if [ -f "$dir/$f" ] && [ -s "$dir/$f" ]; then
    echo "  = $dir/$f 已存在"
    return 0
  fi
  curl -sL -C - --max-time 1800 --retry 3 --retry-all-errors -o "$dir/$f" "$url" \
    || { [ -n "$alt" ] && curl -sL --max-time 1800 -o "$dir/$f" "$alt"; }
  if [ -s "$dir/$f" ]; then echo "  ✓ $dir/$f ($(du -h "$dir/$f" | cut -f1))"; else
    echo "  ✗ $dir/$f 下载失败"; rm -f "$dir/$f"; return 1; fi
}

echo "→ bert-base-chinese"
for f in config.json vocab.txt tokenizer_config.json tokenizer.json; do fetch bert-base-chinese "$f"; done
fetch bert-base-chinese pytorch_model.bin

echo "→ chinese-roberta-wwm-ext（HF 镜像慢时自动改用 ModelScope）"
for f in config.json vocab.txt tokenizer_config.json tokenizer.json special_tokens_map.json; do fetch chinese-roberta-wwm-ext "$f"; done
fetch chinese-roberta-wwm-ext pytorch_model.bin \
  "https://www.modelscope.cn/models/dienstag/chinese-roberta-wwm-ext/resolve/master/pytorch_model.bin"

echo
echo "校验：两个权重都应约 411,577,000 字节"
ls -l */pytorch_model.bin models/*/pytorch_model.bin 2>/dev/null || ls -l */pytorch_model.bin
echo "首次训练时 src/config.py 会自动把 .bin 转成 .safetensors（torch<2.6 必需）"

# 预训练基座下载说明

`src/config.py` 的 `ensure_backbone()` 会在缺文件时自动下载，正常情况你**不需要**手动操作。
只有在下载被墙/中断时才按下面步骤手动补。

需要的两个目录（与 `config.py` 的 `BACKBONES` 对应）：

| 本地目录 | HuggingFace repo | 大小 |
|---|---|---|
| `models/bert-base-chinese/` | `bert-base-chinese` | ~412 MB |
| `models/chinese-roberta-wwm-ext/` | `hfl/chinese-roberta-wwm-ext` | ~412 MB |

每个目录至少要这四个文件：`config.json`、`vocab.txt`、`tokenizer_config.json`、`pytorch_model.bin`
（`chinese-roberta-wwm-ext` 还需要 `special_tokens_map.json`、`tokenizer.json`；`added_tokens.json` 有无皆可）。

## 方式一：镜像 + huggingface_hub（推荐）

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from src.config import ensure_backbone as e; print(e('bert-base-chinese')); print(e('chinese-roberta-wwm-ext'))"
```

## 方式二：curl 手动下（镜像偶发截断时用断点续传）

```bash
W=$(pwd)/models
mkdir -p "$W/bert-base-chinese" "$W/chinese-roberta-wwm-ext"
B=https://hf-mirror.com

# 小文件
for f in config.json vocab.txt tokenizer_config.json tokenizer.json; do
  curl -sL -o "$W/bert-base-chinese/$f"     "$B/bert-base-chinese/resolve/main/$f"
  curl -sL -o "$W/chinese-roberta-wwm-ext/$f" "$B/hfl/chinese-roberta-wwm-ext/resolve/main/$f"
done
curl -sL -o "$W/chinese-roberta-wwm-ext/special_tokens_map.json" \
     "$B/hfl/chinese-roberta-wwm-ext/resolve/main/special_tokens_map.json"

# 权重：-C - 断点续传，中断后重跑同一条命令即可
curl -L -C - -o "$W/bert-base-chinese/pytorch_model.bin" \
     "$B/bert-base-chinese/resolve/main/pytorch_model.bin"
curl -L -C - -o "$W/chinese-roberta-wwm-ext/pytorch_model.bin" \
     "$B/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin"
```

校验大小（必须对得上，否则 transformers 会在加载时抛 `UnpicklingError`）：

```bash
ls -l models/*/pytorch_model.bin     # 两个文件都应约 411,578,000 字节
```

## 方式三：离线拷贝

把课程材料里的 `pretrain_models/bert-base-chinese/` 整个目录拷到 `models/` 下即可
（目录名必须是 `bert-base-chinese`，代码按这个名字找）。

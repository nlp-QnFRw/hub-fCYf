---
title: 文本分类脚本定位本地 BERT 模型
domain: project
tags: [文本分类, BERT, HuggingFace, 本地模型路径]
code_refs: [BERT_PATH, src/train.py, src/evaluate.py, src/predict.py, src/compare_class_weight.py, src/explore_data.py]
updated: 2026-10-20
confidence: high
source: codex-session
---

# 文本分类脚本定位本地 BERT 模型

## 一句话结论
> BERT 模型位于项目根目录的 `pretrain_models/bert-base-chinese`；脚本应从自身文件位置推导项目根目录，避免依赖运行命令时的当前目录。

## 背景 / 它解决什么问题
项目把 BERT 配置、词表和权重保存在 `pretrain_models/bert-base-chinese/`。`from_pretrained` 收到不存在的路径时，会尝试按 Hugging Face Hub 仓库名解析；绝对本地路径就会触发 `HFValidationError`。

## 关键代码位置
| 环节 | 文件 | 路径 |
|---|---|---|
| 训练 | `src/train.py` | `ROOT / "pretrain_models" / "bert-base-chinese"` |
| 评估 | `src/evaluate.py` | `ROOT / "pretrain_models" / "bert-base-chinese"` |
| 推理 | `src/predict.py` | `ROOT / "pretrain_models" / "bert-base-chinese"` |
| 类别权重对比 | `src/compare_class_weight.py` | `ROOT / "pretrain_models" / "bert-base-chinese"` |
| 数据探索 | `src/explore_data.py` | `ROOT / "pretrain_models" / "bert-base-chinese"` |

## 关键约束
- `ROOT` 使用 `Path(__file__).resolve().parent.parent`，即项目根目录。
- 本地模型目录应包含 `config.json`、`vocab.txt` 以及 `model.safetensors` 或 `pytorch_model.bin`。
- 若手动传入 `--bert_path`，必须指向该模型目录；路径错误会被 Transformers 当作 Hub repo id 处理。

## 常见坑 / 排障要点
- 报错路径落在 `xiaoyuqing-last-term/pretrain_models/`：默认路径向上级目录跳多了；改用项目根目录下的 `pretrain_models/bert-base-chinese`。
- 路径确认后仍报错：检查该目录是否存在，并确认配置、词表和权重文件齐全。

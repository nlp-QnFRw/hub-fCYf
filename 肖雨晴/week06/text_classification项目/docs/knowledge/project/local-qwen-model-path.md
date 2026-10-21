---
title: Zero-shot 分类脚本定位本地 Qwen 模型
domain: project
tags: [文本分类, Qwen2-0.5B-Instruct, HuggingFace, 本地模型路径]
code_refs: [src_llm/classify_llm.py, MODEL_PATH, load_model]
updated: 2026-10-20
confidence: high
source: codex-session
---

# Zero-shot 分类脚本定位本地 Qwen 模型

## 一句话结论
> `classify_llm.py` 的默认模型目录是项目根目录下的 `pretrain_models/Qwen2-0.5B-Instruct`。

## 背景 / 它解决什么问题
`classify_llm.py` 用 `AutoTokenizer.from_pretrained` 和 `AutoModelForCausalLM.from_pretrained` 加载本地 Qwen 模型。旧写法从项目根目录再向上跳两级，导致模型目录不存在。

## 关键代码位置
| 环节 | 文件 | 路径 |
|---|---|---|
| 项目根目录 | `src_llm/classify_llm.py` | `Path(__file__).resolve().parent.parent` |
| 默认模型目录 | `src_llm/classify_llm.py` | `ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"` |
| 模型加载 | `src_llm/classify_llm.py` 的 `load_model` | `AutoTokenizer`、`AutoModelForCausalLM` |

## 关键约束
- 模型目录包含 `config.json`、`tokenizer.json` 和 `model.safetensors`。
- 显式传入 `--model_path` 时，该参数会覆盖默认目录。

## 常见坑 / 排障要点
- 错误路径落在 `xiaoyuqing-last-term/pretrain_models/`：检查是否误用 `ROOT.parent.parent`；应使用 `ROOT / "pretrain_models"`。

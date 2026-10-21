---
title: SFT 训练与评估脚本的本地 Qwen 模型路径
domain: project
tags: [SFT, Qwen2-0.5B-Instruct, HuggingFace, 本地模型路径, HFValidationError]
code_refs: [src_llm/train_sft.py, src_llm/evaluate_sft.py, MODEL_PATH]
updated: 2026-10-20
confidence: high
source: codex-session
---

# SFT 训练与评估脚本的本地 Qwen 模型路径

## 一句话结论
> `train_sft.py` 与 `evaluate_sft.py` 的默认 Qwen 模型路径应为项目根目录下的 `pretrain_models/Qwen2-0.5B-Instruct`。

## 背景 / 它解决什么问题
旧写法 `ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"` 会落到 `xiaoyuqing-last-term/pretrain_models/`，该目录不存在。本地模型实际位于 `week6/text_classification项目/pretrain_models/Qwen2-0.5B-Instruct`。传入不存在的绝对本地目录后，Hugging Face 加载器可能按 Hub 仓库 ID 校验并报 `HFValidationError: Repo id must be in the form ...`；这里无需添加 `repo_type`。

## 关键代码位置
| 环节 | 文件 | 内容 |
|---|---|---|
| SFT 训练默认模型 | `src_llm/train_sft.py` | `ROOT = Path(__file__).resolve().parent.parent`，`MODEL_PATH = ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"` |
| SFT 评估默认模型 | `src_llm/evaluate_sft.py` | 同样使用脚本位置解析项目根目录和默认模型目录 |
| 模型加载 | 上述两个脚本 | `AutoTokenizer.from_pretrained` / `AutoModelForCausalLM.from_pretrained` |

## 常见坑 / 排障要点
- 报错中的绝对路径若落在 `xiaoyuqing-last-term/pretrain_models/`，先核对 `MODEL_PATH` 是否多跳了两级。
- `--model_path` 可覆盖默认路径，需指向含 `config.json`、tokenizer 文件和模型权重的本地目录。
- 2026-10-20 再次收到相同错误时，复查发现 `train_sft.py` 的路径常量已重新变成旧写法，而 `evaluate_sft.py` 仍为正确写法；文件为何被改回尚未确认。重新修正训练脚本后，增加了绝对本地模型路径的 `config.json` 检查，报错时会显示当前脚本路径，便于区分旧代码、运行参数覆盖和其他副本。
- 此结论基于代码与文件目录检查；本次没有启动训练或评估。

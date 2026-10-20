---
title: Week 4 tokenizer 加载本地 BERT 模型
domain: project
tags: [tokenizer, BertTokenizer, bert-base-chinese, 本地模型路径]
code_refs: [tokenizer.py, MODEL_PATH]
updated: 2026-10-20
confidence: high
source: codex-session
---

# Week 4 tokenizer 加载本地 BERT 模型

## 一句话结论
> `week4/tokenizer.py` 根据脚本位置加载 `week6/text_classification项目/pretrain_models/bert-base-chinese`，不再依赖 Windows 绝对路径或当前工作目录。

## 背景 / 它解决什么问题
原脚本把模型目录写死为 `E:\\pretrain_models\\bert-base-chinese`，在当前 macOS 环境中不存在。仓库内实际模型位于 `week6/text_classification项目/pretrain_models/bert-base-chinese`。

## 关键代码位置
| 环节 | 文件 | 内容 |
|---|---|---|
| 路径定位 | `tokenizer.py` | `MODEL_PATH` 从 `Path(__file__).resolve()` 计算仓库内模型目录 |
| tokenizer 加载 | `tokenizer.py` | `BertTokenizer.from_pretrained(str(MODEL_PATH))` |

## 常见坑 / 排障要点
- 不要使用 `E:\\...` 等其他机器的绝对路径。
- 从不同工作目录启动脚本不会影响基于 `__file__` 计算的模型路径。
- 模型目录需要包含 `config.json`、词表和模型权重。

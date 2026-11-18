---
title: Qwen 文本分类的固定 few-shot 提示词模式
domain: project
tags: [文本分类, Qwen2-0.5B-Instruct, few-shot, zero-shot, 提示词]
code_refs: [src_llm/classify_llm.py, classify_one, FEW_SHOT_SYSTEM_PROMPT]
updated: 2026-10-20
confidence: high
source: codex-session
---

# Qwen 文本分类的固定 few-shot 提示词模式

## 一句话结论
> `classify_llm.py` 默认使用原有 zero-shot `SYSTEM_PROMPT`；传入 `--few_shot` 时，在 system 消息中追加 15 个类别各两条固定训练集标题示例。

## 背景 / 它解决什么问题
项目通过 Qwen2-0.5B-Instruct 直接分类 TNEWS 标题。固定训练集示例避免使用验证集标签作为提示词内容，并让相同 `--seed`、`--num_samples` 的两种模式评估同一批验证集样本。

## 关键代码位置
| 环节 | 文件 | 内容 |
|---|---|---|
| 提示词 | `src_llm/classify_llm.py` | `SYSTEM_PROMPT`、`FEW_SHOT_EXAMPLES`、`FEW_SHOT_SYSTEM_PROMPT` |
| 模式选择 | `src_llm/classify_llm.py` | `--few_shot` 选择 system 消息内容 |
| 单条推理 | `src_llm/classify_llm.py` | `classify_one(..., system_prompt=...)` |

## 关键约束
- 15 个类别各有两条固定示例，示例取自 `data/train.json`，不取自 `data/val.json`。
- 不传 `--few_shot` 时沿用原 zero-shot 提示词和 `outputs/llm_zero_shot_results.json`。
- few-shot 结果写入 `outputs/llm_few_shot_results.json`，避免覆盖 zero-shot 结果。
- 两种模式都要求模型只输出一个类别名，以适配 `parse_prediction`。

## 常见坑 / 排障要点
- few-shot 每条样本都携带 30 条示例，推理耗时可能明显高于 zero-shot；比较结果时应固定 `--seed` 和 `--num_samples`。

---
title: Qwen few-shot 准确率低于 zero-shot 的结果诊断
domain: project
tags: [文本分类, Qwen2-0.5B-Instruct, few-shot, zero-shot, 输出格式, 准确率]
code_refs: [src_llm/classify_llm.py, FEW_SHOT_EXAMPLES, FEW_SHOT_SYSTEM_PROMPT, classify_one, parse_prediction]
updated: 2026-10-20
confidence: medium
source: codex-session
---

# Qwen few-shot 准确率低于 zero-shot 的结果诊断

## 一句话结论
> 同一批 200 条验证标题中，few-shot 准确率由 36% 降至 23%，主要可观察变化是模型大量续写“新闻标题：…”而不输出类别，导致无法解析的比例由 29% 升至 55%。示例数量、编排和模型容量对这一格式漂移的具体贡献尚未做消融验证。

## 背景 / 它解决什么问题
- `文本分类各种训练方式效果对比.txt` 在分析时为 0 字节，没有可读取的指标；以下数据来自 `outputs/llm_zero_shot_results.json`（SHA-256 `8e869b8bcaf2a99e8299e3fb800b7e57d4f2ce81f048cca87df60b3af276bbd4`）和 `outputs/llm_few_shot_results.json`（SHA-256 `019c0a1eb3a9927a425498194ec1e759835818752cbf42e3a28a68ac334fe0cf`）。
- 两份结果的 200 条 `text` 与 `true_label` 逐条相同，属于配对比较。few-shot 30 条示例均能在 `data/train.json` 找到，且与训练集标签一致。

## 已观察到的证据
| 指标 | Zero-shot | Few-shot |
|---|---:|---:|
| 总正确数 / 准确率 | 72/200，36% | 46/200，23% |
| 未解析输出 | 58/200，29% | 110/200，55% |
| 可解析输出中的正确率 | 72/142，50.7% | 46/90，51.1% |
| 以“新闻标题：”开头的原始输出 | 0 | 82，且均未解析 |

- 配对结果：31 条由 zero-shot 正确变为 few-shot 错误，5 条反向改善。31 条退化中有 26 条变为无法解析，其中 24 条以“新闻标题：”开头。
- few-shot 原始输出例如“新闻标题：柯洁力擒朴”“新闻标题：美国政府宣布将向”，明显偏离只输出类别名的要求；部分被 `max_new_tokens=8` 截成不完整标题。已有的输出不足以证明增大生成长度能解决分类问题。
- 体育 14/19→3/19，娱乐 13/18→16/18，科技 12/20→8/20；不同类别变化方向不一致。故事和证券样本极少，不能推断其稳定表现。

## 可能机制（推断）
1. `FEW_SHOT_SYSTEM_PROMPT` 在 system 消息里按类别连续放入 30 组“新闻标题：… / 类别：…”示例，而用户消息也以“新闻标题：… / 类别：”结束。大量续写新标题的现象说明模型有时跟随示例表面格式，而没有执行最终分类指令；尚未分离示例数量、顺序及消息角色的影响。
2. `parse_prediction` 只查找 15 个类别名，续写标题以及“游戏”“房地产”等非预设标签直接记错。这是准确率下降最直接的计分机制。可解析子集的正确率几乎不变，但两个子集不同，不能据此断言语义分类能力不变。
3. 本地模型为 Qwen2-0.5B-Instruct；小模型可能更难稳定遵守长提示词中的严格输出格式。这里是合理假设，并未与更大模型对照。
4. 固定每类 2 条训练样本确实有正确标签，但仅凭这点不保证示例选取、排列和任务格式有利于当前模型。文献也报告 in-context 示例的格式与顺序可显著影响分类结果。

## 关键代码位置
| 环节 | 文件 | 内容 |
|---|---|---|
| 示例构造 | `src_llm/classify_llm.py` | `FEW_SHOT_EXAMPLES`、`FEW_SHOT_SYSTEM_PROMPT` 连续拼接 30 条示例 |
| 生成 | `src_llm/classify_llm.py` | `classify_one` 用贪心解码及 `max_new_tokens=8` |
| 解析 | `src_llm/classify_llm.py` | `parse_prediction` 仅查找预设类别名 |

## 常见坑 / 排障要点
- few-shot 不保证优于 zero-shot；先分开报告“是否输出有效标签”和“有效标签是否正确”。
- 若优化，先针对格式问题尝试更简短示例、将示例组织为独立对话轮次、让最后一条指令明确指定唯一输出标签，或在生成时限制为 15 个候选标签；再比较不同示例数量和顺序。以上均尚未在本项目重跑验证。
- 不要简单把“新闻标题：…”自动映射成某类，也不要仅增大 `max_new_tokens` 就把输出当作有效分类。

## 待确认 / 遗留问题
- [ ] 在固定同一批样本下逐项改变示例数量、顺序、消息格式和输出约束，比较准确率与未解析率。
- [ ] 对疑似数据标注噪声及类别边界另行审查。

## 外部研究依据
- Lu 等，《Fantastically Ordered Prompts and Where to Find Them》，https://arxiv.org/abs/2104.08786 ：few-shot 分类对示例顺序敏感。
- Min 等，《Rethinking the Role of Demonstrations》，https://arxiv.org/abs/2202.12837 ：示例的标签集合、文本分布和格式会影响 in-context 分类。

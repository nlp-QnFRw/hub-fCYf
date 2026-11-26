# BERT 事件抽取

建议按 `dataset.py → model.py → train.py → predict.py → bio_visualize.py` 阅读。命令均从 `diyCode` 根目录运行，完整使用方法见 [项目 README](../README.md)。

## 数据到标签

`dataset.EventDataset` 读取一条 `text + event`：先给每个字符标 `O`；按 `start/end` 给角色标 `B-ARG:角色`、`I-ARG:角色`，再给触发词标 `B-TRG:事件类型`、`I-TRG:事件类型`。重叠时后画的触发词覆盖原标签，单条 BIO 无法同时表示重叠 span。

随后 tokenizer 给出 `offset_mapping`；本实现对每个 token 取其**起始字符**的标签。Linear 的特殊 token 为 `-100`，损失函数忽略；CRF 的特殊 token 为 `O`，padding 由 attention mask 排除。`collate` 按 batch 内最长序列动态补齐。输入超过 `max_length` 被截断。

## 模型和训练

`model.EventModel` 先用预训练 BERT 输出 token 表示，再通过 Linear 计算每个标签的分数。Linear 逐 token 使用交叉熵；CRF 额外学习相邻标签、起点和终点的转移分数，以前向算法计算损失，以维特比算法解码。本 CRF 没有 BIO 硬约束，不能保证输出合法 BIO。

`train.py` 按种子从训练集划分 10% 内部验证，按验证 loss 选 checkpoint；使用 AdamW、学习率调度、梯度累积和裁剪。示例：

```bash
/d/miniconda3/envs/py312/python.exe -u -m bert.train --decoder linear --device cuda
/d/miniconda3/envs/py312/python.exe -u -m bert.train --decoder crf --device cuda
```

两种损失的计算方式不同，不要直接比较 loss 大小。各自保存在 `outputs/bert_linear.pt`、`outputs/bert_crf.pt`；再次训练同一方法会覆盖该方法的文件。

## 推理和可视化

`predict.BertPredictor` 加载 checkpoint，对长文分段并把 BIO 标签合并为 span；每个片段最多取第一个触发词，再按事件 schema 和原文校验。相同角色的 `I` 标签允许跨纯空白续接，这是有副作用的启发式。

`bio_visualize.py` 为 `/bio` 提供数据：训练模式展示真实目标与**当前模型**预测；推理模式显示原始 token 预测。字符标签是投影；它不保存也不重现之前训练过程。网页结构化 JSON 使用 `predict.py` 的完整抽取流程，而不是直接把字符投影当最终结果。

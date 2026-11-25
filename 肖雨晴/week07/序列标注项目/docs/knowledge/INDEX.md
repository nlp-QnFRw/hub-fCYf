# 项目知识索引

## 项目实现

| 关键词 / 别名 | 主题 | 文档 | 更新 | 置信 | code_refs |
|---|---|---|---|---|---|
| train_sft.py、人民日报、SFT、LoRA、CPU、MPS、内存不足、OOM、batch_size、grad_accum、max_length、NaN、答案截断 | SFT 内存参数与无有效目标的截断风险 | project/sft-memory-and-truncation.md | 2026-10-27 | high | src_llm/train_sft.py, parse_args, main, SFTDataset.__getitem__, collate_fn |

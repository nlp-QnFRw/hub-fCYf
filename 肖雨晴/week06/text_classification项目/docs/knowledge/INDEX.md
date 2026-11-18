# 项目知识索引

## 项目实现

| 关键词 / 别名 | 主题 | 文档 | 更新 | 置信 | code_refs |
|---|---|---|---|---|---|
| BERT 本地模型路径、HFValidationError、Repo id must be in the form、from_pretrained、预训练模型路径 | 文本分类脚本定位本地 BERT 模型 | project/local-bert-model-path.md | 2026-10-20 | high | BERT_PATH, src/train.py, src/evaluate.py, src/predict.py, src/compare_class_weight.py, src/explore_data.py |
| Qwen2-0.5B-Instruct、本地 Qwen 模型路径、classify_llm.py、MODEL_PATH、from_pretrained | Zero-shot 分类脚本定位本地 Qwen 模型 | project/local-qwen-model-path.md | 2026-10-20 | high | src_llm/classify_llm.py, MODEL_PATH, load_model |
| Few-Shot、Zero-Shot、SYSTEM_PROMPT、FEW_SHOT_SYSTEM_PROMPT、每类两个示例、classify_llm.py | Qwen 文本分类的固定 few-shot 提示词模式 | project/few-shot-classification-prompt.md | 2026-10-20 | high | src_llm/classify_llm.py, classify_one, FEW_SHOT_SYSTEM_PROMPT |
| llm_zero_shot_results.json、zero-shot 分类表现、体育娱乐科技、国际房产、未解析输出、类别误判 | Qwen zero-shot 200 条结果的分类别分析 | project/qwen-zero-shot-results-analysis.md | 2026-10-20 | high | src_llm/classify_llm.py, parse_prediction |
| zero-shot 为什么体育娱乐科技较好、类别边界、标签解析、标注噪声、低分类别原因 | Qwen zero-shot 类别表现差异的原因分析 | project/qwen-zero-shot-performance-causes.md | 2026-10-20 | medium | src_llm/classify_llm.py, SYSTEM_PROMPT, LABEL_NAMES, parse_prediction |
| few-shot 比 zero-shot 准确率低、few-shot 退化、输出新闻标题、未解析率、示例顺序 | Qwen few-shot 准确率低于 zero-shot 的结果诊断 | project/qwen-few-shot-regression-analysis.md | 2026-10-20 | medium | src_llm/classify_llm.py, FEW_SHOT_EXAMPLES, FEW_SHOT_SYSTEM_PROMPT, classify_one, parse_prediction |
| SFT 模型路径、train_sft.py、evaluate_sft.py、HFValidationError、Qwen2-0.5B-Instruct | SFT 训练与评估脚本的本地 Qwen 模型路径 | project/sft-local-qwen-model-path.md | 2026-10-20 | high | src_llm/train_sft.py, src_llm/evaluate_sft.py, MODEL_PATH |
| SFT 再次报同样的错、train_sft.py 路径被改回、当前运行脚本、config.json 检查 | SFT 路径重复报错的定位记录 | project/sft-local-qwen-model-path.md | 2026-10-20 | high | src_llm/train_sft.py, MODEL_PATH |
| SFT 5K 达到 BERT 53K、数据效率、Qwen 与 BERT 对比、评估样本不一致、LoRA | Qwen SFT 与 BERT 效果接近的解释及评估限制 | project/sft-vs-bert-data-efficiency.md | 2026-10-20 | medium | src/train.py, src_llm/train_sft.py, src_llm/evaluate_sft.py, SFTDataset |

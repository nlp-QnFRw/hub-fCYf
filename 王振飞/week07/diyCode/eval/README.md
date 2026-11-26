# 评测与错误分析

所有命令从 `diyCode` 根目录运行，示例显式使用 GPU：

```bash
/d/miniconda3/envs/py312/python.exe -m eval.evaluate --decoder linear --device cuda
/d/miniconda3/envs/py312/python.exe -m eval.evaluate --decoder crf --device cuda
/d/miniconda3/envs/py312/python.exe -m eval.analyze_errors --decoder linear --device cuda
```

`evaluate.py` 默认读取 `dev_extract.jsonl`，可加 `--data-dir` 和 `--limit`。它只打印结果；Git Bash 中可用 `> outputs/linear_eval.json` 保存。触发词、论元和整事件均用集合精确匹配，论元是 `(类型, 角色, 文本)`。同一条样本多个预测事件的论元会合并到集合，整事件则分别比较；该实现不等同于老师的 Counter/事件对齐口径。

`analyze_errors.py` 保存 `outputs/bert_<decoder>_predictions.jsonl` 和 `outputs/bert_<decoder>_error_analysis.json`；再次运行会覆盖它们。对误报的边界/角色等分类是字符串启发式，需要人工核实，不能视作确切根因。

`compare_whitespace.py` 使用已有 `outputs/bert_linear_predictions.jsonl` 作为修复前基线，并在 `dev_detect_clean.jsonl` 测试有无事件。若基线文件已用当前代码重跑覆盖，此比较将失去修复前基准；输出 `outputs/bert_linear_空白修复及检测评估.json` 也会覆盖。检测集标签不等同于人工复核后的线上真值。

与其他方案对比时，应固定数据、预处理、评测口径及样本集合，并记录训练配置、随机种子、耗时和显存。dev 抽取集全是带事件的样本，不能单独反映负样本误报。

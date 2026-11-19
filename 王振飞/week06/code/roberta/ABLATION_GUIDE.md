# RoBERTa 消融实验（可选实验组）

已有四组结果时，在 `code` 目录运行以下命令，补跑 exp4、exp5、exp8，并将所有已完成实验汇总：

```powershell
D:\miniconda3\envs\py312\python.exe roberta/run_ablation.py --experiments exp4 exp5 exp8
```

不带 `--experiments` 时也默认选择这三组。`--experiments` 可配置需要补跑的编号，例如 `--experiments exp5 exp8`。支持 exp1（CLS 基线）、exp2（Mean）、exp3（Max）、exp4（CLS + Mean 拼接）、exp5（学习率 3e-5）、exp6（最大长度 32）和 exp8（Label Smoothing 0.1）。没有列出的组当前尚未实现。新增目录需要包含基线 exp1，例如 `--experiments exp1 exp4 exp5 exp8`。

已完成的组会跳过，报告自动包含目录内所有已完成组；数据、模型和基本训练参数必须和原实验相同。所有组每轮使用去重验证集 Macro-F1 选择最佳轮次，再用同一验证集汇总对比；这不是独立测试集结果。exp5 的编码器学习率为 3e-5、分类头为其 5 倍；exp8 的平滑系数为 0.1，保留基线的类别加权。

结果保存在 `outputs/roberta_ablation/`：`report.md` 是报告，`summary.csv` 是指标汇总，`exp*_confusion.png` 是混淆矩阵，`metrics_comparison.png`、`class_f1_comparison.png` 和 `examples.png` 是对比图，`examples.csv` 和 `prediction_comparison.csv` 是预测实例，`exp*.json` 保存各组数据。每 20 步显示训练进度和预计时间，预计时间不包含验证。

只保留全局 Macro-F1 最佳的 `best_model.pt`，每组预测和指标仍会保留。替换权重时可能暂时同时占用新旧两个模型文件；内存也需要暂存候选模型。不要同时启动多个实例，也不要在运行期间移动或删除结果。

中断后使用相同命令重新启动，已完成的组跳过，未完成的一组从头训练，并非从中断的步骤恢复。修改基本参数、数据或模型时，请使用新的 `--output` 目录。

快速测试（不会保存模型，结果不代表正式效果）：

```powershell
D:\miniconda3\envs\py312\python.exe roberta/run_ablation.py --experiments exp1 exp4 exp5 exp8 --epochs 1 --batch_size 2 --max_steps 1 --max_eval 30 --no_model --output outputs/roberta_ablation_smoke
```

使用最佳模型预测（自动读取对应最大长度和池化方式）：

```powershell
D:\miniconda3\envs\py312\python.exe roberta/predict_roberta.py --ckpt outputs/roberta_ablation/best_model.pt --out outputs/roberta_ablation/best_pred.json
```

预测需要使用训练时相同的本地预训练模型和分词器。

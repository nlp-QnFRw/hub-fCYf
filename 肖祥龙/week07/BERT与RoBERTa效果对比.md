# BERT 与 RoBERTa 事件抽取效果对比

在 930 句 `dev_extract` 上，RoBERTa 的最高完整事件 F1 为 **0.0731**（CRF + 解码期硬约束），BERT 的最高值为 **0.2177**（CRF + 解码期硬约束）。按相同方法配对，RoBERTa 在完整事件 F1 上高于 BERT 的有 0 组，持平 0 组，低于 BERT 的有 5 组。

## 主表（dev_extract 930 句，统一评测口径）

| 方法 | trigger F1 | argument F1 | event_exact F1 | 参数量 | 训练 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| BERT Linear | 0.8293 | 0.5630 | 0.1564 | 102.4M | 636.7 s（GPU） | 11.26 ms/句（GPU） |
| RoBERTa Linear | 0.5738 | 0.3188 | 0.0510 | 102.4M | 7936.2 s（CPU） | 84.97 ms/句（CPU） |
| BERT CRF | 0.8320 | 0.5815 | 0.1697 | 102.4M | 2742.5 s（GPU） | 74.28 ms/句（GPU） |
| RoBERTa CRF | 0.6439 | 0.3459 | 0.0647 | 102.4M | 9346.0 s（CPU） | 73.39 ms/句（CPU） |
| BERT CRF + 解码期硬约束 | 0.8464 | 0.6566 | 0.2177 | 102.4M | 0 s（复用 CRF） | 39.55 ms/句（GPU） |
| RoBERTa CRF + 解码期硬约束 | 0.6022 | 0.3820 | 0.0731 | 102.4M | 0 s（复用 CRF） | 70.86 ms/句（CPU） |
| BERT CRF + 触发词回填 | 0.8320 | 0.5831 | 0.1763 | 102.4M | 0 s（复用 CRF） | — |
| RoBERTa CRF + 触发词回填 | 0.6439 | 0.3459 | 0.0647 | 102.4M | 0 s（复用 CRF） | — |
| BERT 两段式流水线 | 0.6769 | 0.3690 | 0.0957 | — | 405.5 s（历史阶段） | 49.87 ms/句（GPU） |
| RoBERTa 两段式流水线 | 0.4271 | 0.2690 | 0.0156 | 797.1M | 4535.0 s（CPU） | 132.45 ms/句（CPU） |

触发词、论元和完整事件 F1 由课程统一评测器复算。完整事件匹配要求事件类型、触发词文本以及按角色和文本计数的论元全部一致。BERT 的历史训练使用 RTX 4060 Laptop GPU。RoBERTa 使用本机 Intel Core i7-12700K CPU，配置 8 个计算线程。耗时反映各自的执行环境，速度排名需要在同一设备重测。

BERT Linear 和 CRF 的耗时采用随预测发布的指标 JSON，和课程 §8.1 的展示值来自不同记录。历史流水线 JSON 记录 405.5 s，其中包含专家重训与评估阶段。课程笔记另记完整流程 1,377.5 s。RoBERTa 流水线训练列为检测器 1663.4 s、类型分类器 1198.4 s 和论元专家 1673.2 s 的合计，参数量合计 15 个子模型。

## F1 变化（RoBERTa − BERT，百分点）

| 方法 | 触发词 Δ | 论元 Δ | 完整事件 Δ |
|---|---:|---:|---:|
| Linear | -25.55 | -24.42 | -10.54 |
| CRF | -18.81 | -23.56 | -10.50 |
| CRF + 解码期硬约束 | -24.42 | -27.46 | -14.46 |
| CRF + 触发词回填 | -18.81 | -23.72 | -11.16 |
| 两段式流水线 | -24.98 | -10.00 | -8.01 |

在 RoBERTa CRF 上，解码期硬约束使完整事件 F1 从 0.0647 变为 0.0731（+0.84 个百分点），触发词回填后为 0.0647（+0.00 个百分点）。两项变体分别以同一 CRF 为基础。

## 实验口径与复现

- 联合抽取使用 12 层编码器、3 epoch、最大长度 256、批大小 32 和随机种子 42。基础学习率为 `2e-5`，分类头学习率为基础学习率的 5 倍。拟合数据为 5,173 句，另有 574 句内层验证数据用于选取权重。
- 流水线的检测器使用 8,000 条分层抽样数据，检测器和类型分类器各训练 1 epoch。13 个类型条件论元专家各使用前 4 层训练 3 epoch，学习率为 `8e-5`。
- 本地 `../../model/roberta` 提供 12 层中文 BERT 兼容架构和 WordPiece 词表。实验迁移旧版权重键，并逐项校验编码器张量。运行版本为 PyTorch `2.6.0+cpu` 和 Transformers `5.5.3`。
- 在 `..` 中运行 `.venv\Scripts\python.exe -B run_experiments.py`，可继续未完成的正式实验或重新生成报告。训练日志、完整预测和复算指标在 `outputs` 下。

- Linear [BERT 指标](../week7-信息抽取/code/outputs/metrics/disc/bert_linear_256.json) · [RoBERTa 指标](outputs/metrics/disc/roberta_linear_256.json)
- CRF [BERT 指标](../week7-信息抽取/code/outputs/metrics/disc/bert_crf_256.json) · [RoBERTa 指标](outputs/metrics/disc/roberta_crf_256.json)
- CRF + 解码期硬约束 [BERT 指标](../week7-信息抽取/code/outputs/metrics/disc/bert_crf_256_hard.json) · [RoBERTa 指标](outputs/metrics/disc/roberta_crf_256_hard.json)
- CRF + 触发词回填 [BERT 指标](../week7-信息抽取/code/outputs/metrics/disc/bert_crf_256_rule.json) · [RoBERTa 指标](outputs/metrics/disc/roberta_crf_256_rule.json)
- 两段式流水线 [BERT 指标](../week7-信息抽取/code/outputs/metrics/disc/pipeline.json) · [RoBERTa 指标](outputs/metrics/disc/roberta_pipeline.json)





## 一句话结论

**换模型之前先看数据。** 20 秒训完的 TF-IDF 已经到 **0.915**（THUCNews）/ **0.507**（TNEWS），
与 BERT 的差距只有 **1.0–1.7 个点**（TNEWS 上统计不显著，p=0.25）；同架构换预训练权重（BERT → RoBERTa）只值 **+0.49 个点**；
而清洗 TNEWS 里 11% 的重复样本，能带来 **3.2 个点**。数据质量比换模型更值钱。

## 实验矩阵

| 轴 | 取值 |
|---|---|
| 数据集 | TNEWS（15 类 / 53k / 不均衡 / 11% 泄漏）· THUCNews（10 类 / 180k / 均衡 / 1.15% 泄漏） |
| 预训练模型 | bert-base-chinese · Chinese-RoBERTa-wwm-ext（同 12 层 / 768 隐层，只换权重） |
| 方法 | 关键词规则 · TF-IDF（字）· TF-IDF（词）· TextCNN · BERT · RoBERTa |
| 评测 | 去重 test · Macro-F1（主）+ ACC + 每类 F1 + 混淆矩阵 · 500 次 bootstrap 95% CI · McNemar |

## 结果

| 方法 | THUCNews ACC | THUCNews Macro-F1 | TNEWS ACC | TNEWS Macro-F1 | 训练成本（THU） |
|---|---|---|---|---|---|
| 关键词规则 | 0.3038 | **0.3326** | 0.2705 | **0.2922** | 0s |
| TF-IDF（字） | 0.9153 | **0.9152** | 0.5198 | **0.5071** | 6s |
| TF-IDF（词） | 0.8983 | **0.8983** | 0.5012 | **0.4887** | 6s |
| TextCNN | 0.8772 | **0.8770** | 0.4886 | **0.4807** | 72s |
| BERT-base | 0.9323 | **0.9324** | 0.5284 | **0.5176** | 11.9min |
| RoBERTa-wwm-ext | 0.9372 | **0.9373** | — | — | 12.4min |

> TNEWS 的 RoBERTa 因时间限制未跑；模型替换的结论以 THUCNews 上的 BERT→RoBERTa 对照为准。
> 95% CI、每类 F1、混淆矩阵、McNemar 见 `outputs/summary/summary.md` 与 `outputs/analysis/analysis.md`。

## 关键发现

1. **预训练的优势只有 1–2 个点**：THUCNews 上 BERT 比 TF-IDF 高 1.7 点（显著），TNEWS 上只高 1.0 点且不显著
   （2 epoch 下 BERT 在 TNEWS 尚未收敛）。
2. **同架构换预训练权重收益很小**：BERT → RoBERTa +0.49 点（p=0.014），小于清洗 11% 重复样本的 3.2 点。
3. **中文短标题上字 n-gram 稳定优于 jieba 词 n-gram**（+1.7 ~ +1.8 点）。
4. **泄漏会系统性抬高词面方法**：TNEWS 含重复 val 上 TF-IDF 0.560，去重后 0.528；
   重复样本本身 0.821。主实验统一用去重 test。
5. **成本差两个数量级**：TF-IDF 训练 <10 秒、单条 <0.3 ms；BERT 训练约 13 分钟、单条约 45 ms。

## 目录

```
week7-模型与数据集迁移/
├── README.md                  # 本文件：结论 + 复现入口
├── REPORT.md                  # 完整实验报告（数据 / 方法 / 结果 / 分析 / 局限）
├── data/                      # TNEWS + THUCNews（含 README 说明来源与泄漏）
├── models/                    # 本地 BERT / RoBERTa 权重说明
├── src/
│   ├── run.py                 # 单次实验入口：--dataset --method
│   ├── data.py                # 读数据 / 去重 / 切分 / 分层抽样
│   ├── metrics.py             # 统一指标 + bootstrap CI + McNemar
│   ├── leakage.py             # 泄漏对指标影响的小实验
│   ├── aggregate.py           # 汇总表 + 图 + 错误分析
│   ├── make_slides.py         # 生成汇报 PPT
│   └── methods/               # rule / tfidf / textcnn / transformer_cls
├── scripts/                   # 一键复现脚本
├── outputs/                   # 每次 run 的 run.json + predictions.jsonl
├── figures/                   # 报告与 PPT 用图
└── docs/                      # 汇报 PPT
```

## 快速复现

```bash
# 0) 环境（Python 3.10+，有 GPU 更好；MPS 也能跑）
pip install -r requirements.txt

# 1) 主实验：THUCNews 6 种方法
bash scripts/run_thucnews.sh
#    主实验：TNEWS 6 种方法
bash scripts/run_tnews.sh

# 2) 泄漏实验 + 汇总表 + 图
bash scripts/run_analysis.sh

# 3) 生成 PPT
python -m src.make_slides
```

单次实验：

```bash
python -m src.run --dataset thucnews --method bert --train-limit 60000
python -m src.run --dataset tnews --method tfidf_char
python -m src.run --dataset thucnews --method roberta --epochs 2
```

> 本机把依赖装在 `.deps/`（系统 site-packages 无写权限），运行时需要
> `source scripts/env.sh`（等价于 `PYTHONPATH=.deps`）。在其他机器上直接
> `pip install -r requirements.txt` 即可。

## 对比

| | before | after |
|---|---|---|
| 数据集 | TNEWS | TNEWS + THUCNews |
| 预训练模型 | Qwen3-0.6B（生成式） | BERT + RoBERTa（判别式） |
| 方法 | 规则 / TF-IDF / BERT / zero-shot / LoRA / 全量微调 / API | 规则 / TF-IDF(字/词) / TextCNN / BERT / RoBERTa |
| 评测 | 500 条干净 val | 4,421 / 9,880 条干净 test，带 95% CI + McNemar |
| 重点 | 「数据接入」怎么接 | 「换模型 / 换数据」谁更值 |

第 6 周的 TNEWS BERT 0.556 是「3 epoch + 500 条样本」，本周统一 2 epoch、4,421 条 test 上为 0.518；
差距主要来自训练预算和评测样本，而不是代码不一致——这本身也说明「跨周比数字」必须先对齐口径。

## 产物说明

- `outputs/<dataset>/<method>/run.json`：指标、配置、耗时、环境、命令行；
- `outputs/<dataset>/<method>/predictions.jsonl`：逐条预测，可直接做错误分析；
- `outputs/summary/`：`summary.csv` / `summary.md` / `summary.json`；
- `outputs/analysis/analysis.md`：数据概况、McNemar、错误案例；
- `figures/`：benchmark、per-class F1、混淆矩阵、成本-效果、数据量、泄漏；
- `docs/week7_实验汇报.pptx`：10 页汇报。

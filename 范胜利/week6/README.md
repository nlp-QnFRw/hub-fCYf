# 第六周 · BERT 基线 vs 中文 RoBERTa 同配置对比

TNEWS 15 类中文短文本分类，**同一份数据、同一套超参、同一个随机种子**，只换基座：

| 基座 | 说明 | 参数量 |
|---|---|---|
| `bert-base-chinese` | 谷歌官方中文 BERT：12 层 768 隐层 12 头 21,128 词表，中文维基预训练，MLM + NSP | 102.3M |
| `chinese-roberta-wwm-ext` | 哈工大讯飞：同结构，RoBERTa 训练策略 + 全词掩码（WWM），语料规模约 10 倍 | 102.3M |

两者结构、词表完全相同 → 差异主要来自**预训练语料与预训练目标**，而不是模型容量。这是做基座对比最干净的条件。

## 一键复现

```bash
cd week6
pip install -r requirements.txt          # torch / transformers / sklearn / matplotlib / pandas
bash run_all.sh                          # 下模型 → 训 BERT → 训 RoBERTa → 出报告（本机约 1 小时，含下载）
```

分步、便于观察每一步：

```bash
python src/train.py --backbone bert-base-chinese
python src/train.py --backbone chinese-roberta-wwm-ext
python src/compare.py                    # 对比表 + 各类别 F1 变化 + 5 张图 + 报告
```

几十秒验证代码通路（不覆盖正式产物）：

```bash
python src/train.py --backbone bert-base-chinese --smoke
python src/train.py --backbone chinese-roberta-wwm-ext --smoke
python src/compare.py --tag smoke
```

## 目录

```
week6/
├── data/                 train.json(53,360) / val.json(10,000)
├── models/               两个基座权重 + 下载说明
├── src/
│   ├── _bootstrap.py     只做一件事：保证 torch 先于 numpy 加载（否则本机 segfault）
│   ├── config.py         唯一一份训练配置 + 模型定位/下载
│   ├── labels.py         15 类标签口径
│   ├── data.py           读取 / 分层抽样 / 去重 / 动态 padding
│   ├── model.py          基座无关的「BERT + 池化 + 分类头」
│   ├── metrics.py        统一指标（Acc / Macro-F1 / per-class / ECE）
│   ├── train.py          单基座训练（--backbone 选基座）
│   ├── compare.py        对比表 + 各类别 F1 变化 + 对比图 + Markdown 报告
│   └── plotting.py       中文字体与图表样式
├── outputs/
│   ├── metrics_*.json           每个基座的完整指标 + 配置 + 逐 epoch 日志 + 混淆矩阵
│   ├── pred_*.json              逐条预测（idx / 真实 / 预测 / 置信度）
│   ├── comparison_overall.csv   整体指标对比
│   ├── per_class_f1.csv         15 类 F1 / Recall / Precision 及变化
│   ├── comparison_report.md     自动生成的完整报告（含全部表格与图）
│   └── figures/                 5 张对比图
├── run_all.sh
└── requirements.txt
```

## 关键设计（为什么这份对比可信）

1. **配置只有一份**：`config.py` 的 `TrainConfig`，两个 run 逐字段相同；`compare.py` 会先断言两份 config 一致，不一致直接退出，不出「看起来正常」的错报告。
2. **指标只有一个实现**：`metrics.py`，训练与对比共用，避免一个用 weighted F1、另一个用 macro F1。
3. **同一批数据**：分层抽样用固定 seed，两个基座拿到的是**完全相同的 8,000 条**。
4. **同一批判测集**：全量验证集 10,000 条，同一顺序；另报去重验证集（训练集/验证集有 1,099 条完全重复，占 11%）。
5. **主指标 Macro-F1**：15 类、类间样本量 23 倍差距（科技 1,089 vs 证券 45），Accuracy 会被头部类带偏，尾部类塌了只有 Macro-F1 会现形。
6. **动态 padding**：按 batch 内最长序列 padding，而不是全部补到 64。实测每步从 12.4s 降到 ~8s，且不改变模型看到的内容。

## 实验规模（重要）

本机是 Intel i9-9880H + 无可用 GPU（`torch 2.2.2` 是 macOS x86_64 的最后一个版本，MPS 上做 BERT 反向传播会直接崩），
实测 `bert-base` / batch 64 / 平均 27 token 约 **12 秒/步**。全量 53,360 条跑 3 epoch 需要 ≈ 8 小时/单基座，
所以正式对比默认：

| 项 | 值 |
|---|---|
| 训练样本 | 6,000 条（15 类分层抽样，固定 seed） |
| epochs | 2 |
| batch_size | 64 |
| 单基座耗时 | ≈ 35 分钟 |

两个基座用的是**同一子集、同一配置**，所以「谁更好、差多少」的结论不受规模影响；
受影响的只是绝对指标（数据少了，绝对值会低于全量训练）。

有 GPU 时把规模放大到全量（一条命令，无需改代码）：

```bash
python src/train.py --backbone bert-base-chinese --train_size 0 --epochs 3
python src/train.py --backbone chinese-roberta-wwm-ext --train_size 0 --epochs 3
python src/compare.py
```

## 本机踩过的两个环境坑（代码里已规避，写在这里供参考）

1. **`numpy` 先于 `torch` 加载 → segfault（exit 139）**
   环境里有两个 OpenMP 运行时：numpy 走 conda MKL 的 `libiomp5.dylib`，torch 2.2 自带 `libomp.dylib`。
   先 import numpy 会把 MKL 的线程池先初始化，torch 之后开线程时两套运行时抢状态，崩在
   `libomp.dylib __kmp_suspend_initialize_thread`（lldb 抓到的栈），Python 层看不到任何异常。
   `KMP_DUPLICATE_LIB_OK` / `MKL_THREADING_LAYER=GNU` 都无效；**唯一有效的修复是 import 顺序**——
   见 `src/_bootstrap.py`，所有入口脚本第一行 import 它。

2. **`transformers>=4.48` 拒绝加载 `.bin`（CVE-2025-32434）**
   要求 torch>=2.6 才允许 `torch.load`。本机 torch 2.2 没有升级路径，所以
   `config._ensure_safetensors()` 会把 `pytorch_model.bin` 一次性转成 `model.safetensors`
   （顺便处理了 tied embedding 共享存储无法写入 safetensors 的问题），之后所有加载走 safetensors 分支。

3. **中文图表字体**：按候选列表探测（Arial Unicode MS / Hiragino Sans GB / Heiti TC …），
   探测不到会打印警告并回退 DejaVu Sans，不会静默出方框。

## 结果速览（本机实测，6,000 条训练 / 2 epoch / seed 42）

| 基座 | 准确率 | Macro-F1 | 加权 F1 | ECE | 去重验证 Macro-F1 |
|---|---|---|---|---|---|
| BERT-base-chinese | 0.5301 | 0.5143 | 0.5286 | 0.0924 | 0.5059 |
| chinese-roberta-wwm-ext | **0.5403** | **0.5260** | **0.5383** | **0.0754** | **0.5171** |
| Δ（RoBERTa − BERT） | +1.02pp | +1.17pp | +0.97pp | −0.0170（越低越好） | +1.12pp |

- 15 类里 **11 类上涨、4 类下降**；平均 +1.16pp。
- 涨得最多：证券 +4.62pp（n=45）、农业 +3.05pp（n=494）、教育 +2.56pp（n=646）、电竞 +2.43pp（n=659）。
- 跌得最多：故事 −2.21pp（n=215）、财经 −0.67pp（n=956）、文化 −0.25pp、娱乐 −0.13pp。
- 三条口径（全量验证集 / 加权 F1 / 去重验证集）方向一致 → 提升不是靠背验证集或刷头部类。
- 结论：RoBERTa 略优且置信度更可信（ECE 更低），但 1pp 出头属于「小而一致」的差距，正式迁移前建议补 3 个 seed。

完整表格与逐类分析见 `outputs/comparison_report.md`，图见 `outputs/figures/`。

### 复用已下载的权重

权重目录可以放到别处，用环境变量指过去，避免重复下载 800MB：

```bash
WEEK6_MODEL_DIR=/path/to/models python src/train.py --backbone bert-base-chinese
```

## 使用训练好的模型

### 1. 先存权重（训练默认不存，避免 400MB×2 进仓库）

```bash
python src/train.py --backbone chinese-roberta-wwm-ext --save_ckpt
# → outputs/ckpt_chinese-roberta-wwm-ext.pt  约 390MB
#   ckpt 内含 state_dict + 训练配置（pool/max_length/seed）+ 基座名 + 指标，推理端自描述
```

### 2. 推理

```bash
# 单条
python src/predict.py --ckpt outputs/ckpt_chinese-roberta-wwm-ext.pt \
    --text "二手房首付都付了，卖家现在不想卖了怎么办？"

# 批量：txt 一行一条，或 .json / .jsonl（字段名 sentence / text 都认）
python src/predict.py --ckpt outputs/ckpt_chinese-roberta-wwm-ext.pt \
    --file inputs.txt --out preds.json

# 拿验证集抽查 20 条，对照真实标签
python src/predict.py --ckpt outputs/ckpt_chinese-roberta-wwm-ext.pt --file data/val.json --limit 20
```

输出示例：

```
文本                        预测   置信     主题组  投研   Top3
二手房首付都付了…            房产   0.8742   金融    是     房产0.87 / 财经0.06 / 证券0.02
```

每条结果字段：`label_id / label / prob / group / relevant / topk[]`，`--out` 落成 json 供下游服务用。

### 3. 在服务里常驻调用（加载一次，反复推理）

```python
from pathlib import Path
from predict import build_model, predict      # src/ 已在 sys.path

model, tok, meta = build_model(Path("outputs/ckpt_chinese-roberta-wwm-ext.pt"))
res = predict(model, tok, ["二手房卖家违约怎么办"], meta["max_length"], topk=3)[0]
print(res["label"], res["prob"], res["group"], res["relevant"])
```

**四个容易踩的点**

1. ckpt 只存参数、不存分词器——分词器永远跟着基座走（`--backbone` 或 ckpt 里记录的那个）。
2. `pool` / `max_length` 必须与训练一致：脚本从 ckpt 的 config 里自动读，换 pool 不会报错但结果会变差。
3. 推理设备默认 CPU（单句几十毫秒，没必要上 GPU/MPS）。
4. 模型判的是「15 类新闻主题」，不是「业务相关性」；`group` / `relevant` 只是按 labels.py 的映射表派生，口径要改就改那张表，不用重训。

## 图目录

见 `outputs/comparison_report.md`（自动生成，数字与 csv/json 完全一致）：

| 图 | 内容 |
|---|---|
| `fig_01_overall_metrics.png` | 准确率 / Macro-F1 / 加权 F1 分组柱状图 + Δpp |
| `fig_02_per_class_f1.png` | 15 类 F1 对比（上）+ 各类 ΔF1（下），按样本量降序 |
| `fig_03_class_delta.png` | ΔF1 排行（横向条形，正 = RoBERTa 更好） |
| `fig_04_training_curves.png` | 训练 loss / 验证 Macro-F1 曲线，空心点 = 最优轮 |
| `fig_05_confusion_matrices.png` | 行归一化混淆矩阵对比 |

## 环境

- Python 3.12，torch 2.2.2，transformers 4.55.0，scikit-learn 1.9，matplotlib 3.10
- 设备自动选择：CUDA > MPS（仅 Apple Silicon）> CPU；`--device` 可强制指定，`--threads` 控制 CPU 线程
- 模型自动下载走 `HF_ENDPOINT=https://hf-mirror.com`（国内直连 huggingface.co 常超时）

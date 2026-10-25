# 基于 RNN / LSTM / BiLSTM 的十二种语言识别

## 一、项目目标

构建一个**字符级**的语言识别模型——输入一段文本，自动判断它属于哪种语言。

- **12 种目标语言**：阿拉伯语 (ar)、英语 (en)、西班牙语 (es)、印地语 (hi)、日语 (ja)、韩语 (ko)、葡萄牙语 (pt)、俄语 (ru)、泰语 (th)、土耳其语 (tr)、越南语 (vi)、中文 (zh)
- **任务类型**：多分类
- **模型对比**：分别用 RNN、LSTM、BiLSTM 三种循环结构训练，评估效果差异

## 二、目录结构

```
作业/
├── README.md                   ← 本文件
├── data/                       ← 数据集
│   ├── train.csv               训练集（9600 条）
│   ├── dev.csv                 验证集（1200 条）
│   ├── test.csv                测试集（1200 条）
│   └── vocab.txt               字符级词表（3441 条）
│
└── code/                       ← 源代码
    ├── config.py               全局超参数与路径配置
    ├── vocab.py                词表构建与文本 → ID 转换
    ├── trainData.py            CSV 数据加载
    ├── model.py                RNN / LSTM / BiLSTM 模型定义
    ├── train.py                训练循环、早停、评估、交互入口
    ├── compare.py              多模型批量对比脚本
    │
    ├── checkpoints/            训练完成的权重文件
    │   ├── rnn_best.pt
    │   ├── lstm_best.pt
    │   └── bilstm_best.pt
    │
    └── reports/                compare.py 生成的对比报告
        ├── model_comparison.csv
        └── model_comparison.png
```

## 三、架构设计

```
输入文本 "你好世界"
    │
    ▼
┌──────────────────┐
│  Vocab.texts_to  │  字符级分词 → 查表 → Padding / Truncation → [B, 64]
│    ids()         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Embedding      │  3441 × 128 → [B, 64, 128]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  RNN / LSTM /    │  输出每个时间步的隐藏状态 → [B, 64, H*]
│    BiLSTM        │  H = 128，BiLSTM 时 H* = 256
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   LayerNorm      │  对时间维做归一化
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Masked Mean     │  排除 PAD 位置，对有效字符求均值 → [B, H*]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Dropout        │  0.3
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Linear         │  H* → 12（12 类 logits）
└──────────────────┘
```

### 各模块职责

| 文件 | 职责 |
|---|---|
| `config.py` | 集中管理所有超参数（模型类型、embedding 维度、hidden size、dropout、batch size、学习率、早停 patience 等）和路径 |
| `vocab.py` | 从 `vocab.txt` 加载词表；若文件不存在则从 train.csv 按字符统计构建；提供 `texts_to_ids()` 将文本批量转为定长 ID 序列 |
| `trainData.py` | 读取 csv → 调用 `vocab.texts_to_ids()` → 返回 `torch.Tensor`，分 x_train / x_dev / x_test 三套 |
| `model.py` | 根据 `config.MODEL_TYPE` 动态构建 RNN / LSTM / BiLSTM；Embedding 层设 `padding_idx=0`；分类头输出 12 维 logits |
| `train.py` | 训练循环：每 epoch 先 shuffle、再分批训练、梯度裁剪防爆炸、早停用 dev Accuracy、自动保存最优权重；`__main__` 提供训练 / 评估 / 交互预测的 CLI |
| `compare.py` | 扫描 checkpoints 目录下所有 `*_best.pt`，用 `importlib.reload` 动态切换 `MODEL_TYPE` 逐个加载，统一在 dev 和 test 上评估，输出 CSV 和对比柱状图 |

## 四、实现过程

### 4.1 数据处理

- **字符级分词**：`list(row[0])` 将每条文本拆成字符列表，每个字符作为独立 token
- **动态词表**：词表大小从 `vocab.txt` 文件行数动态读取，避免硬编码漂移
- **定长序列**：所有文本截断或填充到 MAX_LEN=64（数据中最长句 62 字符）
- **Label 编码**：`LABELS = ['ar','en',...,'zh']`，`LABELS.index(row[1])` 转为 0-11 的整数

### 4.2 模型结构

三种模型共享同一套分类头，仅循环层不同：

| MODEL_TYPE | 循环层参数 | 输出维度 | 参数量 |
|---|---|---|---|
| rnn | `nn.RNN(128, 128, num_layers=2, dropout=0.3)` | 128 | ~508K |
| lstm | `nn.LSTM(128, 128, num_layers=2, dropout=0.3)` | 128 | ~706K |
| bilstm | `nn.LSTM(128, 128, num_layers=2, dropout=0.3, bidirectional=True)` | 256 | ~1104K |

序列聚合方式：用 PAD mask 排除填充位置，对有效字符的隐藏状态**求均值**得到句向量，再接分类头。

### 4.3 训练策略

| 策略 | 实现 |
|---|---|
| **Shuffle** | 每个 epoch 用 `torch.randperm(len(x_train))` 重新打乱数据顺序 |
| **梯度裁剪** | `clip_grad_norm_(model.parameters(), 5.0)`，RNN 必需，防梯度爆炸 |
| **优化器** | Adam，lr=1e-3 |
| **损失函数** | CrossEntropyLoss |
| **早停指标** | Dev Accuracy（越大越好），每轮评估一次 |
| **早停 patience** | 连续 4 轮 Dev Acc 无创新高 → 停止训练 |
| **随机种子** | `SEED=42`，在 `Model()` 实例化前设置，保证权重初始化可复现 |
| **最优模型保存** | Dev Acc 创新高时覆盖写入 `checkpoints/{model_type}_best.pt` |

### 4.4 多模型对比

`compare.py` 的核心技巧：Python 进程内用 `importlib.reload(model_module)` 配合动态修改 `config.MODEL_TYPE`，无需重启即可依次构建三种结构。脚本自动扫描 `checkpoints/` 目录下的所有 `*_best.pt`，从文件名推断模型类型，统一在 dev 和 test 集上评估，输出结构化表格和柱状图。

## 五、实现结果

### 5.1 训练日志示例（以 BiLSTM 为例）

```
Epoch 1/100  Train Loss: 1.8345
[dev] Loss: 0.9652  Acc: 0.7250
Epoch 2/100  Train Loss: 0.6234
[dev] Loss: 0.3821  Acc: 0.9108
...
Epoch 9/100  Train Loss: 0.1023
[dev] Loss: 0.0412  Acc: 0.9883
Epoch 10/100 Train Loss: 0.0856
[dev] Loss: 0.0393  Acc: 0.9892
Epoch 11/100 Train Loss: 0.0712
[dev] Loss: 0.0405  Acc: 0.9875     ← 未创新高，patience 开始累加
...
早停触发，验证集 Acc 连续多轮无改善
```

### 5.2 三种模型对比结果

| 模型 | Dev Loss | Dev Acc | Test Loss | Test Acc | 参数量 |
|---|---|---|---|---|---|
| **BiLSTM** | 0.0393 | **98.92%** | 0.0501 | **98.89%** | 1,104K |
| LSTM | 0.0493 | 98.83% | 0.0501 | 98.22% | 706K |
| RNN | 0.0424 | 98.50% | 0.0463 | 98.44% | 508K |

### 5.3 结果分析

- **BiLSTM 效果最好**：双向结构让每个字符都能同时看到前文和后文，对语言识别这种依赖全局上下文的任务最有利
- **三种模型差距不大**：字符级语言识别本身不算难——不同语言的字符集差异很大（中文汉字、日语假名、韩语谚文、阿拉伯文从右到左），模型很容易学会"哪些字符出现代表哪种语言"
- **RNN 测试集 Loss 最低但 Acc 不是最高**：说明 RNN 的置信度校准更好，但 argmax 之后分对的条数略少于 BiLSTM
- **BiLSTM 参数是 RNN 的 2 倍多**：精度提升（+0.45%）是否值得翻倍的计算量和内存开销，要看实际需求

### 5.4 快速使用

```bash
# 安装依赖
pip install torch pandas matplotlib

# 进入代码目录
cd code/

# 训练（交互式 CLI，选 1 开始训练）
python train.py

# 加载已训练模型评估 test 集（交互式 CLI，选 2）
python train.py

# 输入一句话让模型预测语言（交互式 CLI，选 3）
python train.py
# → 请输入语句：Hello world
# → 预测结果：en  置信度：0.998

# 一键对比所有 checkpoints
python compare.py
```

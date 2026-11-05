# work — 字符级语言模型：从零训练 vs 微调预训练

两条线跑在同一份语料、同一个验证集上，最后能放在一张表里看：

```
              ┌─ GPT（从零训练，单向） ──────────────┐
data/raw/*.txt ┤                                      ├─ compare.py → 阶段对比表
              └─ BERT（加载预训练权重，双向 MLM） ──┘
```

本目录**完全独立**，语料自带。
唯一的外部依赖是 `..\bert-base-chinese`（BERT 那条线要用）。

## 快速开始

```bash
conda activate py312
cd work

python selftest.py                        # 1) 先验证实现（20 项断言，约 10s）
python prepare_data.py                    # 2) 清洗 + 去重 + 划分 + 编码（约 7s）

python train.py                           # 3a) GPT 1200 步，约 1 分钟
python generate.py --prompt "今天天气"     # 3b) 采样看效果

python bert_train.py                      # 4a) BERT 1200 步，约 4 分钟
python compare.py                         # 4b) 生成两边的阶段对比表
```

依赖：`torch / numpy / tqdm / tokenizers`，见 `requirements.txt`。

## 文件

| 文件 | 作用 |
|---|---|
| `model.py` | GPT：因果自注意力、pre-LN 残差块、权重绑定、采样解码 |
| `data.py` | 词表、uint16 二进制语料、随机裁剪取 batch、固定窗口评估 |
| `prepare_data.py` | 清洗 → 精确去重 → 近重复去重（MinHash）→ 划分 → 建词表 → 编码 |
| `train.py` | GPT 训练：warmup+余弦、梯度裁剪、AMP、断点续训、CSV 日志 |
| `generate.py` | 采样（temperature / top-k / top-p） |
| `bert_model.py` | BERT：12 层 encoder、MLM 头，**直接加载 bert-base-chinese 权重** |
| `bert_data.py` | WordPiece 分词、句子打包成块、MLM 动态遮字（80/10/10） |
| `bert_train.py` | BERT 继续训练：线性衰减、分阶段记录 MLM loss / 命中率 / 伪 PPL |
| `compare.py` | 读两份 metrics.csv，输出阶段对比表 + `comparison.md` |
| `selftest.py` | 20 项断言：掩码方向、归一化、loss、优化器、数据管线、权重加载、遮字能力 |
| `data/raw/demo.txt` | 演示语料，6 万行中文短句（含原始重复行与爬虫垃圾，供清洗演示） |
| `runs/` | GPT 产物：`best.pt`(16M) / `last.pt`(47M) / `metrics.csv` |
| `runs_bert/` | BERT 产物：`final.pt`(454M) / `metrics.csv` / `comparison.md` |

## 为什么 BERT 不用 transformers

这台机器的 `transformers 4.55` 与已装的 `tokenizers`（模块 0.23.3、元数据 0.21.4）和
`huggingface-hub 0.36.2` 版本对不上，`import transformers` 直接抛 `ImportError`。

与其改别人的环境，不如把 BERT 写出来——反正它比 GPT 还简单（没有因果掩码、没有采样）。
分词仍然用官方的 `tokenizer.json`（`tokenizers` 库能正常工作），所以分词结果和 HF 完全一致；
权重按 checkpoint 的原键名加载，严格模式下 **207 个张量全部命中**，
只要有一个键名或形状错位就会直接报错——这本身就是一次结构校验。

一个坑：这份 `bert-base-chinese` 是老版 checkpoint，LayerNorm 的参数叫 `gamma`/`beta`
而不是 `weight`/`bias`，所以 `bert_model.py` 自己写了 `LayerNorm` 而不是用 `nn.LayerNorm`。

## 实测结果

语料：6 万行 → 清洗去重后 5.36 万行 → 训练 5.09 万行 / 验证 2679 行，**重叠 0 行**。

### GPT（从随机初始化开始）

| 步数 | 看过 token | 验证 PPL | bpc |
|---|---|---|---|
| 200 | 1.64M | 96.81 | 6.60 |
| 600 | 4.92M | 54.32 | 5.76 |
| 1200 | 9.83M | 43.71 | 5.45 |
| 1260（续训） | 10.32M | **43.36** | 5.44 |

对照基线：1-gram 546.39 / 2-gram 142.66 / 3-gram 520.75。
（3-gram 比 2-gram 差不是笔误：80 万 token 下三元组大量只出现 0~1 次，平滑项把概率压平了。）

### BERT（从预训练权重开始）

| 阶段 | 看过 token | MLM loss | 遮字命中率 | 伪 PPL |
|---|---|---|---|---|
| 0（原版，未训练） | 0 | 2.1378 | 63.3% | 13.34 |
| 200 | 0.31M | 1.4017 | 69.3% | 7.56 |
| 600 | 0.92M | 1.3230 | 72.3% | 7.23 |
| 1200 | 1.84M | **1.2723** | **72.3%** | **6.88** |

第 200 步就吃掉了大部分收益，之后进入平台期——5 万行语料对 119M 参数的模型来说太小了。

### 两边放一起看

| | GPT | BERT |
|---|---|---|
| 参数量 | 4.07M | 119.11M |
| 起点 | 随机初始化 | 预训练权重 |
| 终点 | PPL 43.36 | 伪 PPL 6.88 |
| 看过 token | 10.32M | 1.84M |
| 训练耗时 | 61s | 240s |

**这个表不能横着比大小。** 两个指标任务不同：

- GPT 的 PPL = `exp(-log P(下一个字 | 左边))`，只能用左边；
- BERT 的伪 PPL = `exp(-log P(被遮的字 | 左右两边))`（Salazar et al., 2020），偷看了右边。

BERT 数值低是因为它作弊——能看到未来。这正是「双向编码器不能做生成」的原因：
它没有「下一个字」的概念，**要比生成，BERT 根本不能上桌**。
真正能得出的结论只有两条：预训练权重让 BERT 一上来就有 13.34 的水平；
而 GPT 从零开始，靠一千万 token 把 PPL 从 3325（随机初始化的理论值）压到 43.36。

## 相对课件里 LSTM 版本的改动

| 问题 | 这里的做法 |
|---|---|
| 语料重复 37.7%，打乱切分后验证集 60% 是训练集原句 | 先清洗、再去重（精确 + MinHash），**之后**才划分，重复率降为 0 |
| 字表外的字直接丢弃，制造伪相邻二元组 | 统一映射到 `[UNK]`（索引 0） |
| 无梯度裁剪、无学习率调度 | `--grad-clip 1.0`、warmup + 余弦（GPT）/ 线性衰减（BERT） |
| 只存 best，中断就得重头再来 | `last.pt` 存模型+优化器+scaler+步数，`--resume` 接着跑 |
| 训练与验证的切块口径不同 | 训练随机裁剪、评估固定不重叠窗口，按 token 加权 |
| 自检数据导出了但没人用 | `selftest.py` 20 项断言，含「改后面的字不能影响前面输出」（GPT）与「必须影响」（BERT） |
| bias / LayerNorm 也吃 weight decay | 参数分组，只对二维以上权重衰减 |

## 常用参数

```bash
# 模型规模
python train.py --n-layer 6 --n-embd 384 --n-head 6 --block-size 384
python bert_train.py --max-len 128 --batch-size 24 --steps 800

# BERT 只验证能跑（CPU，20 步）
python bert_train.py --device cpu --steps 20 --max-len 64 --batch-size 4 --save-model 0

# 换语料
python prepare_data.py --input ../循环神经网络语言模型/corpus_general.txt --max-vocab 6000
```

`bert_train.py` 的 `--bert-dir` 留空会自动找（依次试 `BERT_DIR` 环境变量、`../bert-base-chinese`、
`D:\Baidu\BaiduNetdiskDownload\bert-base-chinese`），找不到会列出所有尝试过的路径。

## 已知限制

- 演示语料只有 6 万行短问句、4~119M 参数，学到的只是**句式**：GPT 能生成像中文的短句但不会真回答问题；
  BERT 的伪 PPL 在第 200 步后基本不动，是语料太小而不是代码有问题。
- 换行符占训练 token 的 6.39%，GPT 会经常输出空行。这是语料按行组织的必然结果。
- 评估窗口固定不重叠，窗口首字的左侧没有上下文，PPL 会比滑动窗口口径差几个点——好处是可复现。
- `runs_bert/final.pt` 有 454MB（fp32 的 119M 参数）。只要指标不要权重的话加 `--save-model 0`。
- BERT 的伪 PPL 每次要逐位置遮一遍，块开多了会很慢（默认只算 6 个块）。

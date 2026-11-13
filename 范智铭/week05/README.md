# week05：字符级 Transformer 的 SFT 指令微调

在 week04 预训练模型（`checkpoints/best.pt`）基础上，用 `data/sft_train.jsonl` 做
**SFT（Supervised Fine-Tuning，指令微调）**，让模型学会「问：… 答：…」的问答格式，
并且**只对答案部分计算 loss**。

- 预训练：学会「中文下一个字该是什么」——通用语言能力；
- SFT：学会「看到问题该怎么答」——问答格式、答案风格、任务类型。

---

## 目录结构

```
week05/
├── config/
│   └── config.yaml              # 模型结构（与预训练一致）+ SFT 超参
├── data/
│   ├── sft_train.jsonl          # SFT 训练数据 3000 条（instruction/output）
│   ├── sft_val.jsonl            # SFT 验证数据 200 条
│   ├── vocab.txt                # 字符级词表 5343 个 token（与预训练共用，勿改）
│   └── stats.json               # prepare_sft.py 生成的数据统计
├── code/
│   ├── tokenizer.py             # CharTokenizer（与 week04 相同）
│   ├── prompt.py                # 统一问答模板（训练 / 推理共用）
│   ├── transformer_block.py     # 单层 Transformer Block（与 week04 相同）
│   ├── my_module.py             # Decoder 模型（与 week04 相同，权重可直连）
│   ├── sft_train.py             # ★ SFT 训练入口
│   ├── demo.py                  # ★ SFT 推理：问 → 答 / 对比 / 交互式
│   └── util/
│       ├── sft_dataset.py       # ★ SFTDataset：模板 + 答案掩码 + 错位 + padding
│       ├── generate.py          # 生成 / 答案 perplexity
│       ├── prepare_sft.py       # 数据体检（长度、OOV、泄漏、mask 可视化）
│       └── verify_pipeline.py   # 管线自检（32 条样本过拟合测试）
├── checkpoints/
│   ├── best.pt                  # week04 预训练权重（只读，不会被覆盖）
│   └── sft_best.pt              # ★ SFT 产物（val loss 最优）
├── docs/
│   └── sft_train_tutorial.md    # ★ sft_train.py 逐行讲解（教学文档，建议先读）
├── train.log                    # 本次 SFT 训练日志
└── README.md
```

与 week04 的关系：`tokenizer.py` / `prompt.py` / `transformer_block.py` / `my_module.py`
**原样复用**（模型结构一字不改，所以 `best.pt` 的权重能直接加载），新增的只有 SFT
数据管线、训练脚本和推理脚本。

> `data/vocab.txt` 是从 week04 复制过来的（5343 行），必须与 `best.pt` 的
> `embedding.weight` 行数严格对应 —— 换词表就等于换模型，只能重新预训练。
> 注意：词表第 76 号 token 就是一个空格字符本身，读取词表时**不能 strip 空白**
> （会把 5343 行读成 5342 行，直接与 embedding 对不上）。

> **想逐行读懂训练代码？** 见 [`docs/sft_train_tutorial.md`](docs/sft_train_tutorial.md)：
> 按代码从上到下的顺序，把 `sft_train.py` 的 433 行逐段讲清楚（为什么这么写、
> 不写会出什么问题、日志里的数字怎么手算出来）。

---

## 一、SFT 与预训练到底差在哪

| | 预训练 `week04/code/train.py` | SFT `week05/code/sft_train.py` |
|---|---|---|
| 初始化 | 随机初始化 | 加载 `checkpoints/best.pt` |
| 样本 | 整篇语料拼起来，按 129 切块 | 一条样本 = 一条问答 |
| 标签 | `y = x` 右移一位，**每个位置都算 loss** | 只对 **`答：` 之后**的 token 算 loss |
| 目标 | 学会说中文 | 学会按问答格式回答 |
| 学习率 | 1e-4（Adam） | 3e-4（AdamW + warmup + 余弦退火） |
| 评估 | test loss / perplexity | val 答案 loss + Greedy 生成质量 |

### 1. 模板必须只有一份（`prompt.py`）

```
问：{instruction} 答：{output}<eos>
```

预训练、SFT、推理三处共用同一个 `format_prompt()`，否则推理时模型看到的
格式和训练时不匹配，效果会明显变差。`<eos>` 是唯一终止符：训练行尾、答案末尾、
生成停止都用它。

### 2. 只对答案算 loss（prompt mask）

一条 59 token 的样本，实际参与 loss 的只有 12 个：

```
[问][：][给][定][两][个][数][字]…[答][：]  较大的数字是 564。<eos>
└──────────── 47 个 token，label = -100 ────────────┘└── 12 个监督 token ──┘
```

- 用 `CrossEntropyLoss(ignore_index=-100)` 实现；
- 不 mask prompt 的话，模型会把容量花在复述问题上，且不同题目长度差异会带偏梯度；
- 训练集监督 token 占比 42.6%（84,412 / 197,956）。

### 3. 错位一位（最容易踩的坑）

因果模型 `logits[t]` 预测的是**第 t+1 个 token**，所以 `SFTDataset`
返回的是 `(input_ids[:-1], labels[1:])`。忘了错位会出现两个症状：

- loss 虚高、几乎不降（模型在学「前一个字是什么」，而不是「下一个字」）；
- batch 越大越差（padding 把错位放大）。

本次开发中就踩到了这个坑：逐条 loss 3.04，一批 32 条却变成 7.32。
改完后逐条与批内 loss 完全一致（2.757 = 2.757），说明右侧 padding 对因果注意力无害。

### 4. padding 与 batch

- 同一 batch 内右对齐 pad 到最长样本，`pad_id = 0`，pad 位置 label 仍是 `-100`；
- 因果 mask 保证真实 token 只看左边，永远不会看到右侧的 pad，所以**不需要 attention mask**，
  也没有额外的 loss 污染；
- 样本最长 117 token（`max_len = 128`），随机组 batch 后平均有效长度约 63/110。

### 5. 学习率与调度

```yaml
lr: 3e-4              # 5% 线性 warmup + 余弦退火到 3e-5
weight_decay: 0.01    # 只作用在 Linear/Embedding 权重，bias 与 LayerNorm 不衰减
grad_clip: 1.0
label_smoothing: 0.0
```

实测 5e-5 太保守（2 轮只降 0.08），3e-4 收敛快且稳定，所以默认用 3e-4。

---

## 二、数据管线

```
sft_train.jsonl (3000)  ─┐
sft_val.jsonl   (200)   ─┤  json.loads → 清洗 NBSP/零宽字符 → 套模板
                         ▼
        input_ids = encode("问：" + instruction + " 答：" + output + <eos>)
        labels    = [-100] * len(prompt) + encode(output) + [<eos>]
                         ▼
        __getitem__ → (input_ids[:-1], labels[1:])  →  collate 右侧 pad
```

长度策略（位置编码只有 128 个位置）：优先保留 prompt，超出时从答案尾部截断，
并保证答案至少留 1 个 token。当前数据 **0 条被截断**。

数据体检结果（`python code/util/prepare_sft.py`）：

| | train | val |
|---|---|---|
| 条数 | 3000 | 200 |
| prompt 长度 (min/中位/max) | 13 / 39 / 53 | 16 / 38 / 53 |
| 答案长度 | 5 / 23 / 65 | 7 / 23 / 64 |
| 样本总长度 | 24 / 64 / 117 | 33 / 64 / 111 |
| 被截断 | 0 | 0 |
| 监督 token 占比 | 42.6% | 43.2% |
| OOV 字符 | 0（清洗后） | 0 |
| train/val 完全重复 | 0 组 | — |

---

## 三、训练

### 启动

```bash
python code/util/prepare_sft.py      # 1. 数据体检（可选，但建议先跑）
python code/sft_train.py             # 2. SFT 训练 → checkpoints/sft_best.pt
```

### 命令行覆盖

```bash
python code/sft_train.py sft.epochs=10 sft.lr=2e-4    # OmegaConf 风格覆盖
python code/sft_train.py --epochs 5 --save checkpoints/tmp.pt
python code/sft_train.py --limit-train-batches 20     # 冒烟测试，写 smoke.pt
```

### 本次训练结果（GTX 1660 Ti，15 epoch，约 80 秒）

| 指标 | SFT 前（best.pt） | SFT 后（sft_best.pt） |
|---|---|---|
| val 答案 loss | 3.0441 | **2.6999**（epoch 13） |
| val 答案 perplexity | 20.99 | **14.89** |
| 训练 loss（末轮） | — | 2.1988 |
| Greedy 完全匹配率（10 条） | 0% | 0% |

训练曲线要点：

- val loss 从 3.04 稳定降到 2.70，第 13 轮最优；
- 训练 loss 继续降到 2.17 而 val 不再降 → 开始过拟合，所以只保存 val 最优的那一轮
  （早停耐心 4 轮，本次 15 轮跑完未触发）；
- 完全匹配率一直是 0%，符合预期：这是一个 2.6M 参数的字符级小模型，
  它学到的是**答题格式和句式**，不是算术/推理（见「已知局限」）。

### Checkpoint 格式（`checkpoints/sft_best.pt`）

```python
{
  "model":        state_dict,     # 完整的 SFT 后权重
  "optimizer":    state_dict,     # AdamW 状态
  "scheduler":    state_dict,     # LambdaLR 状态
  "cfg":          dict,           # 完整配置快照（模型 + SFT 超参）
  "epoch":        13,
  "val_loss":     2.6999,
  "val_ppl":      14.88,
  "gen_em":       0.0,
  "base_val_loss":3.0441,
  "history":      [...],          # 每轮指标
  "init_ckpt":    ".../best.pt",  # 从哪个预训练权重出发
}
```

`demo.py` 只依赖 `model` + `cfg`，所以换 checkpoint 不用改配置。
`sft_train.py` 会拒绝把结果写到 `best.pt`，避免覆盖预训练权重。

---

## 四、推理（demo.py）

```bash
python code/demo.py                     # 内置问题 + 验证集前 5 条对比
python code/demo.py --interactive       # 交互式问答，Ctrl+C 退出
python code/demo.py --compare           # 预训练 best.pt vs SFT 并排对比
python code/demo.py --temperature 0     # 贪心解码（稳定、可复现）
```

实测输出（采样，temperature 0.7 / top-k 40 / top-p 0.9）：

```
问：计算一组数字的平均值。 数字：10, 20, 30, 40, 50
答：这组数字的平均值为30。                ← 答对了

问：给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 128 数字2: 395
答：数字1和数字2的最小公倍数是95。          ← 格式对，数值错

问：检查这个电子邮件地址是否符合格式：li.lei@example.com
答：该图片中的图书是Jomadins，建议您在电脑上搜索其他手机。   ← 跑题
```

`--compare` 能明显看出 SFT 的作用（同一问题）：

```
问：计算一组数字的平均值。 数字：10, 20, 30, 40, 50
预训练：这组数字的平均值为 10。
SFT   ：这组数字的平均值是 10.5。
```

预训练模型会把「答：」也当作正文往下续写、甚至串到别的问句里；
SFT 模型稳定停在「一句答案」，并会输出 `<eos>` 停止。

每个问题还会打印**答案 perplexity**（只算答案 token）：

- ppl 1~3：模型很自信，答案像它训练里见过的说法；
- ppl 越高，答案越「别扭」。

---

## 五、自查工具

```bash
python code/util/prepare_sft.py        # 数据体检：长度/OOV/泄漏/mask 可视化
python code/util/verify_pipeline.py    # 管线自检：32 条样本过拟合
```

`verify_pipeline.py` 是排查「SFT 不收敛」最快的办法：

```
过拟合验证 | 32 条样本 | batch 最长 100 token | 监督 token 1026
  step   50 | loss 0.0555
  step  100 | loss 0.0154
  ...
  ✓ 管线正常：模型能把少量样本背下来，loss 降不下去就是容量/数据量的问题
```

如果 loss 卡在 3~5，优先检查：label 是否错位、prompt 是否被误 mask、词表是否与权重匹配。

---

## 六、已知局限

这个任务的上限由预训练模型决定：**2.6M 参数、d=128、6 层、字符级**。

- ✅ 学会了：问答格式、`<eos>` 停止、「这组数字的平均值为…」这类句式；
- ❌ 学不会：真正的算术、逻辑、翻译。它只是在「像答案一样说话」；
- 训练 loss 2.17 / val loss 2.70 的差距说明 3000 条数据已经开始过拟合，
  再加轮数只会更差（早停 + 保存 best 已经处理）。

想真正提升问答质量，需要先提升预训练模型（更大 d、更多层、更多语料），
而不是继续调 SFT 超参。SFT 代码本身与模型规模无关。

---

## 七、环境

| 组件 | 版本 |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124（GTX 1660 Ti，6GB） |
| NumPy | 2.5.4 |
| OmegaConf | 2.4.0 |

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`python code/sft_train.py` / `python code/demo.py` 直接跑，不需要虚拟环境。

---

## 八、完整流程

```bash
# 0. 前提：checkpoints/best.pt（week04 预训练）+ data/vocab.txt 已就位
python code/util/prepare_sft.py      # 1. 数据体检
python code/sft_train.py             # 2. SFT 训练（约 80 秒）
python code/demo.py                  # 3. 看效果
python code/demo.py --compare        #    和预训练模型对比
python code/demo.py --interactive    #    自己随便问
```

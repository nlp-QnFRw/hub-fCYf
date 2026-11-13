# sftlab · 从零跑通一次 SFT

用 `sft_train.zip` 里的 3000 条指令数据，从零训练一个 11.6M 参数的中文小模型，
完整走一遍 **预训练 → SFT** 两阶段，亲眼看到 Base Model 如何从「续写器」变成「问答助手」。

全程**不依赖 transformers、不下载任何模型权重、不需要联网**，一张 4070 上 2 分钟跑完。

> 本项目对应课件《从 BERT 到大模型》第五部分（SFT）。代码注释里标了每一段对应哪一页的概念。

---

## 目录结构

```
work/
├── configs/              pretrain.yaml / sft.yaml —— 两个阶段的超参
├── data/
│   ├── raw/              sft_train.jsonl（3000）/ sft_val.jsonl（200）
│   └── processed/        vocab.json（字符级词表 2286）
├── src/sftlab/           框架本体
│   ├── tokenizer.py      字符级分词器（含 <|user|> 等特殊 token）
│   ├── template.py       ★ chat template 与 loss mask
│   ├── data.py           两种 Dataset：全位置 loss / 只算回答
│   ├── model.py          decoder-only Transformer（因果 mask 手写实现）
│   ├── generate.py       温度 / top-k / top-p 采样
│   ├── train.py          训练循环（AMP、warmup+cosine、早停、存盘）
│   └── utils.py          配置、种子、日志
├── scripts/
│   ├── prepare_data.py   解压 → 建词表 → 打印统计
│   ├── train.py          训练入口（--stage pretrain|sft）
│   ├── show_mask.py      ★ 把一条样本的 loss mask 打印出来
│   ├── eval.py           验证集 loss / ppl + 样例报告
│   ├── plot_curves.py    从日志还原训练曲线
│   └── chat.py           Base 与 SFT 并排对话
├── docs/SFT_NOTES.md     概念笔记 + 常见坑 + 升级到真模型的路径
└── outputs/              检查点、日志、指标、评估报告
```

---

## 环境

```bash
conda activate py312
# 环境里已装好 torch 2.13 + CUDA；如需重装：
pip install -r requirements.txt
```

依赖只有 `torch / numpy / pyyaml`，没有 `transformers`，也没有任何模型下载。

---

## 快速开始

```bash
cd work

python scripts/prepare_data.py                 # 1. 解压 + 建词表（1 秒）
python scripts/show_mask.py                    # 2. 看 loss mask（理解 SFT 的关键）
python scripts/train.py --stage pretrain       # 3. 预训练，约 1 分钟
python scripts/train.py --stage sft            # 4. SFT，约 1 分钟
python scripts/plot_curves.py                  # 5. 画训练曲线
python scripts/chat.py -p "什么是人工智能？"    # 6. Base 与 SFT 并排对比
```

想先确认流程能不能跑通，加 `--max-steps 30` 几十秒即可：

```bash
python scripts/train.py --stage pretrain --max-steps 30
```

---

## 两个阶段到底在做什么

| | 阶段一 pretrain | 阶段二 SFT |
|---|---|---|
| 数据形态 | `问题 回答\n问题 回答\n…` 拼成长流 | `<|user|>问题<|assistant|>回答<|end|>` |
| 谁参与 loss | **每个 token** | **只有回答部分**（prompt 全部 -100） |
| 学到什么 | 中文的统计规律 → 会接龙 | 应答格式 + 何时停 → 会回答 |
| 初始化 | 随机 | 阶段一的 `best.pt` |
| 参数量 | 11.62M | 11.62M（一模一样） |
| 对应 PPT | 第一部分 | 第五部分 25–32 页 |

**架构、参数量、attention mask、训练目标全都没变**——变的只有数据和 loss 的位置。
这就是 PPT 第 29 页「SFT 到底改变了什么」的代码版本。

---

## 关键点 1：loss mask 长什么样

```bash
python scripts/show_mask.py
```

```
模型看到的字符串：
  <|user|>给定两个数字，请选择其中的较大数…<|assistant|>较大的数字是 564。<|end|>

prompt 44 token ｜ 回答 12 token ｜ 序列共 56 token ｜ 参与 loss 21.4%
--------------------------------------------------------------------------
 位置 │ token        │ 是否学
   43 │ <|assistant|> │ —
   44 │ 较            │ 学 ✔
   45 │ 大            │ 学 ✔
  …
--------------------------------------------------------------------------
mask 分布（█ 学习 / · 忽略）:
  ············································████████████
```

prompt 段全部 `-100` 不产生梯度。整个数据集平均只有 **44.7%** 的 token 参与训练——
省下来的算力全砸在「怎么答」上，这就是 SFT 的样本效率来源。

---

## 关键点 2：为什么模型能「答完就停」

回答末尾补了 `<|end|>`，它和其他 token 一样参与 loss。推理时把 `end_id` 加进停止集合，
生成到它就结束。**没有这个 token，模型学到的是永不停止地往下编**——这是手写 SFT 最容易漏的一步。

---

## 实测结果（RTX 4070 Laptop，单次运行）

| 阶段 | 训练步数 | 耗时 | 训练 loss | 验证 loss | 验证 ppl |
|---|---|---|---|---|---|
| 阶段一 · pretrain（best，第 11 轮） | 575（存于第 253 步） | 56s | 1.709 | **2.868** | 17.6 |
| 阶段二 · SFT（best，第 3 轮） | 2350（存于第 282 步） | 63s | 1.788 | **3.064** | 21.4 |
| 阶段二 · SFT（last，第 25 轮） | 2350 | 63s | **0.007** | 4.238 | 69.3 |

`best.pt` 按验证集挑、`last.pt` 是最后一轮。

**两个阶段都是「先学规律、再背数据」——这是最值得讲的现象：**

| 阶段二 SFT · epoch | 1 | 3 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|
| val_loss | 3.184 | **3.064** | 3.092 | 3.450 | 3.793 | 4.094 | 4.238 |

训练 loss 一路掉到 0.007，验证 loss 却在第 3 轮见底后涨到 4.24。18 万 token 的语料，
11.6M 参数的模型两三轮就把「规律」学完，剩下的轮次全在记答案。

两个检查点的用途不同：

- `last.pt`：训练集题目能**逐字复现**（验证过），适合演示「格式学对了、答完会停」；
- `best.pt`：验证 loss 更低，遇到新问题更像样一点，但训练不充分，容易说半句话。

```bash
python scripts/chat.py -p "…"                                   # 默认 last.pt
python scripts/chat.py -p "…" --sft-ckpt outputs/sft/best.pt    # 换 best 对比
```

---

## 训练曲线

```bash
python scripts/plot_curves.py     # 从 train.log + metrics.json 还原，不需要重训
```

![预训练曲线](outputs/curve_pretrain.png)

![SFT 曲线](outputs/curve_sft.png)

两张图叠在一起（`outputs/curves_all.png`）就是完整的对照：横轴是轮数，深蓝是训练 loss、橙色是验证 loss、红色空心点是验证集最优点，右侧浅红区域是**过拟合区**（最优点之后，训练还在降、验证已经在涨）。

同样的形态出现了两次，说明瓶颈不在「预训练还是 SFT」，而在数据量：18 万 token
对 11.6M 参数来说，够学语法和答题格式，不够学知识。

---

## 为什么效果只有这个水平？

这不是 bug，是三个硬约束：

1. **数据量**：3000 条、约 18 万 token，只够学会「句式模板」，学不会知识。GPT-3 的 SFT 用了上万条人工标注，且背后有万亿 token 的预训练底座。
2. **字符级词表**：一个汉字一个 token，`238` 是三个 token，算数几乎不可能学会；真实模型用 BPE，且靠规模而非规则解决算术。
3. **模型规模**：11.6M 参数、6 层。而且 3000 条数据对 11.6M 参数来说是严重过参数化（每个参数只分到约 15 个 token），必然先记住再谈泛化。

对应的三条改进路径：

| 想要什么 | 怎么做 |
|---|---|
| 回答更像样 | 换 BPE 词表（`tokenizers`）+ 更多数据 |
| 真的会做题 | 换 0.5B~1.5B 预训练底座 + LoRA 微调 |
| 只是演示机制 | 就用本项目，1 分钟一轮，改一行 config 就能做对照实验 |

升级到真模型的步骤见 `docs/SFT_NOTES.md` 最后一节。

---

## 可以做的小实验

| 改什么 | 预期现象 |
|---|---|
| 把 `template.py` 里 prompt 的 mask 去掉（改成全 `ids`） | 收敛变慢、同轮数下回答更差 → 证明 mask 有用 |
| 删掉 `<|end|>` | 生成的回答停不下来，一直编到 max_tokens |
| `configs/sft.yaml` 里 `init_from: null` | 不从预训练热启动，收敛明显更慢更差 |
| `block_size` 降到 64 | 长样本被截断，回答残缺（看 prepare 的 truncated 统计） |
| `temperature` 调到 0 / 1.5 | 0 = 每次一样但易复读；1.5 = 胡言乱语 |

---

## 常见坑（都已在代码里处理）

1. **label 必须和 input 同长、同切分**：截断时 ids 与 labels 要一起切，否则 mask 整体错位一格。
2. **padding 位置的 label 填 -100，不是 pad_id**，否则模型会认真学「怎么输出 pad」。
3. **词表只用训练集统计**，别把验证集喂进去。
4. **loss 要用 float32 算**：bf16 下 log_softmax 精度不够，loss 会抖。
5. **同步 SDK / 大 tensor 别阻塞事件循环**：本项目是纯训练脚本，但 `generate` 里的采样是逐 token 串行的，批量推理需要另行处理。

---

## 参考

- 课件：`../从BERT到大模型.pptx` 第五部分（SFT）与第六部分（API）
- 数据：`../sft_train.zip`
- 相关演示：`../web/generation_lab.html`（采样参数）、`../llm_failover`（API 容错）

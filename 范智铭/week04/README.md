# 字符级 Transformer 语言模型

从零实现的 Transformer Decoder，在中文问答语料上训练字符级自回归语言模型。支持训练 / 早停 / checkpoint 保存 / 续写 / 通顺度判断。

---

## 目录结构

```
week04/
├── config/
│   └── config.yaml              # 模型结构 + 训练超参
├── data/
│   ├── corpus_general.txt       # 原始语料（每行一条独立文本）
│   ├── train.txt                # 训练集（加了 <eos>）
│   ├── test.txt                 # 测试集
│   ├── vocab.txt                # 字符级词表（5262 个 token）
│   └── stats.json               # 数据统计信息
├── code/
│   ├── tokenizer.py             # CharTokenizer：字符级编解码
│   ├── transformer_block.py     # 单层 Transformer Block（MHA + FFN）
│   ├── my_module.py             # 完整 Decoder：Embedding → N×Block → Linear
│   ├── train.py                 # 训练入口（含 evaluate / 早停 / best 保存）
│   ├── demo.py                  # 推理入口（续写 / perplexity / 交互模式）
│   └── util/
│       ├── dataset.py           # MyDataset：拼接长序列 → 固定长度切块
│       ├── prepare_data.py      # 数据准备脚本（语料 → train/test/vocab）
│       └── verify_data.py       # 数据校验脚本
├── checkpoints/
│   └── best.pt                  # 训练产物（模型权重 + 优化器 + 配置）
└── README.md
```

---

## 模型架构

纯 Decoder Transformer，因果 Mask，自回归 next-token prediction：

```
输入 id 序列 (B, T)
        │
        ▼
Token Embedding + Position Embedding + Dropout
        │
        ▼
  × N 层 Transformer Block:
      ├── Multi-Head Self-Attention (causal mask)
      │     └── QKV → split heads → scale-dot → causal mask → softmax → concat → Linear
      ├── Add & Norm
      ├── FFN: Linear → GELU → Linear
      └── Add & Norm
        │
        ▼
Linear → (B, T, vocab_size)   # 每个位置对所有词的 logits
```

当前配置（`config.yaml`）：

| 超参 | 值 |
|------|----|
| model_size | 128 |
| num_heads | 8 |
| num_layers | 6 |
| d_ff | 512 |
| dropout | 0.1 |
| max_len | 128 |
| vocab_size | 5262 |

---

## 数据管线

### 原始 → 训练产物

```
corpus_general.txt (395,286 行, 562 万字符)
        │ prepare_data.py
        ▼
train.txt / test.txt     每行加 <eos>，按 9:1 切分
vocab.txt                训练集字符频次统计，前 4 行为 <pad> <unk> <bos> <eos>
stats.json               统计信息
```

### Dataset 切块方式

```
所有行编码后拼接成一条长 token 序列
        │
        ▼
按 max_len + 1 不重叠切块 (stride = 129)
        │
        ▼
每块 → (x, y)，其中 x = block[:-1], y = block[1:]   ← y 是 x 右移一位
```

训练集 5,419,462 token → **42,011 个样本**；测试集 602,184 token → **4,668 个样本**。

---

## 训练

### 启动训练

```bash
python code/train.py
```

### 训练特性

| 特性 | 说明 |
|------|------|
| 设备自动选择 | 有 CUDA 走 GPU，否则 CPU |
| 固定随机种子 | random / numpy / torch / cuda 全设置 |
| 梯度裁剪 | `clip_grad_norm_` 1.0 |
| DataLoader | `shuffle=True`（train）/ `pin_memory=True` |
| 早停 | `early_stop_patience` 个 epoch test loss 不下降就停 |
| Best 保存 | 每次刷新 test loss 就存 `checkpoints/best.pt` |
| 定期日志 | 每 100 step 打印一次 avg loss |
| 每 epoch 评估 | train loss + test loss + perplexity |

### 早停配置（config.yaml）

```yaml
hparams:
  epochs: 100
  early_stop_patience: 10     # 设为 0 或负数则关闭早停
```

### 训练输出示例

```
训练开始 | device=cuda | 每 epoch 657 步 | early_stop_patience=10
  Epoch 1 [100/657] | step_loss 6.8418 | avg 7.5527
  ...
Epoch 1/100 | train 6.1799 | test 5.4514 | ppl 233.1 | 33.2s
  ✓ 新 best，已保存到 checkpoints/best.pt
...
Epoch 99/100 | train 2.8912 | test 3.2259 | ppl 25.2 | 32.8s
  ✓ 新 best，已保存到 checkpoints/best.pt
```

### Checkpoint 格式

```python
{
    "model":       state_dict,               # 模型权重
    "optimizer":   state_dict,               # 优化器状态
    "cfg":         OmegaConf.to_container(), # 配置快照
    "epoch":       99,
    "val_loss":    3.2259,
}
```

---

## 推理（demo.py）

三种实际应用场景：

### 场景 1：续写

给定开头，模型自回归往后生成（采样 + temperature）。

```bash
python code/demo.py
```

```
开头: "如何才能"  →  "如何才能赚到钱？"
开头: "请问"      →  "请问您的这笔交易是支付宝账户是多少？"
开头: "微信"      →  "微信聊天记录，怎么改"
```

### 场景 2：通顺度判断（Perplexity）

Perplexity 越低，模型越觉得这句话像真的。实际可用于过滤乱码/灌水评论。

```
  5.2  ✓ 通顺  如何破解别人邮箱密码？
  3.7  ✓ 通顺  明天天气怎么样？
281.7  ✓ 勉强  xyzxyz 明天天气怎么样？       ← 开头 OOV，ppl 升高
34M    ✗ 可疑  今 天 天 气 不 错              ← 空格都是 OOV
```

### 场景 3：交互式

```bash
python code/demo.py --interactive
```

手动输入开头，实时看续写结果，Ctrl+C 退出。

### 换 checkpoint

```bash
python code/demo.py --ckpt checkpoints/best.pt
```

---

## 完整使用流程

```bash
# 1. 准备数据（首次运行 / 语料更新后）
python code/util/prepare_data.py
python code/util/verify_data.py       # 可选：校验产物完整性

# 2. 训练
python code/train.py

# 3. 推理
python code/demo.py                   # 场景 1 + 2
python code/demo.py --interactive     # 场景 3
```

---

## 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0
- OmegaConf
- NumPy

---

## 最终训练指标

| 指标 | 值 |
|------|----|
| 训练 epoch | 99 |
| train loss | 2.89 |
| **test loss** | **3.23** |
| **test perplexity** | **25.2** |
| OOV rate | 0.0075% |
| vocab size | 5,262 |

# 字符级 Transformer 模型 SFT（监督微调）实现要点

> 背景：已在足够语料上用 [transformer_lm.py](./transformer_lm.py)（GPT 结构字符级 Transformer，2.66M 参数）完成预训练，现需在**较小语料**上进行 SFT。
>
> SFT 的本质：**加载预训练权重 → 在小语料上用小学习率继续训练 → 时刻防过拟合**。

---

## 1. 权重加载：必须"结构性复原"，不能 new 一个就 load

关键坑：直接 `TransformerLM(...)` 传新超参再 `load_state_dict` 会**形状不匹配报错**。正确做法是沿用 ckpt 里存的 args 原样重建（本仓库 `generate` 函数已经是这个写法，SFT 照搬）：

```python
ck = torch.load(args.ckpt, map_location=device)
pre_args = ck["args"]
model = TransformerLM(len(ck["char2idx"]), pre_args["d_model"], pre_args["heads"],
                      pre_args["layers"], pre_args["seq_len"], args.dropout,
                      use_pos=not pre_args.get("no_pos", False),
                      causal=not pre_args.get("no_mask", False)).to(device)
model.load_state_dict(ck["model_state"], strict=True)
```

- **`strict=True`（默认）**：确保预训练参数一个不少、一个不多地加载进来，是防"悄悄漏加载"的保险；
- 加载完先跑一次零步骤评估（复用 `evaluate`），**确认 SFT 前 PPL 与预训练结束时一致**——这是排查加载错误的基本手段；
- 若要复现同一套 train/val 划分，需用预训练时的同一个 seed 重放按行打乱。

## 2. 字表处理：SFT 最容易踩的隐性坑

字符级模型的"词表"就是 `char2idx`，SFT 时有三种选择，含义完全不同：

| 方案 | 做法 | 适用 |
|---|---|---|
| **冻结字表（推荐）** | 直接用 ckpt 里的 `char2idx`，SFT 语料中不在表内的字按 `CharDataset` 现有逻辑丢弃 | 一般情况；保证 embedding / 输出头权重全部有效继承 |
| 扩表 | 新增字符**追加到表尾**，只初始化新行：`nn.init.normal_(model.tok.weight[old_V:], std=0.02)` | SFT 语料含大量领域新字符 |
| 重建字表 | 按新语料重建 | ❌ 等于作废全部 embedding + 输出头（还是 weight-tied 的），预训练白费 |

**Weight Tying 注意事项**：本模型 `head.weight = tok.weight`（输出头与输入 embedding 共享同一份张量）。扩表时改一处即改两处，不会出对不上的问题；但新行初始化务必用小方差（std=0.02，GPT-2 惯例），否则初始 loss 会飙升。

## 3. 优化器：新开一个，但用更小的学习率

```python
opt = torch.optim.Adam(model.parameters(), lr=args.lr)   # 建议比预训练小 5~10 倍，如 1e-4
```

- **新开 optimizer**，不尝试恢复预训练时的 Adam 状态（`exp_avg / exp_avg_sq`）——小语料上旧动量意义不大，且预训练脚本本来就没存 optimizer state；
- **学习率必须降**（如 1e-3 → 1e-4 量级）：SFT 是"微调"不是"重训"，大学习率会一把抹掉预训练知识（**灾难性遗忘**）；
- **考虑加 LR 衰减**：小语料几个 epoch 就过拟合，`lr *= 0.5` 每 epoch 的 StepScheduler 或 cosine 都行——预训练阶段"刻意最小实现"的取舍在 SFT 阶段不再适用，衰减是必要的。

## 4. 要不要冻结底层？两种流派

```python
# 方案 A：全部参数微调（简单，小模型首选）
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# 方案 B：冻结 embedding + 前几层，只训后几层 + 输出头
for p in model.tok.parameters():
    p.requires_grad = False
for p in model.encoder.layers[:2].parameters():   # 冻结前 2 层
    p.requires_grad = False
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

- 本模型仅 2.66M 参数，**全参微调通常就够**，冻结是语料特别小（几万字符以下）时的保险手段；
- **Weight Tying 连带效应**：冻结 `tok` 就等于冻结 `head`（同一份权重）；若想"冻 embedding、放输出头"在此结构下做不到，实现时要心里有数。

## 5. 过拟合防护：SFT 阶段的核心矛盾

语料小 → 模型几个 epoch 内就会把训练集背下来。必须做：

```python
best = float("inf")
for ep in range(1, args.epochs + 1):          # epochs 设小，如 2~4
    train_one_epoch(model, opt, sft_train_loader, ...)
    _, v_ppl = evaluate(model, sft_val_loader, device)
    if v_ppl < best:
        best = v_ppl
        torch.save({...}, args.save)           # 存最优，不存最后
    else:
        break                                   # 或计数器触发早停
```

- **留验证集**：哪怕 SFT 语料只有几千行，也要切 5~10% 出来，复用按行打乱切分的逻辑；
- **存 best 不存 last**：预训练脚本已具备该逻辑，照搬；SFT 阶段 val PPL 常在 epoch 2~3 就触底反弹；
- **训练 / 验证 loss 一起打印**：两者开始分叉（train 降、val 升）就是过拟合信号；
- **Dropout 保持 0.1 别关**——它是小语料上最便宜的正则。

## 6. 防灾难性遗忘：数据回放（可选但推荐）

纯用小语料训，模型会忘掉通用语感（表现为续写风格塌缩、高频字复读——与 `--no-mask` 泄漏实验里的"们们们们"现象同源）。做法：

```python
# SFT 语料 : 预训练语料 ≈ 1:1 到 4:1 混合
mix = sft_text + "\n" + pre_text[:len(sft_text) * 2]
```

混一个比例的原始语料进来一起训，几乎零成本，效果通常明显更好。

## 7. 工程细节清单

- [ ] **保存时换文件名**（如 `tf_sft.pt`），别覆盖预训练 ckpt——预训练权重是不可再生资产；
- [ ] ckpt 里照旧存全量 `char2idx / idx2char / args`，并**额外记一个标记**（如 `"stage": "sft"`），方便日后追溯权重产自哪个阶段；
- [ ] `train()` / `evaluate()` 循环体**原样复用**（`model.train()` ↔ `model.eval()` 成对出现、梯度裁剪保留、token 加权统计口径不变）——SFT 改的只是"初始权重、数据、学习率"，训练闭环本身不需要动；
- [ ] 固定 seed 并**存进 ckpt**，保证 SFT 验证集可复现（现有 `--seed` 机制直接沿用）。

---

## 一句话总结

在这份代码基础上做 SFT，改动集中在五处：

1. **从 ckpt 的 args + 字表原样重建模型并 strict 加载**
2. **字表冻结**（或只对表尾新行做小方差初始化）
3. **新 Adam + 小 5~10 倍的学习率**
4. **小 epoch 数 + 验证集选优存档**——防过拟合是第一要务
5. **可选混入预训练语料防遗忘**

训练循环、评估口径、`generate` 全链路均可原样继承。

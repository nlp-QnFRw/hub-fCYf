# SFT 概念笔记

> 配合课件《从BERT到大模型》第五部分使用。这里只记「写代码时必须知道」的部分，
> 概念本身请对照 PPT 25–32 页。

---

## 一、三阶段在代码里的位置

```
预训练 Pre-training     语料 → next-token prediction（全位置 loss）
        ↓               本仓库：train.py --stage pretrain
SFT                     指令对 → 同一目标，但只算回答（loss mask）
        ↓               本仓库：train.py --stage sft
对齐 RLHF/DPO           偏好数据 → 学「哪个回答更好」（本项目不涉及）
```

三阶段**都是自回归语言模型训练**，没有一步改架构。区别只在数据从哪来、哪些 token 算 loss。

---

## 二、chat template 的四个必备元素

| 元素 | 作用 | 漏了会怎样 |
|---|---|---|
| `<|user|>` | 标出提问方的开始 | 模型分不清哪段是问、哪段是答 |
| `<|assistant|>` | 标出回答的开始 | 推理时没有「该你说了」的信号 |
| `<|end|>` | 标记说完 | 停不下来，一直编到 max_tokens |
| `-100` mask | 只学回答 | 花一半算力学「复述问题」 |

真实模型的模板更复杂（ChatML、Llama-3 的 `<|start_header_id|>` 等），但这四件事一件都不能少。
换模型时**必须**用官方模板，模板不对，模型能力直接掉一半——这是实际项目里最常见的翻车点。

---

## 三、loss mask 的三个细节

**1. 错位发生在模型内部，不是数据里**

```python
ids    = [A, B, C, D]        # A B 是 prompt，C D 是回答
labels = [-100, -100, C, D]
logits[:, :-1]  ←→  labels[:, 1:]
```
位置 1（预测 C 的那一步）的 label 是 C，参与 loss；
位置 0（预测 B）的 label 是 -100，被跳过。
所以 **prompt 的最后一个 token 仍然有梯度**——它要预测回答的第一个字。
这一点和「prompt 完全不算 loss」的直觉不同，画 mask 时容易数错一格。

**2. mask 比例决定训练效率**

本项目实测只有 44.7% 的 token 参与 loss。真实场景里 system prompt 很长时，这个比例会低到 10% 以下，
于是有了「packing（多条短样本拼进一个 block）」这类工程优化。

**3. 不能只 mask、不截断**

样本超过 `block_size` 时必须把 `ids` 和 `labels` **一起**切，否则 mask 会整体错位。
`data.py` 的 `SFTDataset` 里 `truncated` 计数就是在盯这件事。

---

## 四、验证集的正确用法

- 词表只用**训练集**统计（本项目 `prepare_data.py` 就是这么做的）。
- 每轮结束在验证集算一次 loss，用它挑 `best.pt`，而不是凭训练 loss 判断。
- 训练 loss 降、验证 loss 升 = 过拟合，此时继续训练只会让模型更会背答案。
  本项目 25 轮里第 3 轮 val 最低（3.06），之后一路涨到 4.24 —— 这条曲线本身就是教学素材。

---

## 五、四个常见坑

| 坑 | 现象 | 修法 |
|---|---|---|
| pad 位置的 label 填了 pad_id | 模型学会输出一长串 `<pad>` | 统一填 -100 |
| 忘了 `<|end|>` | 回答永不结束 | 加特殊 token 并计入 loss |
| bf16 下直接算 cross_entropy | loss 抖动、偶尔 NaN | logits 先 `.float()` |
| 用验证集统计词表 | 指标虚高 | 只用训练集建词表 |

---

## 六、升级到真实模型（LoRA 路线）

本项目是「从零」，真做业务通常反过来：拿一个预训练好的底座，只调一小部分参数。
PyTorch 环境里装上 `transformers peft datasets` 后，大致是这样：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

name = "Qwen/Qwen2.5-0.5B-Instruct"          # 0.5B 在 8G 显存上够用
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")

# 只训练注意力的 q/v 投影，参数量约 0.1%
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
))

msgs = [{"role": "user", "content": "什么是人工智能？"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
# 关键：只有回答部分算 loss —— 用 tokenizer 得到 prompt 长度后，把它前面的 label 全设成 -100
```

要点与本项目完全一致，只是 `template.py` 换成官方 `apply_chat_template`、
`collate` 换成 `DataCollatorForSeq2Seq`（它内部做的也是 label padding + -100）。
**机制没变，规模变了。**

注意：0.5B 级别底座 + 3000 条数据，回答质量会明显好于本项目，但仍然远达不到「懂知识」——
SFT 教的是「怎么答」，知识来自预训练。

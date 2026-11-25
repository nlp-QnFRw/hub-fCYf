---
title: 人民日报 SFT 的内存参数与截断风险
domain: project
tags: [train_sft, 人民日报, LoRA, CPU, 内存不足, batch_size, grad_accum, max_length, NaN]
code_refs: [src_llm/train_sft.py, parse_args, main, SFTDataset.__getitem__, collate_fn]
updated: 2026-10-27
confidence: high
source: codex-session
---

# 人民日报 SFT 的内存参数与截断风险

## 已验证结论

- 当前脚本在 CUDA 不可用时选择 CPU，没有 MPS 分支；模型强制以 float32 加载。
- 本次环境为 Apple M1 Pro、16 GiB 物理内存；Python 环境 py312，torch 2.6.0、transformers 5.5.3、peft 0.15.0。
- 默认训练 batch_size=4、grad_accum=4、max_length=256；验证 batch_size=训练的两倍。
- --epochs 1 只减少重复遍历次数，不能降低单个 batch 的峰值内存。减小 batch_size 才能降低随批大小增长的激活与输出内存；grad_accum 用于保留每次更新的名义样本数。
- batch_size=1、grad_accum=16 的名义有效 batch 仍为16，但当前代码按微批 token 平均损失再平均，不保证与原 batch=4 的梯度严格一致。
- 未提供 --full_ft 时使用 LoRA，默认 r=8、alpha=16。不建议为内存问题开启全量微调；减少 r 对冻结主干和大词表输出内存的帮助有限。
- --num_train -1 保留全部20864条训练记录。改成2000只能运行子集一轮，不能称为全量数据一轮。

## 内存来源

Qwen2-0.5B-Instruct 本地配置 vocab_size=151936、hidden_size=896、24层。仅 [4,256,151936] 的 float32 logits 就占593.5 MiB，不包括模型权重、激活、loss缓冲和进程开销。这是形状计算，不能当作已测量训练峰值。未获取原始报错栈，尚不能判定实际失败在加载、前向、反向、验证还是系统内存压力。

## 实际数据与 tokenizer 核验

按当前 SYSTEM_PROMPT、apply_chat_template、record_to_target 和本地 tokenizer，实际遍历全部训练集并统计；没有加载模型权重或启动训练。

| max_length | 训练集答案全部截掉 | 训练集答案部分截断 | 前300条验证集答案全部截掉 |
|---|---:|---:|---:|
|128|10839|6427|171|
|192|275|2478|5|
|256|32|407|0|
|320|17|99|0|
|384|15|29|0|
|512|12|14|0|

训练 prompt token 长度中位数128、P95为165、最大621；完整 prompt+target 最大1680。max_length 是总长度，不只是用户文本长度。当前 SFTDataset 先拼接 prompt 与答案，再从右侧整体截断；prompt标签为-100。若 prompt 已占满限制，所有 labels 都为-100。已用当前 PyTorch 的小张量交叉熵复现全 ignore_index=-100 时 loss=NaN；没有有效监督的样本需要过滤或正确切分。

## 建议及边界

降低峰值的首选参数是 --epochs 1 --batch_size 1 --grad_accum 16 --max_length 256 --num_train -1。它是待实测配置，不能保证完成，且原脚本的无有效标签样本问题不能仅靠这些参数解决。

在维持256长度时，可在构建 DataLoader 前筛除无有效目标的样本，并明确报告筛除32条；这是一轮有效样本训练，不是20864条都产生完整监督。若要求每条长记录及JSON都完整参与，需要做与实体标签对齐的切分或其他长序列策略，不能直接截断输入却保留不对应的答案。

若 batch=1 仍超内存，可考虑关闭训练 use_cache、启用梯度检查点、及时释放 outputs 引用，以及单独设置验证 batch。梯度检查点以重算换内存，训练会更慢；目前脚本没有对应 CLI 参数，不应给用户编造 --gradient_checkpointing、--fp16、--device 选项。MPS支持需要改设备选择逻辑，且不会增加物理内存容量。

## 来源与验证限制

- 当前 src_llm/train_sft.py、data/peoples_daily/train.json、validation.json、本地 Qwen 配置与 tokenizer。
- 当前环境 transformers/loss/loss_utils.py 的平均交叉熵实现。
- Hugging Face 梯度累积说明：https://huggingface.co/docs/transformers/grad_accumulation
- Hugging Face 梯度检查点说明：https://huggingface.co/docs/transformers/grad_checkpointing
- 本次没有改动训练源码，没有运行整轮训练；未收到内存报错堆栈。现有 outputs/logs/train_sft.json 及部分旧说明针对早期数据/配置，不能据其证明当前人民日报脚本已成功训练。

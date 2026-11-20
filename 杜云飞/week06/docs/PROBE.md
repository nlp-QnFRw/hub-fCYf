# PROBE：环境实测记录

> 由 `python -m common.probe` 生成。**报告里的耗时/显存只引用这里的数字**，不凭印象。

| 模型 | 加载 | 耗时 | 编码吞吐 |
|---|---|---|---|
| bert-base | ✓ BertTokenizer | 0.0s | 750 docs/s |
| qwen2-0.5b | ✓ Qwen2Tokenizer | 0.2s | chat_template ok (150 字符) |

## GPU 训练/推理
- **BERT 训练**：bs=32 L=256: 272 ms/step, 118 samples/s, 峰值 4.32 GB, 100k×1ep ≈ 14 min
- **LoRA 训练**：bs=2 L=384 r=8: 172 ms/step, 11.6 samples/s, 峰值 3.78 GB, 可训练 1.08M, 10k×3ep ≈ 0.7 h
- **本地生成**：bs=16 prompt=227: 21.4 samples/s, 1000 条 ≈ 47s, 4200 条 ≈ 3.3 min

## 由此得到的硬约束

- BERT：`bs=32 / max_len=256`；显存峰值一旦过 ~7GB，吞吐断崖式下跌（bs=64 会比 bs=32 慢 3 倍）
- LoRA：`bs=2 / max_len=384 / accum=8`；bs≥4 会溢到 WDDM 共享内存，**不报错只是变慢**
- 生成式评估：只跑 `test_bal`（4200 条）与 1000 条抽样，不跑 10k 全量

## 实测 vs 真实训练的偏差（重要）

本探针用定长张量测吞吐，**真实训练要慢 20%~25%**：真实数据长度不均（padding 更多）、
且笔记本满载后热降频（GPU 稳定 77~79°C，功耗 84W→72W）。
实测对照：BERT 100k×3ep 真实单 epoch **1030s**（探针推算 14 min = 840s）；
LoRA 三轮 994s / 1174s / 1223s（逐轮变慢 23%）。**排期时按 +25% 打余量。**

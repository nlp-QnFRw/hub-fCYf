# BERT SFT 问答 Demo

基于 `bert-base-chinese` 微调的问答生成模型。

## 目录结构

```
sft_qa/
├── train.py      # 训练脚本
├── infer.py      # 命令行推理脚本
├── server.py     # Web 服务（浏览器访问）
└── README.md     # 本文件
```

## 前置依赖

- Python 3.8+
- `torch`
- `transformers`

本文件夹已自带模型和数据，独立可运行：
- `./bert-base-chinese/` —— 预训练 BERT 模型
- `./bert-sft-qa/` —— 微调后的问答模型
- `./sft_train/` —— 训练/验证数据

## 使用方式

### 1. 训练模型

```bash
python train.py
```

- 训练 3 个 epoch，batch_size=16
- 模型自动保存到 `../bert-sft-qa/`
- CPU 上约需 90 分钟

### 2. 命令行推理

```bash
python infer.py
```

输入问题后按回车，模型自动生成答案。输入 `exit` 退出。

### 3. Web 服务

```bash
python server.py
```

浏览器打开 `http://localhost:8000/`，在页面输入问题即可看到模型回答。

## 训练数据格式

`sft_train.jsonl` 每行一条 JSON：

```json
{"instruction": "问题内容", "output": "答案内容"}
```

## 实现说明

- 使用 `EncoderDecoderModel` 将 BERT 改造为 Seq2Seq 架构
- Encoder 接收问题，Decoder 自回归生成答案
- Decoder 的交叉注意力层从零训练

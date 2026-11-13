# Week 05 · 基于语言模型的 SFT 训练

用 **Qwen3-0.6B-Base + LoRA** 在课程给的 3000 条中文指令对上做监督微调，把「概率续写器」调成「问答助手」。
完整说明见 [sft_lab/README.md](sft_lab/README.md)，实验报告见 [sft_lab/artifacts/report.html](sft_lab/artifacts/report.html)。

## 结果（RTX 5050 Laptop 8GB，376 步 / 210 秒）

| 指标 | base | LoRA |
|---|---|---|
| answer_ppl | 9.60 | **2.68** |
| 主动收尾率 stop_rate | 6.5% | **98.0%** |
| 长度落区间 | 30.5% | **99.5%** |
| char_f1（随机基线 0.127） | 0.243 | **0.622** |
| 复读率 | 61% | **1%** |

## 目录

```
week05/sft_lab/
├─ common.py              # ChatML 渲染、tokenize+loss mask、字符级指标、LoRA 配置
├─ 00_env_check.py        # 环境体检
├─ 01_download_model.py   # ModelScope 拉 Qwen3-0.6B-Base（断点续传）
├─ 02_prepare_data.py     # 编码 + mask 自检
├─ 03_train.py            # LoRA 训练（--smoke 校准耗时）
├─ 04_eval.py             # 前后对比评估
├─ report/                # 报告构建（模板 + 渲染脚本）
├─ data/  outputs/  artifacts/
└─ models/                # 不入库，跑 01 重新下载
```

## 跑一遍

```bash
cd sft_lab
python 00_env_check.py
python 01_download_model.py
python 02_prepare_data.py
python 03_train.py
python 04_eval.py
python report/build_report.py
```

## 踩到的坑

Qwen3-0.6B-Base 词表里 `<|im_start|>` / `<|im_end|>` 的 embedding 行范数只有 0.35（普通汉字约 0.97，
`<|endoftext|>` 为 1.10），tied embedding 下这两个 token 的 logit 几乎推不动 —— 过拟合 32 条、15 个 epoch 后
`<|im_end|>` 的 loss 仍卡在 7.8。回答终止符换成 base 本来就认识的 `<|endoftext|>` 后，一个 epoch 就学会了。

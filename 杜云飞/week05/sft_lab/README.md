# SFT 练手：把「续写器」调成「问答助手」

Qwen3-0.6B-Base + LoRA + transformers Trainer，在课程 `sft_train.jsonl`（3000 条中文指令对）上做监督微调，
用微调前后的硬指标证明：模型从「接着往下编」变成「被问就答、答完就停」。

## 目录
```
sft_lab/
├─ common.py              # 路径、ChatML、tokenize+loss mask、字符级指标、LoRA 配置
├─ 00_env_check.py        # 环境体检：GPU / bf16 / 模型文件 / 数据 / chat template 探针
├─ 01_download_model.py   # 从 ModelScope 拉 Qwen3-0.6B-Base（断点续传、幂等）
├─ 02_prepare_data.py     # jsonl → tokenized Dataset，并输出 mask 自检
├─ 03_train.py            # LoRA + Trainer（--smoke 校准耗时，--low-mem 降显存）
├─ 04_eval.py             # 微调前后：answer_ppl + 7 个字符级指标 + 10 组对照
├─ report/                # 前端报告构建（与训练代码分开）
│   ├─ build_report.py    # 读 artifacts → 渲染
│   └─ template.html      # 报告模板
├─ models/  data/  outputs/  artifacts/
```

## 跑一遍（约 25 分钟，8GB 显存够用）
```bash
cd sft_lab
python 00_env_check.py          # 全绿再往下
python 01_download_model.py     # 1.19GB，约 4 分钟，断了重跑即可续传
python 02_prepare_data.py       # 产出 data/ 与 artifacts/mask_check.txt
python 03_train.py --smoke      # 20 步，看 s/步 是否可接受
python 03_train.py              # 正式训练 376 步
python 04_eval.py               # 前后各跑一遍，约 3 分钟
python report/build_report.py   # 出 artifacts/report.html 与 report.md
```

## 四个关键点
1. **手写 ChatML**：官方 chat_template 会往 assistant 段塞 `<think>\n\n</think>\n\n`，Base 模型没有思考行为，学了是噪声。
2. **Loss Mask**：提问段 `labels = -100`，loss 只落在回答段；截断时砍提问段左侧，保住 `assistant\n` 角色头。
3. **终止符必须用 Base 认识的 token**：第一版回答以 `<|im_end|>` 结尾，训练后 stop_rate 仍是 0。
   查 embedding 发现它在 Base 词表里几乎是空行（行范数 0.35，普通汉字约 0.97），tied embedding 下 logit 推不动；
   改成 `<|endoftext|>`（范数 1.10）后一个 epoch 就学会了。SFT 教不了模型没见过的符号。
4. **生成必须双 eos**：`eos_token_id=[151645, 151643]`。tokenizer 的 eos 其实是 `151643`，只给 `<|im_end|>` 会一直编到 64 token。

## 实测结果（RTX 5050 Laptop 8GB，376 步 / 210 秒）
| 指标 | base | LoRA | 说明 |
|---|---|---|---|
| answer_ppl | 9.60 | **2.68** | 回答段困惑度 |
| stop_rate | 6.5% | **98.0%** | 会自己收尾 |
| len_in_range | 30.5% | **99.5%** | 长度落进数据分布 |
| char_f1 | 0.243 | **0.622** | 随机基线 0.127 |
| repeat_rate | 61.0% | **1.0%** | 复读退化 |

Base 的典型输出是把提问原样复述一遍（续写器行为），LoRA 之后是短答 + 停止。

## 结论口径
不宣称「模型更聪明」：0.6B + 3000 条短答案，学到的是**输出格式、长度分布、停止行为**。
个别算术题仍会算错（如 246×38 答成 9048），这恰好说明「知识来自预训练，形式来自 SFT」。

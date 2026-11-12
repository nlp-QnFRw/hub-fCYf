# 字符级 Transformer 的 SFT 微调

复用 `../week04/language_model.py` 中的 `LM` 类，从 `../week04/best_model_trans.pt` 继承权重。原训练脚本和原模型均不修改。

## 1. 生成 tagged 数据

在仓库根目录使用 PowerShell 执行：

```powershell
conda run -n py312 --no-capture-output python -B "肖祥龙/week05/prepare_sft_data.py"
```

原始的 `sft_train.jsonl`、`sft_val.jsonl` 保留，另存 `sft_train_tagged.jsonl`、`sft_val_tagged.jsonl`。每条只有 `text` 字段：

```json
{"text":"<|user|>问题内容<|assistant|>回答内容<|end|>"}
```

转换会检查条数、标识格式和逐条还原结果。指令或答案中的空格、标点、换行保持原样，重复运行不会重复叠加标识。

## 2. 开始训练

```powershell
conda run -n py312 --no-capture-output python -B "肖祥龙/week05/sft_train.py"
```

训练脚本直接读取上述两份 **tagged 文件的 text 字段**。三个角色标识各编码为一个独立 token，并增加 `<|pad|>`、`<|unk|>`。原词表编号及参数保留，新正文字符仅从训练集提取；验证集未登录字符映射到 `<|unk|>`，数量记入日志。

上下文从 64 扩至 128：原前 64 个位置向量不变，新增位置复制原位置向量初始化。全参数微调，按批次实际最大长度在右侧补齐。仅对回答及 `<|end|>` 计算下一 token 损失，提示词和补齐位置不计入损失。

默认参数：AdamW，学习率 `1e-4`，weight decay `0.01`，batch size `16`，dropout `0.1`，梯度裁剪 `1.0`，随机种子 `42`。最多训练 10 轮，连续 3 轮验证损失未改善时提前停止。本机 `py312` 安装的是 CPU 版 PyTorch，默认使用 CPU。

最佳**已训练轮次**按回答 token 加权的验证损失选择，保存为 `best_model_sft.pt`。`sft_training_log.json` 包含实际输入文件及其 SHA-256、数据统计、微调前基线、各轮指标、最佳轮次和训练前后生成示例。训练结束会重载磁盘上的最佳模型，再次验证损失。

### 本次训练结果

使用上述默认配置完成 10 轮，最佳轮次为第 10 轮，总耗时约 148.7 秒。

| 同一份 tagged 验证数据 | 回答 Loss | 回答 PPL |
| --- | ---: | ---: |
| 扩展词表和位置编码后、尚未微调 | 5.0562 | 156.99 |
| 第 10 轮最佳模型 | 3.1670 | 23.74 |
| 从磁盘重新加载最佳模型 | 3.1670 | 23.74 |

词表从 4,000 扩到 4,096（91 个训练集新字符及 5 个特殊 token）。训练集没有未知字符；验证集指令和回答分别有 1、4 个未知字符。实际最大输入长度分别为 113、107 个 token，没有截断样本。

六项单元测试全部通过，tagged 数据与原始数据逐条一致，训练日志记录的输入文件哈希也已核对。但生成示例仍出现重复、答非所问和达到长度上限的情况，例如解释 NLP 的问题生成了重复字母。因此当前结果仅能确认 SFT 流程和验证损失改善，尚未达到可靠的指令问答效果。训练前后三组验证问题及实际输出见训练日志。

可覆盖默认参数，例如：

```powershell
conda run -n py312 --no-capture-output python -B "肖祥龙/week05/sft_train.py" --epochs 10 --batch_size 16 --lr 0.0001 --device cpu
```

默认输入输出路径均相对于脚本目录解析，从其他目录运行也可找到资源。若另行指定参数中的相对路径，则按命令的当前目录解析。

## 3. 使用微调模型

### 手动交互问答

和 week04 的 `pinyin_ime.py` 一样，可以直接启动程序，连续输入问题：

```powershell
conda run -n py312 --no-capture-output python -B "肖祥龙/week05/sft_qa.py"
```

在 IDE 中选择 `py312` 解释器，直接运行 `sft_qa.py` 也可以，无需填写启动参数。默认加载该脚本目录下的 `best_model_sft.pt`。

出现 `问题>` 后输入自己的问题，程序显示 `回答>`、耗时和是否正常结束。空行跳过，输入 `q`、`quit` 或 `exit` 退出。每个问题独立作答，与当前单轮 SFT 数据格式一致，不携带历史对话；当前最多输入 126 个字符。

可通过 `--max_new_tokens 128` 增大回答上限，通过 `--model_path` 指定其他兼容的 SFT 检查点。生成仍使用原有的贪心解码，程序会如实展示模型输出。

### 单次命令行问答

```powershell
conda run -n py312 --no-capture-output python -B "肖祥龙/week05/sft_predict.py" --prompt "解释以下缩写的含义：NLP。"
```

脚本自动使用 `<|user|>问题<|assistant|>` 模板，贪心生成，遇到 `<|end|>` 停止。默认最多生成 64 个 token；若未生成结束标识，会提示已达到长度上限。问题包含角色标识后不能超过 128 个 token；生成过程中只保留最近 128 个 token 作为上下文。

## 4. 验证

```powershell
conda run -n py312 --no-capture-output python -B -m unittest discover -s "肖祥龙/week05" -p "test_sft.py" -v
```

覆盖数据转换及还原、重复转换、非法标识、独立特殊 token、回答损失的移位边界、动态补齐、原参数继承、验证集词表隔离、反向传播、检查点重载和生成结束行为。

模型约两百万参数，且现有数据中有错误答案。验证困惑度下降仅表示对当前数据的预测改善，不保证事实正确、算术能力或通用问答能力。

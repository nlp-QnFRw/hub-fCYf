# 金融公告事件抽取：BERT 与 LLM 实践

这是参照同级老师项目 `../code/`，为学习事件抽取独立编写的项目。输入金融公告，选择 **BERT + Linear、BERT + CRF、LLM zero-shot 或 LLM few-shot**，得到事件类型、触发词和角色论元。另有 BIO 可视化页面，帮助理解训练标签、token 对齐和模型预测。

> 项目用于学习实验，不是完整的生产系统。当前 BERT 主要基于句级单事件样本训练，对无事件文本和多事件长文仍需人工核对。

## 1. 先从哪里开始

| 目的 | 查看或操作 |
|---|---|
| 启动网页体验已有模型 | 阅读下方第 3 节 |
| 学习每个字的 BIO 标签 | 浏览器打开 `http://127.0.0.1:8000/bio`，见第 4 节 |
| 重新训练 BERT | 阅读第 6 节；同名模型会被覆盖 |
| 用 DeepSeek 抽取 | 阅读第 5 节并填写本地配置 |
| 理解原理和方法选择 | 阅读 [学习路线与方法选择指南](学习路线与方法选择指南.md) |
| 了解模块实现 | 阅读 [BERT](bert/README.md)、[公共数据](common/README.md)、[评测](eval/README.md)、[LLM](llm_zero_shot/README.md)、[服务](service/README.md) |

## 2. 项目结构

```text
diyCode/
├── bert/                       # 训练和推理
│   ├── dataset.py              # 字符 span → BIO → token 标签
│   ├── model.py                # BERT、Linear、CRF
│   ├── train.py                # 训练、内部验证和保存 checkpoint
│   ├── predict.py              # 加载模型、分段、解码和校验
│   └── bio_visualize.py        # BIO 教学页面的数据
├── common/event_data.py        # 数据路径、schema、标签表、切分、校验
├── llm_zero_shot/llm_extract.py # LLM 配置、提示词、请求与解析
├── eval/                       # 模型评测和错误分析
├── service/
│   ├── app.py                  # 本地 HTTP 服务
│   └── static/                 # 事件抽取页、BIO 教学页
├── outputs/                    # 本地生成的模型、日志和实验产物
├── llm_config.json             # 本地密钥配置，已忽略 Git
├── llm_config.example.json     # 无密钥的配置模板
├── requirements.txt
└── 学习路线与方法选择指南.md
```

默认读取老师目录的 `../code/data/processed/` 和 `../code/pretrain_models/bert-base-chinese/`。`data/sample/` 是快速冒烟数据。checkpoint 保存了预训练模型的绝对路径，迁移项目后需要确认该路径仍存在；本项目不是脱离老师目录即可运行的独立数据包。

## 3. 在 Git Bash 启动网页

以下命令以这台机器现有的 Python 环境为例。先进入项目目录，之后本文件的命令都从此目录执行：

```bash
cd "/e/project/py-project/badou/nlp-study/王振飞/week07/diyCode"
/d/miniconda3/envs/py312/python.exe -m service.app
```

浏览器访问：

- 事件抽取：`http://127.0.0.1:8000/`
- BIO 教学：`http://127.0.0.1:8000/bio`

服务只监听本机 `127.0.0.1:8000`；保持 Git Bash 打开，按 `Ctrl+C` 停止。BERT 首次使用时加载并缓存模型，默认有 CUDA 时使用 GPU，否则用 CPU；训练和命令行评测可显式指定 `--device cuda`。

页面选择 Linear 或 CRF 前，必须先训练对应模型。LLM 需要先配置 API。改 Python 代码后要重启服务；改网页后刷新浏览器，必要时按 `Ctrl+F5`。只改 `llm_config.json` 时，下次 LLM 请求会重新读取。

## 4. BIO 页面怎么看

页面分成两种模式，二者都需要本地 Linear checkpoint：

- **训练：真实标注**：从 `train_extract.jsonl` 选短样本，先显示字符金标，再展示 `EventDataset` 实际生成的 token 训练目标。第四步对比金标与当前已训练模型的预测，绿色一致、红色不同。它是实时推理，不是过去每轮训练的录像，也不会更新权重。
- **推理：输入句子**：输入自定义文本，查看 BERT + Linear 原始 token 预测及投影到字符的标签。第四步右侧可查看最终事件 JSON，也可切回 token 表。

`B` 表示片段开头，`I` 表示片段内部，`O` 表示不属于标注片段；`TRG` 是触发词，`ARG` 是角色论元。模型实际按 token 预测：一个 token 覆盖多个字符时，字符格只是展示投影；没有 token 的空格不应被当作模型预测的 `O`。Linear 训练中特殊 token 标签为 `-100`，不参与交叉熵损失。

教学页左侧只显示截断到 `max_length` 的一次原始预测；右侧 JSON 使用首页相同的**分段、解码、校验**流程。长文两侧的上下文和覆盖范围不同，不能逐字符一一对应。token 的最大 softmax 分数也不是完整事件正确的概率。

## 5. 配置 DeepSeek / 兼容 API

在项目根目录打开 `llm_config.json`，将 Key 填入 `api_key`：

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-flash",
  "api_key": "填写自己的 API Key"
}
```

若文件不存在，可复制 `llm_config.example.json` 为 `llm_config.json` 后填写。`base_url` 是 API 根地址，不带 `/chat/completions`；代码会自动追加该路径。配置值以服务商当前提供的信息为准。

配置文件中的**非空字段优先**；字段为空时回退到对应的 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 环境变量。`llm_config.json` 已由 `.gitignore` 忽略，但 Key 仍以明文保存在本机，不要分享该文件或将密钥粘贴到聊天中。API 会接收输入文本，可能产生费用。

Zero-shot 只提供任务说明和 schema；Few-shot 额外加入 `llm_zero_shot/llm_extract.py` 中 `EXAMPLES` 的一个固定“股份回购”示例。两者都不训练模型，也不保证 Few-shot 一定更好。

## 6. 训练 BERT

### 数据与标签

老师的派生数据 `../code/data/processed/train_extract.jsonl` 每行是一句文本和一个事件，`event_schema.json` 定义 13 种事件和 61 个去重角色。联合 BIO 标签数为 `1 + 2 × (13 + 61) = 149`。位置 `start/end` 是 Python 原文字符下标，左闭右开，即 `text[start:end]` 应等于标注文本。

例如，`甲公司拟回购股份。` 中的 `回购` 在 `[4, 6)`，可以标成 `B-TRG:股份回购`、`I-TRG:股份回购`。实际样本可能有多个论元；单层 BIO 无法同时表达相互重叠的 span，当前实现先画论元，再画触发词。

### 完整数据训练

先用 `nvidia-smi` 检查显卡空闲显存，再执行下列命令之一。`--device cuda` 明确要求 GPU，CUDA 不可用时会报错，不会自动回退 CPU：

```bash
# BERT + Linear
/d/miniconda3/envs/py312/python.exe -u -m bert.train \
  --decoder linear --batch-size 2 --grad-accum 4 --max-length 256 --device cuda

# BERT + CRF
/d/miniconda3/envs/py312/python.exe -u -m bert.train \
  --decoder crf --batch-size 2 --grad-accum 4 --max-length 256 --device cuda
```

默认 3 个 epoch、学习率 `2e-5`、随机种子 42。训练集随机划出 10% 作内部验证，按最低验证 loss 保存最佳 checkpoint；dev 不参与选择。显存不足可减小 `--batch-size`，梯度累积通过 `--grad-accum` 控制；输入过长会按 `--max-length` 截断。

快速检查流程可指定 `--data-dir ../code/data/sample --epochs 1`，但 sample 结果不能当正式指标。**快速训练和正式训练会写入同名文件**：若已有需要保留的模型，请先备份。

### 产物

```text
outputs/bert_linear.pt             Linear 最佳模型
outputs/bert_linear_history.json   Linear 每轮训练/验证 loss
outputs/bert_crf.pt                CRF 最佳模型
outputs/bert_crf_history.json      CRF 每轮训练/验证 loss
```

同一种方法再次运行训练会覆盖对应 checkpoint 和日志；两种方法互不覆盖。模型保存了权重、标签表、最大长度和预训练路径。`outputs/` 被 Git 忽略，请自行备份需要保存的结果。

## 7. 评测与已有实验

```bash
/d/miniconda3/envs/py312/python.exe -m eval.evaluate --decoder linear --device cuda
/d/miniconda3/envs/py312/python.exe -m eval.evaluate --decoder crf --device cuda
```

默认使用 930 条 `dev_extract.jsonl`；调试时可加 `--data-dir ../code/data/sample --limit 20`。脚本只将 JSON **打印到终端**，不会自动保存；在 Git Bash 可通过 `> outputs/linear_eval.json` 保存标准输出。

当前评测逐样本使用集合进行精确匹配：触发词按事件类型和文本，论元按事件类型、角色和文本，完整事件要求触发词和论元集合都相符。P、R、F1 分别反映预测精确程度、金标覆盖程度及两者的调和平均。集合会去重；多片段预测的论元会按事件类型汇集，不能当作完善的篇章事件关联评测。

本机一次实验，使用相同 930 条、相同当前解码与评测代码：

| 方法 | 触发词 F1 | 论元 F1 | 完整事件 F1 |
|---|---:|---:|---:|
| Linear | 89.86% | 70.56% | 27.66% |
| CRF | 90.23% | 70.38% | 26.61% |

这是当前 checkpoint 和数据上的结果，不能概括所有模型；自己的简化评测与老师实现不同，不能直接拿来比较。已有 Linear 的错误分析在本地 `outputs/bert_linear_错误分析.md`；空格解码及检测集实验在 `outputs/bert_linear_解码修复及负样本评估.md`。这些本地产物不随 Git 分发。

更多分析命令与限制见 [eval/README.md](eval/README.md)。特别是：正样本抽取集上的 F1 不代表无事件文本上的误报能力。之前用 clean 检测集直接测试 Linear，按数据标注得到 65.82% 的负样本误报率；这不是人工复核后的线上误报率。

## 8. 抽取接口和结果

`POST /api/extract` 接受 JSON：

```json
{"text": "甲公司拟回购股份。", "method": "bert_linear"}
```

`method` 可选 `bert_linear`、`bert_crf`、`llm_zero`、`llm_few`。返回结构示意：

```json
{
  "method": "bert_linear",
  "events": [{
    "type": "股份回购",
    "trigger": {"text": "回购", "start": 4, "end": 6},
    "arguments": [{"role": "回购方", "text": "甲公司", "start": 0, "end": 3}]
  }],
  "needs_review": false,
  "review_reasons": []
}
```

示例只说明数据格式，不保证模型会得到相同结果。无事件时 `events` 为空并提示复核；`needs_review` 由当前规则和警告决定，并非经过校准的正确率。角色不在 schema、片段无法定位等可能被过滤并产生提示。`POST /api/bio` 用于教学页，详见 [service/README.md](service/README.md)。

## 9. 常见问题

| 现象 | 检查方法 |
|---|---|
| 提示尚未训练模型 | 查看 `outputs/bert_linear.pt` 或 `bert_crf.pt` 是否存在，训练与推理 decoder 是否一致 |
| 模型路径不存在 | checkpoint 记录训练时的预训练模型绝对路径；检查对应文件是否仍存在 |
| CUDA 不可用或显存不足 | 用 `nvidia-smi` 检查驱动和空闲显存；确认当前 Python 的 PyTorch 支持 CUDA；减少 batch size |
| 端口 8000 被占用 | 停止已经启动的旧服务，再重新启动 |
| 网页还是旧版或出现未知接口 | 重启 Python 服务，并在浏览器按 `Ctrl+F5` |
| LLM 返回 401 | 检查配置文件的 Key、地址及服务商；注意文件非空字段优先于环境变量，不要公开密钥 |
| LLM JSON 解析失败 | 这是模型响应解析错误，不代表“没有事件”；重试并检查服务端提示 |
| 结果需要人工复核 | 查看 `review_reasons`；为空结果、非法角色或无法定位的片段等情况会触发提示 |

## 10. 当前边界

- 没有训练独立事件检测器，没有规则补全、置信度校准或 LoRA；不能保证无事件输入不误报。
- 训练样本是句级单事件，单层 BIO 难以表达重叠；长文按无重叠片段抽取，每个片段只取第一个触发词，跨片段论元和多个真实事件可能遗漏或错配。
- token 标签取原文起点字符的标签，长内容在训练时截断。相同 I 标签允许跨纯空白合并，可修复一些边界，但也可能错误合并。
- 生成式结果会检查 schema 和原文片段；同一文本出现多次且未提供可靠位置时取首次并提示歧义。校验无法保证语义正确。
- 数据及预训练权重来自老师目录，请遵守其原有许可与使用限制。

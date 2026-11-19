# USAGE_GUIDE · 怎么跑、怎么调、怎么排错

## 一、环境准备

```bash
pip install torch transformers peft scikit-learn matplotlib tqdm openai
```

- Python 3.10+；有 GPU 更好（BERT 训练和大模型推理），CPU 也能跑通规则 / TF-IDF / BERT 推理。
- **本地模型**：`pretrain_models/bert-base-chinese`（约 390 MB）、`pretrain_models/Qwen3-0.6B`（约 1.2 GB）。
  `common/model_paths.py` 会从脚本所在目录**向上逐级**查找 `pretrain_models/<模型名>`，
  找不到会直接报错并列出尝试过的路径——不会偷偷联网下载（课堂离线可用）。
- **本地路线不需要 Key**；外部 API 那条路需要（`openai` SDK 已在依赖里），用环境变量配置，见 Step 6b。

自检（确认模型能被找到）：

```bash
python common/model_paths.py
# Qwen3-0.6B           -> D:\badou\八斗课程\pretrain_models\Qwen3-0.6B
# bert-base-chinese    -> D:\badou\八斗课程\pretrain_models\bert-base-chinese
```

## 二、数据

| 文件 | 规模 | 说明 |
|---|---|---|
| `data/train.json` | 53,360 条 | `{"idx","sentence","label"}`，label 是 0–14 |
| `data/val.json` | 10,000 条 | 同上；**其中 1,099 条与训练集重复** |

所有评测默认走 `common/data.py` 的 `load_clean_val()`：先去重、再按 `seed=42` 抽样。
想用「含重复的全量 val」对照，加 `--leak`（仅 TF-IDF 脚本提供）。

---

## 三、逐步运行

### Step 1　规则 / 关键词（秒级）

```bash
python rule/classify_rule.py --n 500
```

内部流程：读干净验证集 → 标题里找 15 个类别名 → 找不到就猜训练集最多的类 → 算指标。

预期输出：

```
样本 500 条（干净集，重复已剔除 1099 条）
命中率 6.60%
| 规则/关键词 | 0.126 | 0.0716 | 0 |
```

> 命中率只有 6.6%，说明这类任务规则法上限就这么高——这是全项目所有方法的「下限」。

### Step 2　TF-IDF + LinearSVC（20 秒）

```bash
python tfidf/classify_tfidf.py --n 500
python tfidf/classify_tfidf.py --n 500 --ngram 1,3 --C 0.5     # 调参对比
python tfidf/classify_tfidf.py --leak                          # 在全量 val 上跑，看泄漏虚高
```

> **别忘了它的定位**：TF-IDF 在**数据封闭**（内网 / 无预训练权重 / 合规 / 自有黑话语料）场景是首选；
> 通用短文本上它只是基线（0.472 vs BERT 0.544）。而且训练数据越少，它与微调的差距越大
> （训练集 15% 时差距从 2.6 拉到 5.7 个点，见 `ARCHITECTURE.md` 的「数据量实验」）。

预期输出（干净集）：`TF-IDF+SVM | 0.512 | 0.4721 | 0`
预期输出（`--leak`，全量 val 10,000 条）：`0.5615 / 0.5407` —— **比干净集高 3 个点，就是那 11% 重复贡献的**。

### Step 3　BERT：训练（GPU 约 14 分钟）

```bash
python bert/train_bert.py                            # 默认 cls 池化、3 epoch、2e-5
python bert/train_bert.py --pool max --epochs 5      # 换池化 / 加轮数
python bert/train_bert.py --no_class_weight          # 关掉类别加权做对照
```

内部流程：分层学习率（BERT 2e-5 / 头 ×5）→ Warmup 10% + 线性衰减 → 类别加权 CE → AMP 混合精度
→ 每个 epoch 在 val 上算 Macro-F1 → **按 Macro-F1 存最优** 到 `outputs/bert_ckpt.pt`。

#### 三个「训练技巧」开关（对应课件 P22–P23）

```bash
python bert/train_bert.py --label_smoothing 0.1      # 软标签（常取 0.05~0.1）
python bert/train_bert.py --focal_gamma 2            # Focal Loss（与类别加权共用同一张 alpha 表）
python bert/train_bert.py --fgm                      # FGM 对抗训练（训练时间约 ×1.5）
python bert/train_bert.py --fgm --fgm_eps 0.5        # 扰动强度默认 1.0，可调小
```

**这三个在本任务上都不涨**（同一批 8,901 条去重验证集、同 seed、3 epoch）：

| 配置 | Macro-F1 | 相对基线 | 备注 |
|---|---|---|---|
| 类别加权（默认） | 0.5344 | — | 唯一有业务意义的：长尾类 recall 从「全错」回到「能中一部分」 |
| + Label Smoothing 0.1 | 0.5272 | **−0.7** | 15 类软标签把有限的信号又稀释了 |
| + Focal Loss γ=2 | 0.5345 | ±0 | 难样本主要来自标签口径重叠，不是「学不会」 |
| + FGM ε=1.0 | 0.5346 | **+0.02** | 几乎不动，训练时间还 ×1.5 |

→ 讲法：这些旋钮解决的是「类别极不均衡（>100× 不均衡）」「数据少到明显过拟合」两类问题；
这个任务两个都不占。**先跑基线、看混淆矩阵，再决定要不要动 loss。**

#### 冒烟模式（不跑全程，验证代码能通）

```bash
python bert/train_bert.py --epochs 1 --max_steps 3 --max_eval 300 --no_save
# [cfg] pool=cls loss=CE class_weight=on fgm=off epochs=1 steps/epoch=3 eval_n=300 save=off
```

`--max_steps` 每个 epoch 只跑 N 步、`--max_eval` 只评 N 条、`--no_save` 不落盘（省 390 MB）。
启动时会打印一行 `[cfg]`，一眼看出「池化 / 损失 / 是否加权 / 是否 FGM / 步数 / 是否落盘」。

预期：`epoch 1: val_acc≈0.56 val_f1≈0.53`，`epoch 2~3` 小幅波动，最佳约 `0.55 / 0.53`。

### Step 4　BERT：推理与评测

```bash
python bert/predict_bert.py --n 500
python bert/predict_bert.py --ckpt outputs/bert_ckpt.pt --n 500
```

预期输出：`评测 500 条（干净集）| 138 ms/条（batch32）| ECE=0.14`
`| BERT 微调 | 0.556 | 0.5436 | 0 |`

> 单条推理约 265 ms，batch=32 时降到 138 ms/条（本机 16 线程 CPU；GPU 快一个量级）。

### Step 5　Qwen3 零样本（GPU 约 5 分钟 / 500 条）

```bash
python llm_zero_shot/classify_zero_shot.py --n 500
```

预期输出：

```
| Qwen3 zero-shot | 0.1683 | 0.105 | 1 |
预测覆盖类别数：8/15
```

> 8/15 就是「预测塌缩」：15 个类里模型只吐过 8 个，绝大多数是「文化」。
> 这不是代码 bug，是 0.6B 模型在 15 类上的真实能力边界。

### Step 6　Qwen3 + LoRA SFT（GPU 约 44 分钟 / 5000×2 epoch）

```bash
python llm_lora/train_lora.py --num_train 5000 --epochs 2
python llm_lora/train_lora.py --num_train 1000 --epochs 1    # 课堂快速版（约 10 分钟）
python llm_lora/train_lora.py --max_steps 20                 # 只冒烟测流程
```

预期输出（训练）：

```
trainable params: 2,293,760 || all params: 598,343,680 || trainable%: 0.3834
epoch 1: train_loss=0.78 val_loss=0.69
epoch 2: train_loss=0.63 val_loss=0.68
adapter → outputs/lora_adapter
```


### Step 6b　全量微调（对比 LoRA）

```bash
python llm_lora/train_lora.py --full_ft --num_train 3000 --epochs 2 --batch_size 2 --grad_accum 8 --gradient_checkpointing
```

- 8G 显存卡**必须**加 `--gradient_checkpointing`，否则 OOM；
- 全量微调用 bf16 + lr 2e-5（脚本已自动切换），LoRA 用 fp16 + 2e-4；
- 产物是完整权重 `outputs/full_ft_ckpt/`（约 1.2 GB），不是 adapter。

预期：`[full-ft] 参数量 596.0M（全部可训练），lr=2e-05`，两个 epoch 的 val_loss 0.693 → 0.666，峰值显存 5.58 GB。

评估（脚本会自动识别「adapter 目录」还是「全量权重目录」）：

```bash
python llm_lora/eval_lora.py --adapter outputs/full_ft_ckpt --n 500 --out outputs/full_ft_pred.json
# | Qwen3 + full_ft | 0.536 | 0.4919 | 0 |
```

> 对比 LoRA：0.484 → 0.492，只多 0.8 个点，但显存、时间、产物都贵一个量级。

### Step 6c　外部 API 零样本（不训练，走网络）

```bash
# 只需要一个 Key；DeepSeek 官网申请后 export
export LLM_API_KEY=sk-xxxx                 # Windows: set LLM_API_KEY=sk-xxxx
export LLM_BASE_URL=https://api.deepseek.com   # 默认值，可省略
export LLM_MODEL=deepseek-v4-flash             # 默认值，可省略

python llm_api_zero_shot/classify_api_zero_shot.py --n 500 --workers 8
```

- 代码是标准 OpenAI 兼容写法（`OpenAI(base_url=..., api_key=...)`），换成任何兼容服务只改环境变量；
- 某些网关需要额外 header，用 `LLM_EXTRA_HEADERS='{"X-Custom-Header":"value"}'` 传（可选，代码不绑定具体网关）；
- DeepSeek 是推理模型：`LLM_DISABLE_THINKING=1` 可关掉思考（省 token、降延迟）；不关也不会错，但 `max_tokens` 要给足。

预期：

```
模型 deepseek-v4-flash｜500 条｜490.8s（0.98s/条）｜tokens in/out = 39705/999
| API zero-shot（deepseek-v4-flash） | 0.5161 | 0.4708 | 2 |
预测覆盖类别数：15/15
```

> 关键对比：同样是零样本，本地 0.6B 只有 0.105（塌缩到 8/15 类），API 大模型 0.471 —— 但仍低于 BERT 微调 0.544。

### Step 7　LoRA / 全量 推理与评测

```bash
python llm_lora/eval_lora.py --n 500                      # LoRA
python llm_lora/eval_lora.py --adapter outputs/full_ft_ckpt --n 500 --out outputs/full_ft_pred.json   # 全量
```

预期输出：

```
| Qwen3 + LoRA SFT | 0.53 | 0.4844 | 0 |
预测覆盖类别数：14/15
```

### Step 8　汇总成一张表

```bash
python eval/eval_all.py            # 汇总已有结果
python eval/eval_all.py --run-light # 顺便重跑规则 + TF-IDF
```

输出 `outputs/summary.md`，包含五个方法的准确率 / Macro-F1 / 样本数 / 覆盖类别 / 解析失败。
脚本会自动检查各路结果的 `n_eval` 与 `seed` 是否一致，不一致会告警——**这是「数字不可比」最常见的原因**。

### Step 9　组装成服务，看早报长什么样

```bash
python service/classify_news.py --text "贵州茅台：2023 年年度业绩预告"
python service/classify_news.py --file my_titles.txt
python service/demo_early_report.py
```

单条输出（JSON 契约）：

```json
{"title":"贵州茅台：2023 年年度业绩预告","topic":"金融","sub_topic":"财经",
 "relevant":true,"conf":0.99,"queue":"research","model":"rule","version":"2026-11"}
```

`demo_early_report.py` 会喂 20 条模拟消息，按 research / review / drop 三个队列分组打印——
这就是「早报面板」的数据来源。

---


### Step 10　分流服务（CLI）：输入标题 → 输出接口契约 JSON

模型训好之后，把它包成一个能一直被调用的服务（启动一次、之后每条标题都是毫秒级）。
返回字段与课件 P43 那张「接口契约」一一对应，下游拿 conf 和 queue 就能直接做路由。

```bash
python service/classify_cli.py                                    # 交互模式：输入标题回车，q 退出
python service/classify_cli.py --text "贵州茅台：2023 年年度业绩预告"
python service/classify_cli.py --file inputs.txt --json           # 批量，一条一行 JSON
type inputs.txt | python service/classify_cli.py - --json         # 从管道读
python service/classify_cli.py --model tfidf                      # 强制走 TF-IDF（不依赖 ckpt）
python service/classify_cli.py --use-rules                        # 打开规则快筛，演示 L5 级联
python service/classify_cli.py --conf 0.6 --verbose               # 改阈值 / 额外输出调试字段
```

输出示例（`--json` 时一行一条，方便接 Agent 或脚本）：

```json
{
  "title": "贵州茅台：2023年年度业绩预告，净利润同比增长19%",
  "topic": "金融",
  "sub_topic": "财经",
  "relevant": true,
  "conf": 0.7714,
  "queue": "review",
  "model": "tfidf-sgd-v1",
  "version": "2026-11"
}
```

| 字段 | 说明 |
|---|---|
| `topic` | 4 组主题（金融 / 产业 / 宏观 / 噪音），中文名，给业务看 |
| `sub_topic` | 15 类细标签，给统计用 |
| `relevant` | 是否与投研相关（由主题组派生） |
| `conf` | 置信度，**已做温度校准**（见下） |
| `queue` | `research` / `review` / `drop`，枚举，业务按这个路由 |
| `model` | 这次走的是哪条路径：`bert-cls-v1` / `tfidf-sgd-v1` / `rule-v1` / `fallback-v0` |
| `version` | 接口版本，出问题能追溯 |

两条工程约定：

1. **降级是显式的**：有 `outputs/bert_ckpt.pt` 就走 BERT；没有就自动退回 TF-IDF（启动日志会打印一行 `[init] TF-IDF 后端就绪…`），结果里的 `model` 字段就是证据。调用方永远知道这是不是降级结果。
2. **置信度必须校准才能当阈值用**：原始概率明显偏保守（SGD 的 max-prob 中位数只有 0.42），直接拿 0.8 当阈值时只有 **2.6%** 的验证样本能进 research 队列 —— 接口看着能跑，实际废掉。所以服务启动时会用**留出集做温度缩放**（TF-IDF 用 train 的 10%，BERT 用干净验证集 200 条），拟合出的温度会打印出来（实测 TF-IDF T≈0.75）。这就是课件里说的「温度缩放」。

启动耗时与推理开销：TF-IDF 后端约 **10 秒**（现场拟合词表 + 线性模型）；BERT 后端约 **40 秒**（加载 390 MB 权重）。单条推理都在 20 ms 以内。

## 四、作为模块调用

```python
import sys
sys.path.insert(0, r"D:\badou\八斗课程\week6-数据接入与分流\code")
from service.classify_news import ShuntService

svc = ShuntService(conf_threshold=0.8)          # 模型只加载一次，之后复用
for title in ["宁德时代拟回购不超过 50 亿元公司股份", "国足世预赛客场 1:3 失利"]:
    print(svc.classify(title))
```

> 第 9 周会把 `svc.classify` 直接包成 Agent 的工具（Function Calling / MCP / CLI 三种接法），
> 所以这个函数签名就是对外的「接口契约」，字段不要随便改。

---

## 五、FAQ

**Q1：`未找到本地模型 Qwen3-0.6B`**
A：确认 `八斗课程/pretrain_models/Qwen3-0.6B/` 下有 `*.safetensors`。目录在但没权重（半下载）也会报这个错。

**Q2：没有 GPU 能跑吗？**
A：规则、TF-IDF、BERT 推理都能在 CPU 跑（BERT 500 条约 1 分钟）。BERT 训练和大模型推理建议用 GPU；
`--max_steps 20` 可以在 CPU 上冒烟验证流程。

**Q3：`bert/predict_bert.py` 报找不到 ckpt？**
A：先跑 `bert/train_bert.py`（GPU 约 14 分钟 / 3 epoch），它会在 `outputs/` 下生成 `bert_ckpt.pt`（约 390 MB）。
权重没有随课件打包（压缩包会大 400 MB），跑一次就有；也可以 `--ckpt` 指向别处的权重。

**Q4：跑 `service/classify_cli.py` 提示降级到 TF-IDF？**
A：说明 `outputs/bert_ckpt.pt` 不存在（本目录不带权重，压缩包会大 400 MB）。
先跑 `bert/train_bert.py`（GPU 约 14 分钟）就有了；不训也能用，只是走 TF-IDF 后端。
想强制指定后端：`--model tfidf` / `--model bert`。

**Q4b：zero-shot 只有 0.17 准确率，是不是我跑错了？**
A：不是。0.6B 模型在 15 类中文新闻上就是这个水平（随机基线 0.067）。
它会「塌缩」到少数几类——用 `--n 500` 时看「预测覆盖类别数」那一行确认。

**Q5：SFT 后还是不如 BERT，那还有必要学 LoRA 吗？**
A：有。这个任务的标签是封闭的，判别式天然占优；换成抽取、开放问答、要私有化的场景，结论会反过来（第 7、10 周会再验证）。

**Q6：我的数字和 PPT 上差 1–2 个点？**
A：先确认 `--n` 和 `seed` 一致；再看是「全量 val」还是「去重后」。这两个口径差 3 个点左右很正常。

**Q7：API 返回空串 / 指标突然很高？**
A：推理模型先把 `max_tokens` 消耗在思考 token 上，答案被截断成空串（实测 500 条里 85 条）。把 `max_tokens` 提到 256，或设 `LLM_DISABLE_THINKING=1` 关掉思考；另外注意解析失败的样本要计入准确率分母。

**Q8：`trainable% = 0.0`？**
A：`target_modules` 名字写错（Qwen3 是 `q_proj/k_proj/v_proj/o_proj`）。它不会报错，只会「训了等于没训」——
所以每次训练都要看 `print_trainable_parameters()` 的输出。




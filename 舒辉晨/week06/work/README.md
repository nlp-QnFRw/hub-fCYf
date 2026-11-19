# work · THUCNews 十类新闻分类：方法对比实验

> 换掉预训练模型和数据集重做一遍「分类方法对比」，用来看**结论是不是可迁移**。
> 数据：THUCNews（10 类中文新闻正文，5 万条）；模型：MacBERT-base 与 rbt3；
> 验证：8000 条与训练集完全不重叠的样本，seed=42。

## 与原项目（week6-deeplearning/code）的差别

| | 原项目 | 本项目 |
|---|---|---|
| 数据集 | TNEWS 15 类**新闻标题** | THUCNews 10 类**新闻正文** |
| 标签口径 | 15 类 → 4 组 | 10 类 → 4 组（同一套投研口径） |
| 预训练模型 | bert-base-chinese | **chinese-macbert-base** + **rbt3** |
| 方法 | 规则 / TF-IDF / BERT / LLM zero-shot / LoRA / 全量 | 规则（两档）/ TF-IDF / **TextCNN** / rbt3 / MacBERT / Qwen3 zero-shot |
| 文本长度 | 标题，平均 20 字 | 正文，平均 914 字（截断 256） |

## 目录

```
work/
├── config.py                # 数据规模、模型名、阈值等全局参数
├── common/
│   ├── labels.py            # 10 类 → 4 组主题 + 相关性 + 队列规则
│   ├── data.py              # 解析 cnews、全局去重、分层切分、截断、缓存
│   ├── metrics.py           # 统一指标 + 最容易混淆的类别对
│   └── model_paths.py       # 定位 pretrain_models 下的本地权重
├── rule/                    # L0：类名匹配 + 手工关键词表两档
├── tfidf/                   # L1：字 1-2gram + LinearSVC
├── textcnn/                 # L2：从零训练（不借预训练）
├── bert/                    # L3：MacBERT / rbt3 微调（共用 model.py）
├── zero_shot/               # L4：Qwen3-0.6B 零样本
├── eval/                    # 汇总、数据量扫描、图表与 HTML 报告
├── scripts/                 # 下模型、一键复现
├── data/                    # cnews.train.txt + 切分缓存
├── pretrain_models/         # 本地权重（不随包分发）
├── outputs/                 # 逐条预测 / ckpt / 汇总表
└── reports/                 # report.html + charts/
```

## 运行

```bash
# 0) 环境（已就绪：py312 有 torch 2.13+cu130 / sklearn / matplotlib）
conda activate py312
python scripts/fetch_models.py --with-qwen     # 下载 3 个模型到 pretrain_models/

# 1) 全流程
python scripts/run_all.py                      # 约 60~90 分钟
python scripts/run_all.py --fast               # 跳过零样本与数据量扫描

# 2) 单跑某一路
python rule/classify_rule.py
python tfidf/classify_tfidf.py
python textcnn/train_textcnn.py --epochs 4
python bert/train_bert.py --model macbert --epochs 3
python bert/train_bert.py --model rbt3 --epochs 3
python zero_shot/classify_zero_shot.py --n 300

# 3) 汇总与报告
python eval/eval_all.py
python eval/make_report.py
python eval/scan_scale.py --model rbt3          # 数据量实验
```

## 结果

### 主表（8000 条验证，10 类，seed=42）

| 方法 | 准确率 | Macro-F1 | 难例四类 F1 | 参数量 | 训练耗时 | 解析失败 |
|---|---|---|---|---|---|---|
| **MacBERT 微调** | **0.9828** | **0.9827** | 0.9801 | 102.3M | 1087s | 0 |
| rbt3 微调 | 0.9770 | 0.9770 | 0.9749 | 38.5M | 311s | 0 |
| TF-IDF + SVM | 0.9708 | 0.9707 | 0.9690 | — | **19s** | 0 |
| TextCNN 从零 | 0.9519 | 0.9519 | 0.9492 | 0.84M | 32s | 0 |
| 规则·关键词表 | 0.5404 | 0.5490 | 0.4032 | — | — | 0 |
| Qwen3-0.6B 零样本 | 0.4950 | 0.4952 | 0.5615 | 0.6B | — | 91 |
| 规则·类名匹配 | 0.4069 | 0.4132 | 0.3022 | — | — | 0 |

「难例四类」= 财经 / 房产 / 科技 / 家居，只看这四类重算 Macro-F1，用来在全量分数被天花板压平后看真实差距。

### 数据量实验（固定验证集，只改训练集比例）

| 训练条数 | 比例 | TF-IDF | rbt3 | 差距 |
|---|---|---|---|---|
| 6000 | 15% | 0.9583 | 0.9569 | −0.0014 |
| 12000 | 30% | 0.9623 | 0.9651 | +0.0028 |
| 24000 | 60% | 0.9675 | 0.9715 | +0.0040 |
| 40000 | 100% | 0.9707 | 0.9769 | +0.0062 |

### 结论

1. **判别式微调仍最优**，但优势从原项目的 7 个点缩到 1.2 个点。正文平均 900 字，类别词密度远高于 20 字标题，字面特征已经足够——**文本越长，预训练的价值越被稀释**。
2. **性价比拐点在 rbt3**：38% 的参数、29% 的耗时，拿到 99.4% 的效果（0.9770 vs 0.9827）。预算紧或要频繁重训，不必上 base。
3. **「小样本才是微调主场」在本任务不成立**，甚至方向相反：训练集越小，rbt3 相对 TF-IDF 的优势越小，15% 时还被反超 0.14 个点。原因同上——长文本让 TF-IDF 在几千条时就接近天花板，而预训练知识的兑现需要数据量支撑。**这条规律不能跨任务照搬**。
4. **零样本可用但不能上线**：Qwen3-0.6B 覆盖全部 10 类（原项目在标题上塌缩到 8/15），0.4952 明显强于规则法，但仍低于有监督方案 48 个点，适合冷启动期过渡。
5. **规则法只能当下限**：类名匹配 0.413、关键词表 0.549，难例四类上跌到 0.30 / 0.40，同义改写完全接不住。

完整图表见 `reports/report.html`，逐条预测在 `outputs/*_pred.json`，表格在 `outputs/summary.md`。

## 踩到的坑

- **镜像会把限流页面当成文件存下来**：hf-mirror 超限时返回 HTTP 200 + 一段英文提示，`scripts/fetch_models.py` 最初照单全收，导致 `Qwen3-0.6B/config.json` 缺失，报错是 `Unrecognized model ... should have a model_type key`，很难往下载上想。现在加了 JSON 合法性校验与限流识别，并对 Qwen 走 ModelScope 兜底。
- **零样本必须传 `enable_thinking=False`**，否则 Qwen3 先吐 `<think>`，解析失败率飙升。
- **`eval_all.py` 与 `make_report.py` 取数层级不同**：前者原先少取了一层 `report`，导致整张表全是 `None`。


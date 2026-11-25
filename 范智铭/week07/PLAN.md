# 第七周作业 · 项目规划

> 主题：**同一抽取任务，换模型 / 换 API / 换数据集，量化对比方法差异**
> 作者：范智铭 ｜ 规划时间：2026-11-25 ｜ 版本 v1.0（编码前定稿）

---

## 1. 背景与目标

课堂第七周讲「信息抽取」，示例工程是 DuEE-Fin 金融事件抽取，方法梯队从规则 → BiLSTM-CRF → BERT → 两段式 pipeline → LLM zero/few-shot → LoRA SFT。

本次作业不重复事件抽取，而是选**更基础、可跨数据集复用的中文命名实体识别（NER）**作为抽取任务，把「方法差异」这一个变量单独拎出来做对照实验：

- **换模型**：BiLSTM-CRF（无预训练）→ BERT → RoBERTa
- **换 API**：DeepSeek-Chat 零样本 / 三样本提示（生成式抽取，不训练）
- **换数据集**：简历合成数据（模板＋难例）→ CLUENER2020（人工标注、10 细类）→ MSRA（人工标注、3 粗类、新闻域）

目标不是刷某一个数据集的分，而是回答：**在同一套评测口径下，不同方法的能力差在哪、代价差多少、换数据集后排位会不会翻。**

---

## 2. 研究问题

| 编号 | 问题 | 观测指标 |
| --- | --- | --- |
| RQ1 | 同数据集、同训练量下，BiLSTM / BERT / RoBERTa 的实体级 F1 差多少？ | micro-F1、per-type F1 |
| RQ2 | 预训练带来的收益，在「模板合成数据」和「真实人工标注数据」上一样大吗？ | 三个数据集的 F1 差值 |
| RQ3 | 不训练、直接调 DeepSeek API，能打到有监督模型的几成？zero-shot 与 few-shot 差多少？ | F1、解析成功率、幻觉率 |
| RQ4 | 各方法的错误类型是否相同？ | 边界错 / 类型错 / 漏检 / 多检 四类占比 |
| RQ5 | 精度之外，训练成本、推理延迟、参数量、API 费用差几个量级？ | 训练时长、ms/句、参数量、token 与费用 |

**预设假设（做完再验证，不预设结论）**

- H1：RoBERTa 比 BERT 高 0.5~2 个点，数据越少差距越不稳定。
- H2：BiLSTM 在 500 条简历数据上离 BERT 不远（3~6 个点），在 1 万条 CLUENER 上差距拉大（预训练收益随数据量放大）。
- H3：LLM 的召回不差，但**边界精确率**明显低（多带单位/标点/整句），事件/实体完全匹配吃亏。
- H4：API 的延迟与费用比本地小模型高 2~3 个数量级，few-shot 的 prompt 成本还要再翻倍。

---

## 3. 实验矩阵

3 个数据集 × 5 个方法，共 **15 个实验单元**（LLM 分 zero/few 两档）：

| # | 数据集 | 方法 | 训练 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | resume | BiLSTM-CRF | 是 | 字符向量从零学 |
| 2 | resume | BERT-base-chinese | 微调 | 512 长度 |
| 3 | resume | chinese-roberta-wwm-ext | 微调 | 512 长度 |
| 4 | resume | DeepSeek zero / few | 否 | 全测试集 80 条 |
| 5 | cluener | BiLSTM-CRF | 是 | 160 长度 |
| 6 | cluener | BERT | 微调 | 160 |
| 7 | cluener | RoBERTa | 微调 | 160 |
| 8 | cluener | DeepSeek zero / few | 否 | 固定抽 300 条测试 |
| 9 | msra | BiLSTM-CRF | 是 | 160 |
| 10 | msra | BERT | 微调 | 160 |
| 11 | msra | RoBERTa | 微调 | 160 |
| 12 | msra | DeepSeek zero / few | 否 | 固定抽 300 条测试 |

> 说明：LLM 的 few-shot 示例**只从 train 里取**，绝不碰 dev/test；抽样用固定 seed，样本 id 落盘，保证跨方法完全同批。

---

## 4. 数据集设计

### 4.1 三套数据的分工

| 数据集 | 来源 | 域 | 标签 | 规模（train/dev/test） | 在实验里的角色 |
| --- | --- | --- | --- | --- | --- |
| resume | 本项目模板合成（seed=42） | 简历 | 10 类：NAME/TEL/EMAIL/EDU/SCHOOL/MAJOR/ORG/POS/TIME/SALARY | 500 / 60 / 80 | **小样本 + 模板化**，检验「预训练值不值」 |
| cluener | CLUEbenchmark/CLUENER2020 官方公开集 | 泛领域句子 | 10 类：地址/书名/公司/游戏/政府/电影/姓名/组织机构/职位/景点 | 10,748 / 1,343 / 1,345 | **真实人工标注 + 细粒度**，主战场 |
| msra | MSRA NER（BIO 版，公开镜像） | 新闻 | 3 类：PERSON/LOCATION/ORGANIZATION | 取 10,000 / 1,000 / 1,000（固定切分） | **粗粒度 + 大规模**，看标签体系变粗后各方法是否收敛 |

三套数据的标签体系**互不相同**，因此不做跨数据集迁移，只做「同一数据集内方法对比」+「同一方法跨数据集排位」。

### 4.2 统一中间格式

所有数据统一成 JSONL，一行一条：

```json
{"id": "cluener-test-000123", "text": "…原文…",
 "entities": [{"start": 0, "end": 3, "label": "company"}]}
```

- 字符级 offset，左闭右开；`text[start:end]` 必须等于实体原文（脚本断言）。
- 标签做 ASCII 化（`company` / `name` / `PER` …），避免 Windows 控制台与 seqeval 的编码坑；展示时再映射回中文。
- 每套数据产出 `data/processed/<dataset>/{train,dev,test}.jsonl` + `labels.json` + `stats.json`。

### 4.3 简历合成数据的设计要点

- 10 类字段，模板 + 版式变体（冒号式 / 表格式 / 句子式 / 全角数字 / 大写邮箱 / “至”连接）。
- 人名、学校、公司等槽位池**按 split 切分**，train/dev/test 不共享字符串，避免实体泄漏。
- 干扰项：性别、年龄、籍贯、邮编、QQ、绩点、CET 分数一律 `O`，逼模型学边界而不是「见数字就标」。
- 测试集约 40% 为难例（未见版式），单独统计 hard 子集 F1。

---

## 5. 方法设计

### 5.1 BiLSTM-CRF（无预训练基线）

- 字符级 embedding 128 → BiLSTM(hidden 256, 1 层) → Linear → CRF。
- 词表只用 train 字符，`<unk>` 兜底；dropout 0.3；Adam 1e-3；batch 64；15 epoch；按 dev F1 早停。
- CRF 自己实现（前向 / Viterbi / 负对数似然），不引第三方依赖，便于讲清「为什么 CRF 能约束合法 BIO」。

### 5.2 BERT / RoBERTa 微调

- `AutoModelForTokenClassification`，`num_labels = 2*|L|+1`。
- 子词对齐：fast tokenizer 的 `word_ids()` + `offset_mapping`，同一词首子词 `B-`、后续 `I-`，特殊符 `-100`。
- 超参：lr 3e-5 / 2e-5，batch 32（长度 160），warmup 0.1，weight decay 0.01，fp16，3 epoch，按 dev entity-F1 存 best。
- 推理：token 预测 → 合并同类连续 token → 用 offset 首尾还原字符区间。

### 5.3 DeepSeek API（零样本 / 三样本）

- 模型 `deepseek-chat`，`temperature=0`，`max_tokens=512`，不开思维链。
- Prompt：给出标签定义 + 边界规则 + **只输出 JSON**；实体以「原文片段」给出，程序用 `text.find()` 回填 offset（不让模型算下标）。
- 解析：去 markdown 围栏 → 括号配平截取 → JSON 修复 → 校验标签合法 → 原文对齐；失败保留空结果并计数（不剔除样本）。
- few-shot：每类 1 条，全部取自 train；含一条「无实体」负例。
- 并发 8 线程 + 指数退避重试，响应逐条落盘（可断点续跑）。

---

## 6. 统一评测协议

1. **实体级严格匹配**：`(start, end, label)` 三元组完全一致才算 TP（等价 seqeval `strict`）。
2. 主指标：micro-P / micro-R / **micro-F1**；辅指标：macro-F1（按类型平均）、per-type P/R/F1。
3. 所有方法在**同一份 test 文件**上评；LLM 只用固定子集（id 落盘），其余方法在同一子集上另算一份，保证可比。
4. 错误归因：把 FP/FN 分成「类型错（区间对、标签错）」「边界错（标签对、区间有交集但不完全等）」「完全漏/多」三类，分别统计。
5. 效率：参数量、训练时长（GPU 秒）、推理延迟（ms/句，warmup 后 3 次取均值）、LLM 的 token 数与费用（单价走环境变量，不硬编码）。
6. 可复现：固定 seed=42；数据处理、切分、抽样全部落盘；一条命令复跑。

---

## 7. 工程结构

```text
week07/
├─ PLAN.md / README.md / REPORT.md
├─ requirements.txt
├─ configs/                  # 各方法超参（json）
├─ data/
│  ├─ raw/                   # 原始下载（zip/txt）
│  └─ processed/<dataset>/   # train/dev/test.jsonl + labels.json + stats.json
├─ src/
│  ├─ common.py              # 路径/日志/seed/json 工具
│  ├─ data.py                # 读取、BIO 互转、词表、collator
│  ├─ metrics.py             # 统一评测 + 错误归因
│  ├─ gen_resume.py          # 简历合成数据生成
│  ├─ bilstm_crf.py          # 模型 + CRF
│  ├─ train_bilstm.py
│  ├─ train_transformer.py   # BERT / RoBERTa 共用
│  ├─ predict.py             # 神经模型推理 → 统一 pred jsonl
│  ├─ llm_extract.py         # DeepSeek zero/few-shot
│  └─ compare.py             # 汇总 CSV/Markdown + 图
├─ scripts/
│  ├─ download_data.py       # 下载 CLUENER / MSRA
│  ├─ prepare_data.py        # 转统一格式 + 校验
│  └─ run_all.py             # 端到端一键跑
├─ outputs/<dataset>/<method>/   # 权重、预测、指标、日志
└─ reports/figures/          # 对比图
```

---

## 8. 里程碑与时间预算

| 阶段 | 内容 | 预计耗时 | 产出 |
| --- | --- | --- | --- |
| P0 | 环境确认、数据下载、统一格式与校验 | 40 min | `data/processed/*`、`stats.json` |
| P1 | 评测脚本 + BiLSTM-CRF + 冒烟跑通 | 60 min | 指标链路可信（gold 自评 = 1.0） |
| P2 | BERT / RoBERTa × 3 数据集训练 | 90~120 min（GPU） | 6 组权重与预测 |
| P3 | DeepSeek zero/few × 3 数据集 | 30~40 min（并发） | 6 组预测 + token 账单 |
| P4 | 汇总、作图、错误分析、写报告 | 60 min | `REPORT.md` + 图 + README |

总预算约 **5 小时**，GPU 任务串行执行（8GB 卡同时跑两个 BERT 会 OOM）。

---

## 9. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| CLUENER / MSRA 下载源被墙 | 数据拿不到 | 已验证 `storage.googleapis.com`（CLUENER）与 `cdn.jsdelivr.net`（MSRA）可直连；失败则退化为两套数据并如实说明 |
| 8GB 显存不够 | 训练 OOM | 长度压到 160、batch 32 + 梯度累积、fp16；BiLSTM 走 CPU 也不慢 |
| API key 只在系统变量、当前进程读不到 | LLM 跑不了 | 运行时从注册表 `Machine` 作用域读取并注入子进程；代码同时支持 `DEEPSEEK_KEY` / `DEEPSEEK_API_KEY` / `.env` |
| API 限流 / 超时 | 抽样跑不完 | 8 并发 + 退避重试 + 逐条落盘，可续跑；失败样本保留并计入分母 |
| 模板数据指标虚高 | 结论误导 | 报告里把 resume 与真实数据分开讲，并单列 hard 子集 |
| 生成式输出格式不稳定 | F1 被解析拖累 | 解析器容错 + 统计 JSON 合法率，合法率与 F1 一起报 |
| 中文图表乱码 | 报告难看 | 优先用系统字体（微软雅黑/黑体），失败则图内用英文标签 |

---

## 10. 交付物与验收标准

**交付物**

1. `PLAN.md`（本文）、`README.md`（怎么跑）、`REPORT.md`（结果与分析）
2. `data/processed/` 三套统一格式数据 + 统计
3. `src/` 全部代码（可读、带关键注释）
4. `outputs/` 权重、预测、指标、日志
5. `reports/figures/` 对比图（F1 柱状、per-type、错误类型、精度-延迟散点）
6. `RESULTS.csv` / `RESULTS.md` 汇总表

**验收（自检）**

- [ ] gold 当 pred 跑评测，三套数据 F1 均 = 1.0000（评测口径自证）
- [ ] 三套数据 offset 校验 100% 通过，split 之间无实体字符串泄漏（resume）
- [ ] 15 个实验单元全部有 `metrics.json` + `pred.jsonl`
- [ ] 汇总表里每个数字都能追到具体文件
- [ ] 结论只写「测出来的差异」，不写没有证据的判断

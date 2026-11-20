# THUCTC 中文新闻多分类：多方法对比与系统消融

用 THUCNews（清华 THUCTC 官方语料，74 万篇、14 类）做中文长文本多分类，**同一套数据口径下对比 6 条技术路线**，并系统做消融，回答"哪些配置真的影响结果、哪些只是心理安慰"。

| 级别 | 路线 | 目录 | 训练成本 |
|---|---|---|---|
| L0 | 关键词 / 规则 | `rule/` | 0（CPU 2 min） |
| L1 | TF-IDF + 线性模型 | `tfidf/` | 0（CPU 6~9 min） |
| L2 | BERT 微调（3 个主干 + 6 组超参） | `bert/` | GPU 6~45 min/run |
| L3-a | 本地小模型 zero/few-shot | `llm_local/` | 0（GPU 推理 ~8 条/s） |
| L3-b | DeepSeek-V4.1-Flash zero/few-shot | `llm_api/` | 按 token 计费 |
| L4 | Qwen2-0.5B LoRA SFT | `llm_lora/` | GPU ~45 min/run |

**统一口径**（对比能成立的前提）：
- 训练集 `train_100k`（分层）；数据量消融用嵌套子集 `sub_5k ⊂ 20k ⊂ 50k`
- 主指标：`test_bal`（每类 ≥300 条）上的 **macro-F1**
- 辅指标：`test_nat_10k`（按自然分布）上的 **acc**
- 生成式方法的拒答/解析失败**计入分母**，失败率单列
- 固定 `seed=42`；相邻方法差异用 McNemar 配对检验（`eval/paired_test.py`）

---

## 结果（32 个 run，全部配对可比）

| 方法 | macro-F1 | acc | 主题F1 | 拒答率 | 耗时 |
|---|---|---|---|---|---|
| **BERT-base 100k×3ep** | **0.9628** | **0.9701** | 0.9731 | 0 | 51.5 min GPU |
| TF-IDF 字 n-gram + LinearSVC | 0.9237 | 0.9491 | 0.9474 | 0 | 5.4 min CPU |
| TF-IDF 字 n-gram + LR | 0.9166 | 0.9459 | 0.9418 | 0 | 6.6 min CPU |
| TF-IDF 分词 + LR | 0.9144 | 0.9419 | 0.9406 | 0 | 6.9 min CPU |
| **DeepSeek-V4.1-Flash** 0-shot（API） | 0.8199 | 0.8038 | 0.8949 | **0** | 15.2 min / ¥3.02 |
| LoRA（Qwen2-0.5B 10k×3ep） | 0.8104 | 0.8578 | 0.8495 | 0.6% | 56.5 min GPU |
| Qwen2-0.5B zero-shot | 0.5437 | 0.5274 | 0.7264 | **20.4%** | 7.4 min GPU |
| 关键词规则 | 0.5391 | 0.5100 | 0.6538 | 0 | 2 min CPU |

一句话结论：**微调 BERT 比 TF-IDF 高 3.9pp（p<0.0001）；但 5 分钟的 TF-IDF 比 57 分钟的 LoRA 高 11.5pp；
0.5B 零样本与关键词规则统计上不可分辨（p=0.514）；而 3 元钱的 DeepSeek API 零样本就能与
LoRA 打平（p=0.154）。**

三条与直觉相反的实测结论：
1. **加长输入没有收益**：`max_len` 128/256/384/512 的 macro-F1 全在 1.55pp 内且不单调；
   只留标题才真的掉 4.7pp——读开头就够，不必读完全文。
2. **瓶颈是数据不是模型**：5k→100k 每翻倍涨 5~7pp，100k 仍未饱和；30k 上那些 ±1pp 的消融
   其实是在欠拟合区里比细节。
3. **few-shot 在小模型上单调恶化**：0/1/3/5-shot 的 macro-F1 是 0.544→0.509→0.360→0.065，
   5-shot 有 91.6% 的输出是"正文续写"而不是类名。

完整表与图表见 `runs/summary.md`、`figures/`，分析见 `REPORT.md`。

---

## 5 分钟跑通（最小链路）

```bash
conda activate py312
cd "D:/AI_learn/course-related/Week6/正式_week6作业"

# 0) 环境探针（把"能不能跑"变成数字，写入 docs/PROBE.md）
python -m common.probe

# 1) 数据：下载 → 转单 jsonl → 切分 → 长度统计
python scripts/00_download.py            # 1.45GB，断点续传，校验字节数
python scripts/01_to_jsonl.py            # zip 内 74 万 txt → data/processed/all.jsonl
python scripts/02_splits.py              # train_100k / val_10k / test_nat_10k / test_bal + 嵌套子集
python scripts/03_len_stats.py           # 长度分布 + 各 max_len 截断覆盖率

# 2) 额外主干（可选，走 hf-mirror）
python scripts/04_fetch_backbones.py

# 3) 冒烟：每条路线都能在 2 分钟内跑完（产物不可用于汇报）
python rule/classify_rule.py --demo
python tfidf/train_tfidf.py --demo
python bert/train_bert.py --demo
python llm_local/classify_local.py --demo
python llm_lora/train_lora.py --demo

# 4) 正式实验（清单驱动、串行、可续跑、跳过已完成）
python scripts/10_run_queue.py --list
python scripts/10_run_queue.py --tier t1

# 5) 评测与汇总
python eval/evaluate.py --all
python eval/compare.py                   # → runs/summary.md + figures/
python eval/paired_test.py --all-pairs   # McNemar
```

结果表见 **`runs/summary.md`**（自动生成），分析见 **`REPORT.md`**。

---

## 目录结构

```
common/            唯一真源：labels / paths / seed / text(截断) / data / metrics / contract / prompt / probe
scripts/           00 下载 · 01 转 jsonl · 02 切分 · 03 长度统计 · 04 下载主干 · 10 实验队列
rule/ tfidf/ bert/ llm_local/ llm_api/ llm_lora/     六条方法线，互不 import，只依赖 common/
eval/              evaluate(唯一评测入口) · compare(汇总+画图) · paired_test(McNemar)
data/{raw,processed,splits}/
runs/<run_id>/     meta.json · metrics.json · pred_test*.jsonl · train_log.json · best.pt|adapter/
figures/           len_dist · data_curve · compare_macro_f1 · confusion_best
docs/              PROBE(实测) · PITFALLS(踩坑) · PLAN(预算与偏差)
```

**产物契约**（`common/contract.py` 保证）：每个 run 都要产出
`pred_test.jsonl`（test_nat）与 `pred_test_bal.jsonl`（test_bal），
每行 `{id, gold, pred, raw, status, latency_ms}`，`id` 是 `all.jsonl` 的全局行号
——这是 McNemar 配对与错例交集能算出来的唯一键。

---

## 本机实测（决定所有超参的依据）

RTX 5050 Laptop 8GB / torch 2.11+cu128 / transformers 5.23：

| 配置 | 吞吐 | 显存峰值 |
|---|---|---|
| BERT bs32 / L128 | 231 samples/s | 2.86 GB |
| **BERT bs32 / L256（推荐）** | **121 samples/s** | **4.32 GB** |
| BERT bs32 / L384 | 75 samples/s | 5.76 GB |
| BERT bs64 / L256 | 36 samples/s ⚠️ | 7.24 GB |
| **LoRA bs2 / L384（推荐）** | **12.1 samples/s** | **4.01 GB** |
| LoRA bs4 / L384 | 6.1 samples/s ⚠️ | 7.05 GB |
| LoRA bs8 / L256 | 6.0 samples/s ⚠️ | 8.75 GB |
| 本地生成 bs16 | 40 samples/s | 1.15 GB |

**核心结论：显存一旦接近 7GB，吞吐断崖式下跌（不是 OOM，是静默降级）。**
所有训练脚本都带 `max_memory_allocated < 6.5GB` 断言，越线直接报错而不是慢慢跑。

---

## 常见问题

**Q：为什么不用 transformers 的 `Trainer`？**
A：5.x 改了一批 `TrainingArguments` 参数名，且 Trainer 不方便做显存断言与分组学习率。自写 60 行循环即可，见 `bert/train_bert.py`。

**Q：为什么 LoRA 只做 0.5B，不做全量微调？**
A：494M 参数的 AdamW fp32 状态 + 梯度 + 激活 ≈ 5.9GB 起步，8GB 卡物理不可行。LoRA(r=8) 只训 1.1M（0.22%）。

**Q：DeepSeek 结果为什么有时"答非所问"？**
A：`deepseek-flash` 是推理模型，默认先输出思考内容。必须 `{"thinking":{"type":"disabled"}}` 或 `reasoning_effort`，否则 `content` 为空。见 `docs/PITFALLS.md` #18。

**Q：为什么有两个测试集？**
A：`test_bal` 每类 ≥300 条才谈得上"每类 F1 可比"；`test_nat` 按自然分布才贴近线上 acc。混用会让长尾类的结论失真。

# 第七周作业：FewFC 金融事件抽取 —— 判别式 / 本地 LLM / 云端 API 三条路线对比

在 FewFC（金融小样本事件抽取，10 事件类型 / 18 论元角色）上，用**同一套数据口径、同一个评测函数**跑 **22 个方法点**，回答"信息抽取到底该选哪条技术路线"。

| 级别 | 路线 | 目录 | 配置数 | 成本 |
|---|---|---|---|---|
| L0 | 中文 BERT 系判别式（3 编码器 × 2 层数 × 2 解码器） | `disc/` | 13 | GPU 3–10 min/run |
| L1 | 本地离线 LLM（Qwen2-0.5B / Qwen3-0.6B / 1.7B / 4B） | `llm/` | 6 | 0 API 费用，2.7–47 s/句 |
| L2 | 云端 API（DeepSeek-V4.1-Flash zero/few-shot） | `llm/extract_api.py` | 3 | 68.3 万 token |

**统一口径**（跨路线可比的前提）：

- 主口径 `dev_n200`：seed=42 固定抽 **200 句 / 301 事件**，所有方法都在这同一批句子上评
- 判别式另报 dev 全量 2548 句（组内消融用），并**投影到同一 n200**
- 唯一评测函数 `common/evaluate_events.py`：trigger / argument / event_exact 三级一次算全
- 解码失败、JSON 非法**一律保留 `events: []` 计入分母**，不删样本
- 固定 `seed=42`；报告里的表格由图脚本从 `outputs/metrics/disc/*.json` 自动回填，不手抄

---

## 一句话结论

**同一批 200 句上胜负是分裂的**：LoRA 微调后的本地 0.6B 小模型全面领先（trigger **0.618** / argument **0.457** / event **0.171**）；中文 BERT（MacBERT-12L+CRF，0.506 / 0.288）触发词强于云端 API（0.460 / 0.340）但论元更弱；而 0.6B 的 zero-shot 只有 0.087 / 0.008。

也就是说——**"本地小模型不行"只对 zero/few-shot 成立**。用 3552 句做一次 LoRA（1 epoch、单卡 8GB、50 分钟），trigger 从 0.087 拉到 0.618（**7 倍**），反超云端 API 15.8pp、反超 BERT 11.2pp。

## 主结果（n200 同批 200 句）

| 方法 | trigger F1 | argument F1 | event F1 | JSON 合法率 | 每句耗时 |
|---|---|---|---|---|---|
| **本地 Qwen3-0.6B LoRA-SFT** | **0.6181** | **0.4567** | **0.1705** | 0.99 | 2.68 s |
| **macbert-12L+CRF（n200 投影）** | 0.5059 | 0.2881 | 0.0742 | — | — |
| roberta-12L+CRF（n200 投影） | 0.5051 | 0.2827 | 0.0707 | — | — |
| bert-12L+CRF（n200 投影） | 0.4880 | 0.2607 | 0.0756 | — | — |
| DeepSeek API zero-shot #2 | 0.4648 | 0.3389 | 0.0820 | 1.00 | 0.17 s |
| DeepSeek API zero-shot #1 | 0.4597 | 0.3396 | 0.0707 | 1.00 | 0.17 s |
| DeepSeek API few-shot(3 例) | 0.3453 | 0.2662 | 0.1055 | 0.62 | 0.81 s |
| 本地 Qwen3-1.7B zero-shot | 0.2470 | 0.1103 | 0.0097 | 0.71 | 9.79 s |
| 本地 Qwen3-0.6B few-shot(9 例) | 0.2343 | 0.1301 | 0.0033 | 0.94 | 14.44 s |
| 本地 Qwen3-4B zero-shot | 0.1726 | 0.1314 | 0.0417 | 0.28 | 46.98 s |
| 本地 Qwen3-0.6B zero-shot | 0.0872 | 0.0076 | 0.0000 | 0.88 | 4.48 s |
| 本地 Qwen2-0.5B zero-shot | 0.0307 | 0.0000 | 0.0000 | 0.89 | 3.17 s |

判别式组内消融（dev 全量 2548 句，13 配置）见 `HOMEWORK_REPORT.md` §2.1。**完整分析、bad case 归因、10 条工程踩坑实录见 [`HOMEWORK_REPORT.md`](HOMEWORK_REPORT.md)。**

## 结论要点

1. **触发词**：判别式 > 云端 API（0.506 vs 0.460，+4.6pp）——金融公告的触发词高度模式化（"质押""减持""中标"），5.5k 条标注的 BERT 比通用大模型 zero-shot 更稳。
2. **论元**：云端 API > 判别式（0.340 vs 0.288，+5.2pp），但 API 只抽到 208/301 个事件（判别式抽到 292 个）——生成式论元质量更高，代价是整条事件漏抽。
3. **层数是判别式唯一的大杠杆**：12L→4L 掉 trigger 2.4–3.7pp、argument **9.4–9.5pp**；而 4 层**随机初始化**（假 4 层）把 argument 直接打到 0.020——**"层数消融"与"初始化消融"必须分开做**，否则会把预训练知识的价值误记到层数头上。
4. **三个"改进版 BERT"在这数据规模上没有差异**（trigger 差 ≤0.6pp，argument 差 ≤0.7pp）——"别为 0.5pp 换模型"的实证。
5. **参数量换不来结构化输出能力**：0.5B→0.6B→1.7B→4B 的 zero-shot，4B 的 JSON 合法率只有 **27.5%**、47 s/句（输出顶到 800 token 上限被截断）。8GB 卡的实用上限是 1.7B（bf16 全量驻留）。
6. **云端 API 的 few-shot 反而更差**：trigger 0.460→0.345、JSON 合法率 1.00→0.62，因为示例里有"一句 5 个事件"的长目标串，把模型教成了"先铺开写"，直接撞输出预算。**few-shot 示例要挑短且规整的。**
7. **event_exact 天花板是结构性的**：判别式沿用"一句一事件"解码，而 FewFC 有 30.7% 多事件句，event F1 被钉在 0.07 附近——这不是调参问题，是单序列解码的结构性问题。

## 目录结构

```
week07/
├── HOMEWORK_REPORT.md          # 主报告（结论 / 主表 / bad case / 10 条工程发现 / 复现指引）
├── README.md                   # 本文件
├── common/                     # 公共层（8 个模块）
│   ├── evaluate_events.py      # ★ 唯一评测函数（trigger/argument/event_exact 三级口径）
│   ├── event_data.py           # 数据加载 + BIO 转换
│   ├── models.py / crf.py      # 判别式模型与 CRF 解码
│   ├── model_paths.py          # 本地模型路径解析（MODEL_ROOT 可覆盖）
│   ├── vram_guard.py           # 显存守卫（8GB 卡上的硬约束）
│   └── eval_utils.py / log_utils.py
├── disc/                       # A 组：判别式
│   ├── train.py                # 单配置训练（BERT 系 × 层数 × Linear/CRF）
│   └── run_grid.py             # 13 配置网格（超时继续 + 断点续跑 + 增量落盘）
├── llm/                        # B/C 组：LLM
│   ├── extract_local.py        # 本地离线抽取（zero / few-shot / LoRA）
│   ├── sft_lora.py             # Qwen3-0.6B LoRA 微调
│   ├── extract_api.py          # DeepSeek API 抽取
│   ├── prompt.py               # schema prompt（本地/API 共用）
│   └── run_night_chain.py / run_local_all.py / extract_common.py
├── eval_side/                  # D 组：收口
│   ├── compare.py              # 跨方法对比表
│   ├── make_figures.py         # 四张图
│   ├── fill_report.py          # 从 JSON 自动回填报告表格（不手抄）
│   ├── mine_badcases.py        # bad case 归因
│   └── verify_metrics.py       # 验收：指标齐全 / 非退化 / 行数对得上（22/22 全绿）
├── data/
│   ├── scripts/                # build_dataset.py（FewFC → 三口径）、verify_processed.py
│   └── processed/              # 派生数据（train 5521 / dev 4087 / dev_n200 301 / test 3203 行）
└── outputs/
    ├── metrics/disc/*.json     # 44 份指标（argv / epoch / 训练时长 / 峰值显存 / 延迟 / token 量）
    ├── disc/pred_*.jsonl       # 22 份统一格式预测（可直接重评）
    ├── logs/                   # 训练三件套（train.json / steps.jsonl / console.log）+ 网格执行记录
    ├── figures/*.png           # 4 张图（A 组网格 / 层数消融 / 全方法 n200 / 参数量 vs 微调）
    ├── sft/qwen3_0.6b_lora/    # LoRA 适配器（40 MB，--adapter 直接加载）
    └── checkpoints/            # ⚠️ 仅占位说明，权重文件未上传（见下）
```

## 关于未上传的大文件

| 路径 | 内容 | 体积 | 处理 |
|---|---|---|---|
| `outputs/checkpoints/*.pt` | 13 个判别式 checkpoint | **3.9 GB** | **未上传**（GitHub 单文件上限 100 MB），见 `outputs/checkpoints/README.md` |
| `data/raw/FewFC/` | FewFC 原始数据集（含上游 git 仓库） | 85 MB | **未上传**（第三方数据，按上游 README 自行克隆） |

`outputs/checkpoints/README.md` 里写明了每个 checkpoint 的文件名、对应配置和**重新生成/回归评测的方法**，所以仓库仍然是可复现的。

## 复现

```bash
conda activate py312

# 0. 数据派生（需先克隆 FewFC 到 data/raw/FewFC）
python data/scripts/build_dataset.py && python data/scripts/verify_processed.py

# A 组：13 配置判别式网格（断点续跑，单配置硬超时 900s，失败继续 + 末尾重试）
python disc/run_grid.py --skip-done --deadline 07:00

# B 组：本地 LLM 整链（zero / few-shot / LoRA / 0.5B / 1.7B / 4B）
python llm/run_night_chain.py --deadline 07:00
python llm/sft_lora.py --epochs 1 --batch_size 1 --grad_accum 8 --max_len 1024
python llm/extract_local.py --model Qwen3-0.6B --adapter outputs/sft/qwen3_0.6b_lora --run_name local_Qwen3-0.6B_lora

# C 组：云端 API（需 DEEPSEEK_API_KEY，会消耗 token）
python llm/extract_api.py --thinking disabled --repeat 2

# D 组：收口
python eval_side/compare.py && python eval_side/make_figures.py
python eval_side/fill_report.py && python eval_side/verify_metrics.py
```

- 本地模型统一放在 `D:\AI_learn\Model`，可用环境变量 `MODEL_ROOT` 覆盖（见 `common/model_paths.py`）
- 环境：RTX 5050 Laptop 8GB + 32GB 内存，torch 2.11+cu128 / transformers 5.23，conda py312
- 数据来源：[FewFC](https://github.com/NingNing1213/FewFC)（金融小样本事件抽取，CC BY-SA）

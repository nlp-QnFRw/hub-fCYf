# work · 换模型 / API / 数据集的事件抽取方法对比

> 这是 week7 主项目之外的**第二轮实验**，组织方式照 week6 的 `work/`：
> 不在原代码上加分支，而是复用 week7 的 `common/`（BIO 编解码、唯一评测函数、prompt/解析器），
> 只新增「数据变体 + 新方法编排 + 对比分析」。所有数字都是本机 py312 + RTX 4070 Laptop 8.6GB 实测。

## 一、要回答的四个问题

| # | 问题 | 换的东西 | 结论（先看这里） |
|---|---|---|---|
| Q1 | 换小模型能不能省掉大模型？ | rbt3 38.6M ↔ MacBERT 102.4M | 论元 F1 0.3936 → 0.5798，差 18.6 个点；训练只慢 1.9 倍 |
| Q2 | 判别式 vs 生成式，谁更适合事件抽取？ | CRF 序列标注 ↔ Qwen3-0.6B prompt | 同批 60 条：0.5948 vs 0.1046（zero）；生成式 event F1 仍为 0 |
| Q3 | 没见过的事件类型谁能抽？ | 训练删掉 3 类（heldout） | 两个判别式模型全 0；LLM 靠 schema 拿到 0.0711 论元 / 0.0517 事件 F1 |
| Q4 | 从句子换到篇章会掉多少？ | dev 换成 985 篇公告切句 | 句级论元 0.5798 → 0.3814（−34%）；篇章事件集合 F1 只有 0.0603 |

## 二、数据变体（三个，不是同一个测试集，不要直接比数值）

| 变体 | 构造 | train / dev | 用途 |
|---|---|---|---|
| `sent` | 直接引用 week7 `data/processed`，不复制 | 5,747 / 930 | 对照组 |
| `heldout` | train 删除「被约谈 / 企业破产 / 股东增持」全部样本，dev 只看这 3 类 | 5,355 / 75 | 跨类型泛化；schema 仍保留 13 类全部标签 |
| `doc` | 985 篇公告按句切分（10,252 句），offset 在**句内**用 `find` 重算；额外产出 `doc_index.json` 保存整篇金标 | 沿用句级 train / 10,252 句 | 篇章级事件流；评测句级 + 篇章事件集合 F1 |

构建统计见 `data/build_stats.json`。篇章切句时 **1,747 个跨句论元**无法留在触发词所在句（句级 gold 只收同句论元），但它们完整保留在 `doc_index.json` 的篇章金标里；另有 208 个事件因同句只能放一个事件而被覆盖。这两条是变体本身的口径代价，不是模型误差。

## 三、方法

| 方法 | 换的是什么 | 实现 |
|---|---|---|
| `rule_*` | 零训练基线：train 自动挖触发词词典 + 数值正则 | `methods/rule_baseline.py` |
| `rbt3_crf_*` | 预训练 backbone（38.6M，3 层） | `methods/train_bio.py --model rbt3` |
| `macbert_crf_*` | 预训练 backbone（102.4M，全 12 层） | `methods/train_bio.py --model macbert` |
| `qwen_{zero,few}_*` | 生成式，Qwen3-0.6B 本地 prompt | `methods/qwen_zero.py` |
| `api_zero_*` | 生成式，OpenAI 兼容 API（无 key → NOT_RUN） | `methods/api_zero.py` |
| `*_gate3` | 篇章后处理：只留 ≥3 个论元的事件（事后诊断） | `methods/doc_gate.py` |

判别式训练配置统一：joint 149 标签、CRF 解码、2 epoch、batch 16、max_len 256、lr 2e-5（head ×5）、按 train 内切 10% 选 epoch，dev 不参与选点。生成式统一：60 条固定抽样（`eval_ids_*_n60.json`）、贪心解码、`prompt_max_length=2048`。

## 四、实测结果

完整表见 `outputs/metrics/summary.md`（自动生成）。核心数字：

### 全量 dev（句子级）

| 方法 | dev | trigger F1 | argument F1 | event F1 | 训练 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 规则 | 930 | 0.8155 | 0.1213 | 0.0011 | 0 | — |
| rbt3 + CRF | 930 | 0.7592 | 0.3936 | 0.0565 | 205 s | 7.49 ms/句 |
| MacBERT + CRF | 930 | 0.8105 | **0.5798** | **0.1688** | 381 s | 10.32 ms/句 |

### 同批 60 条投影（判别式投影到 LLM 的 `eval_ids`，可直接横比）

| 方法 | trigger | argument | event | 备注 |
|---|---:|---:|---:|---|
| 规则 | 0.7833 | 0.1607 | 0.0000 | — |
| rbt3 + CRF | 0.7719 | 0.4267 | 0.0702 | — |
| MacBERT + CRF | 0.7731 | **0.5948** | **0.1681** | 全量 930 上也是 0.5948 量级，抽样稳定 |
| Qwen3 zero-shot | 0.4576 | 0.1046 | 0.0000 | JSON 合法 60/60 |
| Qwen3 few-shot | 0.5333 | 0.2194 | 0.0500 | 修掉 prompt 截断后 |
| API zero-shot | — | — | — | **NOT_RUN**（无 key，未填估算值） |

### 换数据集

| 变体 | 最好判别式 argument F1 | LLM argument F1 | 篇章事件 F1 |
|---|---:|---:|---:|
| heldout（75 条，只看未见类型） | 0.0000（rbt3 / MacBERT / 规则均为 0） | 0.0711（zero，n=60） | — |
| doc（10,252 句） | 0.3814（MacBERT） | 0.0215（zero，n=60） | 0.0603 → 闸门 ≥3 论元 **0.0795** |

## 五、分析

1. **预训练规模的收益是真的，但不是等比的。** rbt3 → MacBERT 参数 ×2.65，论元 F1 +18.6 个点（相对 +47%），触发词只 +5 个点（0.7592 → 0.8105）。说明多出来的容量主要花在「角色归属」而非「找到触发词」——和 week7 的结论一致：触发词谁都能做，论元才是分水岭。代价是训练 ×1.9、延迟 ×1.4；若预算紧，rbt3 仍拿到 MacBERT 68% 的论元效果。

2. **同批比较时，小模型 + 标注数据仍大幅领先生成式 0.6B。** MacBERT 0.5948 vs Qwen zero 0.1046，差 49 个点。但注意 Qwen 的 event F1 是 0：它抽出的论元文本/角色经常对，类型却对不上完整事件——这正好呼应 week6 的发现：小模型 zero-shot 能「沾边」，但不能直接上线。few-shot 把 argument 从 0.1046 拉到 0.2194（+11.5 点），是四路里性价比最高的一次改动。

3. **few-shot 曾经「更差」，是 prompt 被截断，不是方法不行。** 实测 zero prompt 约 850 token、few-shot 约 1500 token；week7 默认的 `max_length=1280` 减去 384 生成预算只剩 896 token，few-shot 的 system schema 被截掉，模型开始输出 schema 里根本不存在的类型（如「独立非执行董事逝世」）。把上限提到 2048 后，few-shot 从 0.0137 恢复到 0.2194（16 倍）。**教训：prompt 变长必须同步检查 tokenizer 后的实际 token 数，不能只看字符数。**

4. **未见类型暴露了封闭标签模型的硬边界。** heldout 变体里 train 完全没有这 3 类样本，两个 CRF 模型和规则法的 argument / event F1 全是 0——不是「差」，是结构上不可能：标签头对应的类型-角色从没有被监督过。Qwen zero-shot 靠 schema 文字描述拿到 0.0711 argument / 0.0517 event F1，虽然绝对值低，却是唯一「有可能」的路线。这与 week6 的「15 类 zero-shot 塌缩到 8 类」是同一个现象的两面：生成式的价值在开放输出，不在封闭标签上硬碰判别式。

5. **句子级分数高 ≠ 篇章可用。** MacBERT 在切句后的 dev 上 trigger F1 仍有 0.3713（召回 0.77），但 precision 只有 0.24——它几乎见句就抽。汇总到篇章后，事件集合 F1 仅 0.0603（tp=153 / fp=3433 / fn=1339）。加一个「≥3 个论元才保留」的事后闸门把 F1 提到 0.0795（+32%），说明篇章链路缺的不是抽取模型，而是 week7 `service/extract_service.py` 里那种**检测闸门 + schema 校验 + needs_review**。另外该阈值是在测试集上扫出来的，只能当诊断，不能当无偏上线指标。

6. **规则法的 trigger F1 比 MacBERT 还高，但 event F1 几乎为 0。** 0.8155 vs 0.8105，原因是触发词词典直接来自 train，而 dev 里大量触发词就是那几个高频词；但规则只填时间/金额/比例/数量四类正则，角色和 span 组合对不上，event F1 只有 0.0011。**单看 trigger 会严重高估规则法。**

## 六、口径与限制（引用数字前必读）

- `sent` / `heldout` / `doc` 是三个不同测试集，跨变体的数值**不能**直接比较；只有同变体、同 60 条 `eval_ids` 的 `_n60` 组可以横比。
- LLM 只跑 60 条；同 seed 重复跑存在 bf16 非确定性，零点几个点的波动属噪声，别写成结论。
- API 一路**未运行**（无 `LLM_API_KEY`），`outputs/metrics/api_zero_*.json` 明确写 `NOT_RUN`。补 key 后 `python scripts/run_all.py --step llm` 即可。
- `*_gate3` 的阈值扫过测试集，是事后诊断。
- 判别式只训练 2 epoch（week7 主线是 3 epoch），所以这里的绝对值略低于 week7 的 BERT+CRF 0.5815；本实验的目的是**相对比较**，不是刷新主项目指标。
- 权重不随包分发：rbt3 / MacBERT / Qwen3 复用 week6 `work/pretrain_models/`，路径由 `config.PRETRAIN_ROOTS` 逐级解析。

## 七、复现

```bash
conda activate py312
python data_prep/build_variants.py
python scripts/run_all.py            # 全流程：规则秒级 / 4 次训练约 20 分钟 / LLM 约 6 分钟
python scripts/run_all.py --step compare   # 只重算 60 条投影与汇总表
```

关键产物：`data/*/`（变体）、`outputs/preds/`（逐条预测）、`outputs/metrics/`（统一指标 + `summary.md`）、`outputs/logs/`（训练日志与 LLM 原始输出）、`outputs/ckpt/`（最优权重）。

## 八、目录

```
work/
├── config.py                 路径 / 模型 / 抽样配置
├── common/                   bridge.py（复用 week7 common）· eval_local.py（落盘 + 篇章 F1）
├── data_prep/build_variants.py  三个变体的构建与统计
├── data/{heldout,doc}/       新数据；sent 直接引用 week7 data/processed
├── methods/                  rule_baseline / train_bio / qwen_zero / api_zero / doc_gate
├── eval/compare.py           60 条投影 + 汇总表
├── scripts/run_all.py        一键复现（可 --step）
└── outputs/                  preds / metrics / logs / ckpt
```

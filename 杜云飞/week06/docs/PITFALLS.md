# 踩坑表（现象 → 根因 → 修法）

> 只记录**实际发生过**的坑，每条都能对应到一次真实的报错或一次实测数据。
> 这份表的价值在于：下次遇到同样现象时不用再花半天定位。

## 一、环境与工具链

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `ValueError: Attempting to unscale FP16 gradients.` | 模型先 `.half()` 再配 `GradScaler`：GradScaler 要求梯度是 fp32，模型半精度后梯度也是 fp16 | **权重保持 fp32，只用 autocast**。或直接用 bf16（不需要 scaler），本机实测 bf16 117 vs fp16 121 samples/s，性能几乎无损 |
| 2 | `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()` | fp16 + GradScaler 时首个 step 若遇 inf 会被跳过，scheduler 却仍前进 | 同上：改 bf16 去掉 scaler，这类"跳过"问题整体消失 |
| 3 | `UserWarning: expandable_segments not supported on this platform` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 只有 Linux 支持 | 按平台设置（`os.name != "nt"` 时才设），Windows 上直接跳过 |
| 4 | `pip install jieba` 报 `Tunnel connection failed: 502` | 默认 PyPI 源走代理不通 | 换清华源：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 5 | `huggingface.co` 返回 502，`from_pretrained("hfl/xxx")` 卡住 | 直连被墙/被代理拦截 | 设 `HF_ENDPOINT=https://hf-mirror.com`；或先用 `scripts/04_fetch_backbones.py` 落到本地再离线加载 |
| 6 | 日志里大量 `Redirects are currently not supported in Windows` | torch.distributed 在 Windows 的无害提示 | 忽略，或用 `grep -viE "Redirects"` 过滤噪音 |

## 二、显存与吞吐（8GB 卡上最危险的一类）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 7 | 显存没 OOM，但训练突然慢 3 倍 | **显存越线后不是报错，是静默降级**：bs=64/L=256 峰值 7.24GB，吞吐从 121 掉到 36 samples/s；LoRA bs=8 峰值 8.75GB 也"成功"了（溢到 WDDM 共享内存） | 每 50 step 断言 `max_memory_allocated < 6.5GB`，越线直接抛错；把 bs/max_len 调到 6.5GB 以内 |
| 8 | 偶发 `CUBLAS_STATUS_INTERNAL_ERROR` | 显存贴顶时 WDDM 分页导致的 kernel 报错，**不是**驱动/算子不兼容 | 按"显存不足"处理，减小 batch 或 max_len；不要花时间去查 CUDA 版本 |
| 9 | LoRA `bs=4/L=384` 只比 `bs=2` 快一点点 | 峰值 7.05GB 已越线，吞吐 6.1 vs 12.1 samples/s | 固定 `bs=2 / L=384 / accum=8`，用梯度累积换等效 batch |
| 10 | 0.5B 模型全量微调必 OOM | 494M × (2+2+8)B ≈ 5.9GB 仅权重+梯度+优化器状态，再加激活与 `B×L×151936` 的 logits（bs=8/L=256 光 logits 就 1.2GB） | **只做 LoRA**（可训练 1.1M，0.22%） |

## 三、transformers 5.x 的 API 漂移

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 11 | `apply_chat_template` 返回 dict，`prompt[:max_len]` 报错 | 5.x 默认 `tokenize=True` | 显式 `tokenize=False` |
| 12 | 生成结果前面全是 PAD / 解码出乱码 | 右 padding 时 PAD 会插在答案之前 | `tokenizer.padding_side = "left"` |
| 13 | 解码结果把 prompt 里的类名当答案 | `batch_decode` 把整段都解了 | 只解 `out[:, input_len:]` |
| 14 | Qwen2 的 pad 与 eos 不是同一个 token | Qwen2 `pad=<\|endoftext\|>`、`eos=<\|im_end\|>` | collate 用 `pad_token_id`，生成也显式传 `pad_token_id`；SFT 的 labels pad 用 `-100` |
| 15 | Qwen3 输出里带大段思考导致解析全失败 | 训练/推理模板不一致（一个开 thinking 一个关） | `chat_template_kwargs={"enable_thinking": False}`，训练与推理**必须同参** |
| 16 | `TrainingArguments(evaluation_strategy=...)` 报未知参数 | 5.x 改名 `eval_strategy`、`torch_dtype`→`dtype` 等 | **完全不用 Trainer**，自写训练循环（顺带获得显存断言与分组 lr 能力） |
| 17 | `BertTokenizer` 是否存在？本地目录没有 `tokenizer.json` | 5.x 已把 slow tokenizer 统一为 rust 后端 | 实测 `AutoTokenizer.from_pretrained(本地目录)` 正常，680~800 docs/s，无需转换 |

## 四、DeepSeek API

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 18 | 返回 `content` 为空，只有 `reasoning_content` | `deepseek-flash`（V4.1-Flash）是推理模型，默认开思考；`max_tokens=8` 时思考没写完就被截断 | 关思考：`{"thinking": {"type": "disabled"}}`；或 `{"reasoning_effort": "low"/"high"}`（**参数名不是 `effort`**，写错 400） |
| 19 | 账单远超预期 | 成本 ≈ prompt 长度 × 请求数，few-shot 会把单条 prompt 从 ~110 token 拉到 ~800 token（≈7 倍） | `--estimate` 先试算；响应磁盘缓存；few-shot 用短示例；评估只跑 1000 条抽样 |
| 20 | `os.environ` 取不到 `DEEPSEEK_API_KEY` | 用户级环境变量，已开的终端不继承 | 兜底读注册表 `HKCU\Environment`（本项目的 `get_api_key()` 已实现） |

## 五、数据与口径

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 21 | zip 里的中文目录名是乱码（`Σ╜ôΦé▓`），14 类全变成"未知类别" | 文件名是 **UTF-8 字节但没打 0x800 标志位**，zipfile 按 cp437 解码。注意不是 GBK——按 GBK 还原会得到另一种乱码（`绉戞妧`），我第一次就修错了方向 | 先 `encode("cp437").decode("utf-8")`，失败再试 gbk；带 0x800 标志的直接用 |
| 21b | 成员数 167 万，比官方"74 万篇"多一倍 | zip 里混着 83 万个 `__MACOSX/._xxx`（AppleDouble 资源叉，每个 223 字节） | 跳过 `__MACOSX` 与 `._` 前缀；实测真实文档 836,089 篇 |
| 22 | 想解压 74 万 txt 时发现 NTFS 上极慢 + 占 1.8GB | 小文件数量太多 | **直接流式读 zip**，转成一个 jsonl；原始 zip 可保留，不需要解压 |
| 23 | 三个模型的预测交集算出来是 0 | 各方法用"子集内下标"当 id，抽样顺序不同就对不上 | 所有预测用 `all.jsonl` 的**全局自增 id** 对齐 |
| 24 | 长尾类的 F1 波动很大，"提升了 3pp"不可信 | 按比例抽测试集时"时尚"只有 ~177 条，95%CI 半宽 ≈ ±7pp | 双测试集：`test_bal`（每类 ≥300）算 macro-F1 作主指标，`test_nat_10k` 算 acc |
| 25 | 生成式方法 acc 虚高 | 把解析失败样本剔出分母 | 主表口径 = 全样本计入分母，`parse_fail_rate` 单列 |
| 26 | 同一条命令跑两次指标差 0.5pp | 没设 seed（`shuffle=True` + 未固定随机源） | `set_seed(42)` + `cudnn.deterministic=True`；对比跑 `eval/paired_test.py` |

## 六、长文本特有的坑

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 27 | `max_len=256` 训出来的模型"看不懂全文"，但没人能解释 | THUCNews 正文 700~1500 字，中文 1 字 ≈ 1 token，256 只覆盖 17%~37% | 先跑 `scripts/03_len_stats.py` 给出截断覆盖率；用 `head_tail`（前 60% + 后 40%）与 `title_only` 做对照 |
| 28 | 标题明明是强信号却没被利用 | 直接把正文喂进去，标题被正文淹掉 | 构造输入时**标题永远拼在最前**（`common/text.py` 统一处理，三条方法线共用） |
| 29 | "100k×3ep" 的 BERT 只跑了 2 分钟，val 卡在 0.44 | tokenize 缓存键里只有 split 名，**没有数据指纹**：冒烟时缓存了 1200 条合成样本，正式跑时直接命中旧缓存，于是"100k 训练"实际只有 1200 条 | 键里加 `文件大小 + mtime`；加载后校验条数是否与当前切分一致，不一致就重算并打印 `[缓存失效]` |

## 七、实验编排的坑（队列本身也会错）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 30 | 明明已完成的 `rule` / `tfidf`，每次跑 `10_run_queue.py` 都重跑一遍（白烧 20 分钟） | `guess_run_id()` 里"取不到 `--variant` 就 return None"的判断**放在 script 分支之前**；而 `rule`/`tfidf` 的命令行本来就没有 `--variant`，于是永远推不出 run_id、永远判为"未完成" | 把 `variant is None` 的兜底**下移到 script 分支之后**；这两支的 run_id 由方法名+超参直接构造 |
| 31 | 消融矩阵 20 个格子平铺成一张表，看不出"哪个改动有用" | 汇总脚本只按 macro-F1 排序输出 | `compare.py` 读 `meta.json` 的 `hyperparams` **自动求差**：只列真正变过的字段，并按 ΔF1 排序；同时用"同配置换 seed"的 run 算出 **seed 标准差**，凡 |Δ| < 2σ 的格子显式标注为噪声 |
| 32 | 后台任务日志永远是空的，看不到进度 | 命令末尾接了 `\| tail -60`，`tail` 要等 stdin 关闭才输出，全程缓冲 | 进度看 `runs/_queue.log`（脚本自己 `flush`）；不要给长任务套 `tail` |
| 33 | 用 `sleep 600` 等训练，命令被 SIGTERM 杀掉 | Bash 工具默认超时 120s，`sleep` 不会自动转后台 | 跨分钟等待要显式 `timeout`，或改用"检查型命令"轮询产物文件 |
| 34 | DeepSeek 3-shot 的 macro-F1 只有 0.246、拒答 75%，看起来像"给例子反而变差" | **API 余额耗尽**：`HTTP 402 Payment Required` 被 `call_once` 当成普通失败返回空串，主流程又把空串记成 `abstain`。3126/4200 条实际是账单错误。余额不足时并发额度被压到 5~9，`--concurrency 12` 必然 402 | 402 立即 `SystemExit`（重试无意义）；429 单独用更长退避；新增 `--max_errors`，错误条数超阈值整批中止并声明"本次结果不可对比"。**教训：任何"模型表现断崖"先看 `raw` 字段，别先解释模型** |

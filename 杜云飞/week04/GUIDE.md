# GUIDE.md —— 代码导读 + 调试手册

这份文档解决两个问题：

1. **这 7 个代码文件按什么顺序读，每个文件读到什么程度算读完**
2. **读不懂或跑不通时，怎么一步步把问题逼出来**

前提：`conda activate py312`。下面所有 `python xxx.py` 都指这个环境。

---

## 0. 先建立总体印象

这个项目做的事情只有一句：**让模型学会"看到上文，猜下一个字"，然后把猜字的过程做成可调、可看、可测的东西。**

```
                     ┌──────────────┐
   原始 txt  ───────▶│prepare_novel │──▶ corpus_novel.txt（一行一篇）
                     └──────────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │      train.py      │
                                    │  切窗口 → 前向 →    │
                                    │  cross_entropy →   │
                                    │  反向 → AdamW       │
                                    └─────────┬─────────┘
                                              │ ckpt_novel/best.pt
                        ┌─────────────────────┼─────────────────────┐
                        │                     │                     │
                 ┌──────▼──────┐      ┌───────▼───────┐     ┌───────▼───────┐
                 │  sample.py   │      │    chat.py    │     │  scaling.py   │
                 │ 六种采样对比  │      │  HTTP + SSE   │     │  多档规模对照  │
                 └─────────────┘      └───────┬───────┘     └───────────────┘
                                              │
                                      ┌───────▼───────┐
                                      │   chat.html   │
                                      │  单文件前端    │
                                      └───────────────┘
```

**所有文件共享同一份模型定义**（`model.py`），区别只在于"怎么用它"：

| 文件                 | 行数 | 一句话                          | 读它要回答的问题                                          |
| :------------------- | ---: | :------------------------------ | :-------------------------------------------------------- |
| `model.py`         |  615 | 分词器 + Transformer + 采样算子 | 一个 token 从文字变成概率，中间经过哪几步？               |
| `train.py`         |  425 | 预训练主循环                    | 一次 `loss.backward()` 之前，数据是怎么变成 (x, y) 的？ |
| `sample.py`        |  342 | 采样实验台                      | 同一份权重，为什么换个参数就像换了个模型？                |
| `chat.py`          |  387 | 网页后端                        | 打字之后，字符是怎么一个字一个字蹦到屏幕上的？            |
| `chat.html`        |  892 | 网页前端                        | 一条消息在浏览器里活多久、存在哪？                        |
| `scaling.py`       |  208 | 规模对照                        | 参数量翻倍，loss 降多少？                                 |
| `prepare_novel.py` |  255 | 小说语料清洗                    | 830MB 脏文本怎么变成 248MB 干净语料？                     |

---

## 1. 阅读顺序（6 步，约 3 小时）

不要按文件列表顺序读，按**数据流**读。

### 读之前：先认识源码里的 `#region`

所有 `.py` 和 `chat.html` 都按同一套教学式注释写：`#region 编号 标题` / `#endregion` 把文件切成可折叠的块，
块标题末尾的【】标出里面有什么（`【函数:xxx(参数),return 返回值】`、`【类:xxx】`、`【集合:变量名】`）。
行尾注释跟的是"为什么这么写"和**形状变化**，比如 `(B,T,D) → (B,H,T,hd)`。

- VSCode：编辑器原生支持 `#region` 折叠，左侧行号旁的箭头就能折；`Ctrl+F` 搜 `#region` 可列大纲。
- PyCharm：相同语法，`Ctrl+Alt+.` 折叠。
- 想跳读：直接搜 `#region 4.1`、`#region 12.7` 这种编号，比翻页快。

### 第 1 步：`python model.py`，先看它自己会说什么

```bash
python model.py
```

它会打印参数量、跑一次前向、并把 **Causal Mask 画成下三角**：

```
参数量: 15,152,640
logits (2, 8, 2579) loss 6.234...
Causal Mask（1=可见，0=屏蔽），第 i 行=第 i 个 token 的可见范围：
   ■ · · · · · · ·
   ■ ■ · · · · · · 
   ■ ■ ■ · · · · · ·
```

**这一步的目的是拿到"坐标系"**：你后面读的所有代码，都在服务这个 `(2,8,2579)` 的张量。

> 这里的参数量 15.15M 比正式模型 16.14M 小，因为 `model.py` 末尾写死的是 `vocab_size=2579`，
> 而实际语料词表是 5153 —— 词嵌入差了 0.99M 参数。这种"示例配置 ≠ 实际配置"的差异，
> 恰恰是读代码时要留意的地方。

同时记住三个形状，全项目反复出现：

| 张量       | 形状          | 含义                                           |
| :--------- | :------------ | :--------------------------------------------- |
| `idx`    | `(B, T)`    | B 条样本，每条 T 个 token id                   |
| `logits` | `(B, T, V)` | 每个位置对 V 个候选字的打分                    |
| `loss`   | 标量          | 把 `(B*T, V)` 和 `(B*T,)` 拉平后算的交叉熵 |

### 第 2 步：`model.py` 读三遍

**第一遍只看结构，不看数学**（15 分钟）：从 `GPT.__init__`（第 363 行）往下读，把 `embed → 8×Block → norm_f → lm_head` 这条主线记住。此时你应该能回答："16.14M 参数都花在哪了？"

答案：`8 层 × (Attn 1.18M + FFN 1.18M) ≈ 18.9M`，但**词嵌入和输出层共享同一块权重**（第 372 行 `weight tying`），所以省掉一份 2M。可以用一行验证：

```bash
python -c "
import sys; sys.path.insert(0,'.')
from model import GPT, scale_config
m = GPT(scale_config('base', 5153, 512))
for n,p in m.named_parameters(): print(f'{n:28s} {tuple(p.shape)}')
"
```

你会发现列表里**没有 `lm_head.weight`** —— 它和 `embed.weight` 是同一个 tensor。这就是"参数量 16.14M 而不是 18M"的原因。

**第二遍看 attention（第 194–283 行），这是全项目最难也最重要的一段**（40 分钟）。

四个关键点，每个都有对应的验证方法：

| 关键点                                     | 代码位置       | 怎么验证你读懂了                                              |
| :----------------------------------------- | :------------- | :------------------------------------------------------------ |
| mask 是 `cols <= rows`，不是 `tril`    | 第 267–274 行 | 把 `cfg.fast_attn` 设 False，跑一遍，输出应该几乎一样       |
| `is_causal=True` 只在 `T == Tk` 时能用 | 第 249 行      | 用 KV Cache 生成时 `T=1, Tk=400`，走的是 `attn_mask` 分支 |
| RoPE 的 offset = 已缓存长度                | 第 233 行      | 生成第 100 个字时，它的绝对位置必须是 100，不是 0             |
| KV Cache 用 `torch.cat` 拼接             | 第 243–244 行 | 这就是为什么 `bench_kv` 里 batch=1 反而更慢                 |

**这里有个绝佳的调试练习**：两种 attention 实现（融合核 vs 手写 masked_fill）在数学上等价，所以

```bash
python -c "
import torch, sys; sys.path.insert(0,'.')
from model import GPT, GPTConfig
torch.manual_seed(0)
cfg = GPTConfig(vocab_size=100, block_size=16, d_model=64, n_head=4, n_layer=2)
m = GPT(cfg).eval()
x = torch.randint(0, 100, (1, 8))
a,_,_ = m(x)
m.cfg.fast_attn = False
for b in m.blocks: b.attn.fast_attn = False
b,_,_ = m(x)
print('最大差异', (a-b).abs().max().item())   # 应该在 1e-6 量级
"
```

如果差异很大，说明 mask 写错了（经典的 off-by-one）。**这是以后自己写 attention 时最该先跑的测试。**

**第三遍看采样（第 450–544 行）**（30 分钟）。`filter_logits` 里四步的顺序是被刻意固定的：

```
temperature → ban_ids → 重复惩罚 → top-k → softmax → top-p
```

顺序换了结果就变。比如先 top-p 再 temperature，被截掉的候选就再也回不来了。这段的调试方法是**打印中间结果**：

```bash
python sample.py --show-probs
```

它会直接打印 next-token 的概率分布，你能看到 top-p 到底砍掉了哪些字。

### 第 3 步：`train.py`，重点是"数据怎么变成 (x, y)"

跳过 `main()`，先看这 4 个函数（第 53–157 行）：

```
load_docs  →  一行一篇文档
build_ids  →  每篇编码 + 结尾追加 <eos>，用 array('i') 省内存
LMDataset  →  切窗口：x = ids[i:i+T], y = ids[i+1:i+T+1]
lr_at      →  warmup + 余弦退火
```

**`LMDataset` 是整个预训练的核心，只有 20 行，值得逐字读**。它解释了为什么"训练信号密度 100%：每个位置都要预测下一个字"。

然后用 `main()` 验证你的理解：找到这三行，它们决定了模型学得好不好。

| 行                                                   | 内容             | 改了会怎样                |
| :--------------------------------------------------- | :--------------- | :------------------------ |
| `tokens_per_step = batch × seq_len × grad_accum` | 每步吃多少 token | 决定 lr 该多大            |
| `steps = args.tokens / tokens_per_step`            | 总步数           | 决定 lr 调度曲线形状      |
| `(loss / grad_accum).backward()`                   | 梯度累积要除     | 忘了除，等效 lr 变大 N 倍 |

**必做实验：单批过拟合**（这是判断"模型/代码有没有 bug"最快的办法，见第 3 节）。

### 第 4 步：`sample.py`，把"参数 → 现象"建立直觉

```bash
python sample.py --n 80                    # 六种策略一次跑完
python sample.py --mode temperature        # 只跑温度
python sample.py --show-probs              # 看概率分布
python sample.py --bench                   # KV Cache 加速比
```

读代码时只需关心三件事：

- `fmt()` 第 110 行：只是格式化输出
- `main()` 里每个 `if args.mode in (...)`：一个策略一段
- `bench_kv()` 第 277 行：**它对同一个模型跑 4 种配置并对比，这是"用测量代替猜测"的模板**

`bench_kv` 值得完整读一遍，因为它的结论是反直觉的：**batch=1 时 KV Cache 更慢**。原因是 16M 的小模型喂不饱显卡，每步耗时由 kernel 启动开销决定，跟算了多少 token 无关，而缓存还要多付 `torch.cat` 的代价。

### 第 5 步：`chat.py`，看"流式"是怎么做出来的

只需要读两段：

**`build_prompt()` 第 57 行**——它解释了为什么这个模型"不像 DeepSeek"：

```python
if mode == "chat":
    parts = [f"用户：{h['u']}\n助手：{h['a']}\n" for h in history]
```

Base Model 不认角色，它只是**模仿这个格式**。所以对话模式会跑偏，这是必然的。

**`run_generation()` 第 85 行**——流式的全部秘密在 `on_token` 回调：

```python
def on_token(tid):
    out_ids.append(tid)
    full = tok.decode(out_ids)      # 全量解码最省心（字符级，成本可忽略）
    delta = full[len(sent):]        # 只推新增的部分
    emit({"type": "tok", "t": delta})
```

三个工程细节，每个都能成为面试题：

1. **为什么全量解码再切前缀？** 因为字符级词表下 `decode` 是纯字符串拼接，成本远低于维护增量状态机的复杂度。
2. **怎么做到"停止"按钮真的能停？** `aborted()` 抛 `_Abort`，中断发生在下一个 token 处；而浏览器关页面时，写 socket 会抛 `BrokenPipeError`，服务端借此退出循环。
3. **`max_ctx - max_new` 为什么要留出空间？** 上下文窗口只有 512，如果 prompt 塞满 512，就没有位置放新生成的 token 了。

### 第 6 步：`chat.html`，最后读，但要读懂两件事

830 行里 200 行是 CSS，可以跳过。只看：

- `const S = {...}` 第 403 行：**所有状态都在这一个对象里**（会话列表、当前会话、参数）。状态集中是这类单文件应用能维护的前提。
- `async function send()` 第 631 行：读 SSE 的方式是 `fetch` + `ReadableStream`，不是 `EventSource`（因为要 POST）。

前端调试不用读代码，用**浏览器 F12**：

- Console 有红字 → JS 语法错误（改完 HTML 一定要看这里）
- Network 里 `/api/chat` 一直 pending → 后端在生成，正常
- Network 里 `/api/chat` 404 → 后端没起，页面会进"演示模式"（`demoType()` 第 763 行假装打字）

---

## 2. 每个文件"该改哪里"（把读懂变成会用）

读完不算完，能改才算懂。下面这些改动都很小，但每个都会让你更清楚它在干什么。

| 想验证的事            | 改哪里                                                              | 预期                                |
| :-------------------- | :------------------------------------------------------------------ | :---------------------------------- |
| 去掉 mask 会怎样      | `model.py` 第 269 行改成 `mask = torch.ones_like(cols <= rows)` | loss 会异常低（偷看答案），生成变乱 |
| 位置编码真的有用吗    | `apply_rope` 改成 `return x`                                    | loss 明显升高，长距离依赖失效       |
| Pre-Norm vs Post-Norm | `Block.forward` 改成 `x = self.n1(x + self.attn(x)[0])`         | 8 层还能训，16 层开始不稳           |
| 权重共享省了多少参数  | 删掉 `self.lm_head.weight = self.embed.weight`                    | 参数量 +2.6M，效果变化很小          |
| top-p 的边界行为      | `filter_logits` 里 `mask = prev < top_p` 改成 `cum <= top_p`  | P 很小时会一个候选都不留 → 除零    |
| 重复惩罚的副作用      | `chat.py` 把 `rep_penalty` 拉到 2.0                             | 常见字被压太狠，出现生僻字          |
| 上下文设 256 会怎样   | `--seq_len 256` 重训                                              | 单句仍通顺，但场景连贯性明显变差    |

---

## 3. 调试方法论：五层，从下往上

**核心原则：先证明最小的单元是对的，再往上叠。** 大模型的 bug 有一个特点——它不会崩，它只是"效果不好"。所以必须靠**可验证的不变量**来判断，不能靠眼睛。

### 第 0 层：环境体检（30 秒）

```bash
python -c "
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else '-')
try:
    import triton; print('triton', triton.__version__)
except ImportError: print('triton 缺失 → 无法用 --compile')
"
```

四行输出对应四个能力：能不能跑、跑在哪、能不能 bf16、能不能编译加速。

### 第 1 层：张量层 —— 用 assert 代替 print

不要 `print(shape)` 然后肉眼比对，直接写断言。这三个断言能挡住 80% 的形状错误：

```python
assert logits.shape == (B, T, V), logits.shape
assert loss.dim() == 0
assert torch.isfinite(loss)
```

**形状对不上时，最快的定位法是逐层 forward**：

```bash
python -c "
import torch, sys; sys.path.insert(0,'.')
from model import GPT, scale_config
m = GPT(scale_config('base', 5153, 512))
x = torch.randint(0, 5153, (2, 16))
cos, sin = m.rope_cos, m.rope_sin
h = m.embed(x)
for i, blk in enumerate(m.blocks):
    h, _ = blk(h, cos, sin)
    print(f'block {i}: {tuple(h.shape)} range[{h.min():.2f},{h.max():.2f}]')
"
```

注意 `range` 那一段：**它比形状更有用**。残差网络里激活值应该稳定在 ±10 以内；如果某层突然变成 ±1000，说明那里数值不稳（norm 位置错了，或者初始化没缩放）。

### 第 2 层：单批过拟合 —— 判断"代码有没有 bug"的黄金测试

**这是最值得记住的一招。** 拿一个 batch，反复训它 200 步，loss 必须降到接近 0：

```bash
python -c "
import torch, sys; sys.path.insert(0,'.')
from model import GPT, scale_config
torch.manual_seed(0)
m = GPT(scale_config('tiny', 5153, 512)).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 5153, (4, 128)).cuda()
y = torch.roll(x, -1, dims=1)                 # 目标 = 下一个字
for i in range(200):
    _, loss, _ = m(x, y)
    loss.backward(); opt.step(); opt.zero_grad()
    if i % 40 == 0: print(i, round(loss.item(), 4))
"
```

判据：

| 结果            | 结论                                       |
| :-------------- | :----------------------------------------- |
| loss 降到 < 0.1 | 前向、反向、标签对齐、优化器全部正确       |
| 停在某个值不动  | lr 太小，或 `y` 没做右移（标签错位一位） |
| 变 NaN          | lr 太大 / 忘了 `zero_grad` / bf16 溢出   |
| 降得极慢        | 梯度被 clip 死，或 embedding 没参与训练    |

**如果单批都过拟合不了，训练语料、轮数、参数量都不用看了——问题在代码里。**

### 第 3 层：训练层 —— 看曲线，不看单个数字

`metrics.jsonl` 每 60 步写一条。三个必看的组合：

```bash
python -c "
import json
rows=[json.loads(l) for l in open('ckpt_novel/metrics.jsonl',encoding='utf-8')]
for r in rows[::max(1,len(rows)//12)]:
    print(f\"{r['step']:>5} train {r['train_loss']:.3f} val {r['val_loss']:.3f} lr {r['lr']:.1e}\")
"
```

| 现象                            | 诊断                                | 处置                                                    |
| :------------------------------ | :---------------------------------- | :------------------------------------------------------ |
| train 和 val 同步下降，val 略低 | 健康（val 低是因为 dropout 关掉了） | 继续                                                    |
| val 触底后回升，train 继续降    | **过拟合**                    | 加数据/加 dropout/早停。参考 `docs/overfit_curve.png` |
| train 抖动剧烈                  | lr 太大                             | 降 lr 或加 warmup                                       |
| loss 完全不动                   | 数据坏了（全 `<unk>`）或 lr=0     | 检查 `len(tok.itos)` 和语料编码                       |
| `gnorm` 一直贴着 1.0          | 梯度被裁剪，实际 lr 变小            | 正常现象，不用管；一直 0.4 以下才说明梯度太小           |

### 第 4 层：推理层 —— 判断"是模型差还是用法差"

生成质量不好时，按这个顺序排除，**每一步都只改一个变量**：

```bash
# ① 换 greedy：排除采样的随机性
python sample.py --mode greedy --n 60

# ② 换长 prompt：排除冷启动（这是本项目踩过的坑）
python sample.py --prompt "民国二十三年的秋天，保定府城外的官道上尘土飞扬。天还没亮，福来茶馆的门板就卸了一半，老掌柜把茶碗往前一推，说道：“" --n 80

# ③ 关掉所有截断：看原始分布到底长什么样
python sample.py --show-probs
```

| 现象                    | 大概率原因                       | 验证                                  |
| :---------------------- | :------------------------------- | :------------------------------------ |
| 前几个字乱，后面正常    | 冷启动：prompt 太短              | 加长 prompt，立刻变好                 |
| 整段重复同一句          | 重复惩罚太低 / 温度太低          | `--rep 1.3 --temperature 1.0`       |
| 出现 `★★★`、生僻字 | 语料字符分布异常（不是模型问题） | `grep -o '★' corpus.txt \| wc -l`   |
| 跑到一半断在句号        | EOS 提前触发                     | `--ban_eos` / `stop_at_eos=False` |
| 上下文一长就忘          | `block_size=512` 截断          | 看界面右下角"上下文 xxx/512"          |

### 第 5 层：性能层 —— 三个数就够了

```bash
# ① 吞吐（唯一可信的指标）
grep "tok/s" data/*.log | tail -3

# ② 功耗墙 / 显存
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,power.limit --format=csv

# ③ 是不是被降频
nvidia-smi -q -d PERFORMANCE | grep -E "SW Power Cap|HW Slowdown"
```

**这一层有一个反直觉的坑，是本项目实测出来的**：`nvidia-smi` 报 `clocks.sm = 180 MHz`（满频 3090），但吞吐是正常的。这颗驱动在笔记本上读数不可信，**判断是否降频只能看 tok/s**。

性能调的优先级（每一条都是实测，不是经验）：

| 手段                           | 收益                                     | 代价              |
| :----------------------------- | :--------------------------------------- | :---------------- |
| `torch.compile`（需 triton） | **+57%**（73k→115k）              | 首次编译 1~3 分钟 |
| 改 batch                       | **负收益**（96 就爆显存，掉 6 倍） | —                |
| 降 `eval_iters`              | +5%                                      | 验证噪声变大      |
| 多进程 DataLoader              | +3%                                      | 内存多占一点      |
| 提功耗墙（需管理员）           | 未知，卡被锁 60W                         | 温度上升          |

---

## 4. 五个症状 → 五条排查路径

### 症状 A：训练跑完了，但生成的是乱码

按这个顺序，**每一步都记录结果**，不要跳：

1. `python sample.py --mode greedy` → 如果 greedy 也乱，排除采样问题，进入 2
2. 看语料字符分布：`python -c "import collections;print(collections.Counter(open('data/corpus_novel.txt',encoding='utf-8').read(200000)).most_common(15))"` → 如果前三名是符号，语料坏了
3. 算 tokens/参数：`总token / 参数量`。**< 2 就是严重欠训练**（第一版新闻模型是 0.5，所以出 `★★★★`）
4. 单批过拟合测试（第 2 层）→ 通过说明代码没问题，问题在数据量或容量

### 症状 B：loss 是 NaN

九成是这三个之一：

```python
# ① 学习率太大（尤其配 warmup=0）
python train.py --lr 3e-4 --warmup 200 ...

# ② 除以 0（grad_accum 或 top_p 归一化）
# 在 filter_logits 末尾加：assert probs.sum(-1).min() > 0.99

# ③ bf16 溢出（极少，但 logits 里有 -inf 时会）
# 临时改成 autocast(dtype=torch.float16) 或干脆不开
```

`torch.autograd.set_detect_anomaly(True)` 能定位到具体算子，但会慢 10 倍，只在找不到时用。

### 症状 C：网页打不开 / 白屏 / 没有回复

```bash
# ① 服务在不在
curl -s http://127.0.0.1:8000/api/info      # 应返回 JSON

# ② 端口被谁占了
netstat -ano | findstr :8000

# ③ 后端能生成吗（绕过前端）
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"prompt":"老掌柜把茶碗往前一推，说道：“","max_new":20,"mode":"continue"}'
```

如果 ③ 正常但页面不出字，问题在前端：**F12 → Console**。改过 `chat.html` 之后一定要看这里，一个混进去的 ASCII 引号就能让整个脚本停摆（本项目犯过）。

### 症状 D：OOM（CUDA out of memory）

`8.15 GB` 显存下，batch 和 seq_len 的组合是硬件决定的：

| batch × seq_len | 显存峰值 | 结论                    |
| :--------------- | -------: | :---------------------- |
| 32 × 512        |  3.79 GB | 安全                    |
| 64 × 512        |  7.22 GB | 勉强                    |
| 96 × 512        | 10.72 GB | **爆，掉 6 倍速** |

注意：**超了不会立刻报错，而是开始换页**，表现为"没崩但慢得离谱"。所以看到吞吐突然从 72k 掉到 6k，先怀疑显存，不是怀疑代码。

### 症状 E：中文变问号 / 乱码

网文 txt 有一半是 GBK。`prepare_novel.py` 的 `read_text()` 第 72 行就是干这个的：

```python
for enc in ("utf-8-sig", "utf-8", "gb18030"):
    try: ...
```

`gb18030` 是 GBK 的超集，必须放在最后，因为它几乎不会解码失败（会把任何字节流解成"看起来像字"的东西）。

---

## 5. 调试工具箱速查

| 需求             | 做法                                                                                                         |
| :--------------- | :----------------------------------------------------------------------------------------------------------- |
| 断点             | 代码里插 `breakpoint()`，跑 `python -m pdb xxx.py`（`n` 下一步、`s` 进函数、`p 变量`、`c` 继续） |
| 只跑几行看变量   | `python -i -c "..."` 进交互环境                                                                            |
| 计时             | `python -X importtime`、`time.time()` 打点、`torch.cuda.synchronize()` 后再计时                        |
| GPU 到底在算什么 | `torch.profiler.profile(activities=[ProfilerActivity.CUDA])`                                               |
| 显存泄漏         | `torch.cuda.memory_summary()`、`reset_peak_memory_stats()`                                               |
| 复现性           | `torch.manual_seed(0)` + `torch.use_deterministic_algorithms(True)`                                      |
| 看梯度有没有传到 | `for n,p in m.named_parameters(): print(n, p.grad is None)`                                                |
| 参数量分布       | `sum(p.numel() for p in m.parameters())` 按模块分组打印                                                    |

---

## 6. 十个动手实验（做完这套，这个项目就真正是你的了）

每个实验都给了**预期结果**。结果不符，就回到第 3 节找对应的排查层。

|  # | 实验                  | 命令 / 改动                             | 预期                              |
| -: | :-------------------- | :-------------------------------------- | :-------------------------------- |
|  1 | 确认 mask 正确        | 第 1 节里的 fast_attn 对比              | 差异 < 1e-5                       |
|  2 | 单批过拟合            | 第 2 层的脚本                           | 200 步内 loss < 0.1               |
|  3 | 语料字符分布          | `Counter(open(...).read(200000))`     | 中文常用字，不是符号              |
|  4 | 温度扫描              | `python sample.py --mode temperature` | T=0.5 保守，T=1.5 跑偏            |
|  5 | Top-P 和 Top-K 的区别 | `--mode top_p` / `--mode top_k`     | 分布尖锐时 K 浪费候选，P 自动收缩 |
|  6 | KV Cache 什么时候有用 | `python sample.py --bench`            | batch=1 更慢，batch=8 快 2~3 倍   |
|  7 | 冷启动                | 短 prompt vs 长 prompt                  | 长 prompt 明显更通顺              |
|  8 | 吞吐对照              | `--compile` 开关各跑 3 分钟           | 开编译约快 1.5 倍                 |
|  9 | 过拟合曲线            | 看 `docs/overfit_curve.png`           | val 第 20 epoch 后一路涨到 1364   |
| 10 | 规模对照              | `python scaling.py --tokens 8e6`      | 参数量越大 loss 越低，但不是线性  |

---

## 7. 读完自测（能答上就是真懂）

1. `logits` 的形状是 `(B,T,V)`，为什么算 loss 要先 `view(-1, V)`？
2. `weight tying` 省了多少参数？为什么它不会伤害效果？
3. `is_causal=True` 为什么在 KV Cache 的 decode 阶段不能用？
4. `filter_logits` 里四步的顺序换一下会发生什么？
5. 为什么 `LMDataset` 的 `y` 是 `ids[i+1 : i+T+1]`，而不是别的切法？
6. 梯度累积时为什么要 `loss / grad_accum`？
7. `array('i')` 和 Python `list` 在 1 亿字符上差多少倍内存？
8. 为什么 batch=1 时 KV Cache 反而更慢？
9. 界面里"上下文 352/512"这个数字是怎么算出来的？超了会怎样？
10. 为什么 16M 参数的模型"语法对、事实错"？这和参数量的关系是什么？

<details>
<summary>答案要点（先自己想，再展开）</summary>

1. 交叉熵要求 `(N, C)` 的输入和 `(N,)` 的标签，`B×T` 个位置每个都是一个独立分类任务。
2. 省掉一份 `V×D` ≈ 2.0M 参数；因为输入嵌入和输出投影学的是同一个"词向量空间"，共享反而让表征更一致。
3. 因为 decode 时 `q` 只有 1 行长，`k/v` 有 `Tk` 行，`is_causal` 的"右上对齐"假设不成立，必须显式给 `cols <= rows` 的 mask。
4. 先 top-p 再 temperature：被截掉的候选永远回不来，温度失去意义；先重复惩罚再温度：惩罚幅度会被温度缩放。
5. `y[t] = x[t+1]`，即"看到第 t 个字，预测第 t+1 个"；错一位就是标签泄漏或全错。
6. 因为 `backward()` 会把梯度**累加**，不除的话等效学习率放大 `grad_accum` 倍。
7. `list` 每个 int 是 28 字节的堆对象，`array('i')` 是 4 字节，**7 倍**（1 亿字符：3GB vs 0.4GB）。
8. 小模型每步耗时由 kernel 启动开销主导，跟算多少 token 无关；缓存还要多做一次 `torch.cat`。
9. `len(tok.encode(prompt))`；超过 `block_size` 时 `build_prompt` 从**末尾**截断，丢掉最早的几轮。
10. 语法/搭配是高频统计规律，几千万 token 就能学到；事实是低频长尾信息，需要容量去"记"，16M 参数装不下。

</details>

---

## 8. 一句话总结

**读代码的顺序是从张量到网页，调 bug 的顺序是从最小单元到整体。**

这个项目里所有"效果不好"的问题，最后都归结成三个可测量的数字：

- **tokens/参数**（够不够训练）——低于 2 一定崩，20 附近最优
- **val PPL**（学得好不好）——从 60 降到 20 是"语法通了"，再往下才谈语义
- **tok/s**（算力用没用上）——它决定你能喂多少 token，而它受功耗墙和显存限制

看懂了这三个数字怎么来的，也就看懂了这 7 个代码文件。

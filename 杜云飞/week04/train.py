"""
train.py —— 预训练（Pre-training）脚本：Next-Token Prediction，只做预训练，不做 SFT。
================================================================================
对应课程知识点：
    · 03 节  预训练 loss：每个 token 都参与计算（训练信号密度 100%）
    · 05 节  Scaling Law：改 --preset 就能对比 N 与 loss 的幂律关系
    · 02 节  Causal Mask + Attention 由 model.py 实现

工程上照搬真实大模型训练的做法（只是规模缩小 100 倍）：
    bf16 混合精度 · AdamW + 余弦退火 + 线性 warmup · 梯度裁剪 · 梯度累积
    · 文档级验证集切分（防泄漏）· 定期评估 PPL · 保存最优 ckpt

用法：
    conda activate py312
    python train.py                          # 默认配置，约 40 分钟
    python train.py --preset small --minutes 3   # 快速跑通
    python train.py --preset base --tokens 60e6  # 指定 token 预算
================================================================================
"""

#region 0. 导入库   【模块:标准库 argparse/array/json/math/os/random/tempfile/time + torch】
import argparse
import array
import json
import math
import os
import random
import tempfile
import time

# triton 在 Windows 上读环境变量时按 UTF-8 解码，中文路径会炸；指定 ASCII 缓存目录绕开
# 注意必须放在 import torch 之前：triton 是在 torch 初始化时被拉进来的
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton_cache"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import CharTokenizer, GPT, GPTConfig, EOS, UNK, save_ckpt, scale_config
#endregion

#region 0.1 路径常量   【常量:HERE=脚本目录，CORPUS=默认语料】
HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录，所有相对路径都从这算
# 默认语料指向第四周那份小语料；正式训练用 --corpus 指定小说语料
CORPUS = os.path.join(HERE, "..", "第四周", "字符集语言模型+拼音输入法", "04_language_corpus.txt")
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据
# ══════════════════════════════════════════════════════════════════════════════
#region 1.1 读语料：一行一篇文档   【函数:load_docs(路径),return 文档列表】
def load_docs(path):
    """按行读入语料，每行当作一篇文档（这份语料本身是"标题\n正文\n"的结构）。

    errors="ignore"：语料里偶尔混进坏字节，直接跳过而不是整份报错退出。
    返回：字符串列表，长度 = 文档数。
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        docs = [ln.strip() for ln in f.read().splitlines()]  # 列表，每项是一篇文档
    return [d for d in docs if d]  # 去掉空行，空文档会让 loss 变成 NaN
#endregion

#region 1.2 文档 → id 流   【函数:build_ids(文档列表,分词器),return torch.int32 一维张量】
def build_ids(docs, tok):
    """把每篇文档编码成 id，并在结尾追加 <eos>。
    这就是课程里说的"停止符"——模型因此学会在合适的位置停下，而不是无限续写。

    用 array("i") 而不是 Python list 攒：1 亿字符的语料用 list 会先吃掉 3 GB 内存
    （每个 int 都是 28 字节的堆对象），array 只占 4 字节/个。

    返回：(总 token 数,) 的 int32 张量，所有文档首尾相连挤在一条流里。
    """
    stoi, get = tok.stoi, tok.stoi.get  # 绑定成局部变量，循环里少查两次属性
    buf = array.array("i")  # 4 字节整数数组，装所有 id
    for d in docs:
        buf.extend([get(c, UNK) for c in d])  # 逐字符查表，查不到用 UNK=1
        buf.append(EOS)  # 每篇结尾补停止符，这是"文档边界"的唯一标记
    # frombuffer 是零拷贝视图，但底层内存归 array 管，所以要 clone 一份给 torch 持有
    return torch.frombuffer(buf, dtype=torch.int32).clone()
#endregion

#region 1.3 语言模型数据集   【类:LMDataset】
class LMDataset(Dataset):
    """语言模型数据集：每条样本 = 从长 id 流里随机切一段 (x, y)，y 是 x 右移一位。

        x = ids[i]     ... ids[i+T-1]
        y = ids[i+1]   ... ids[i+T]      ← "给定前 t 个 token，预测第 t+1 个"

    注意位移是在这里做的，不是在前向里 —— 所以 model.forward 里没有任何 shift 操作。
    """

    #region 1.3.1 构造函数：记录 id 流与步长   【函数:__init__(id流,序列长度,步长),return None】
    def __init__(self, ids, seq_len, stride=None):
        self.ids = ids  # 一维 int32 张量，全部文档首尾相连
        self.seq_len = seq_len  # T：每条样本多长
        self.stride = stride or seq_len  # 相邻样本的起点间隔；训练用 T（不重叠），验证用 T/4（重叠更密）
        # 能切出多少条样本：总长减去一条的长度和 1 个标签偏移
        self.n = max(1, (len(ids) - seq_len - 1) // self.stride)
    #endregion

    #region 1.3.2 样本数   【函数:__len__(),return int】
    def __len__(self):
        """DataLoader 靠它决定一个 epoch 有多少 batch。"""
        return self.n
    #endregion

    #region 1.3.3 取一条样本   【函数:__getitem__(下标),return (x (T,), y (T,))】
    def __getitem__(self, i):
        """切出第 i 条样本：x 和 y 相差一个位置。"""
        s = i * self.stride  # 本条样本的起点
        # ids 用 int32 存（省一半内存），喂给 embedding 前再转 long
        x = self.ids[s: s + self.seq_len].long()  # (T,) 输入
        y = self.ids[s + 1: s + self.seq_len + 1].long()  # (T,) 标签，整体右移一位
        return x, y
    #endregion
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 2. 学习率调度：线性 warmup + 余弦退火到 10%
# ══════════════════════════════════════════════════════════════════════════════
#region 2.1 学习率函数   【函数:lr_at(当前步,总步数,基础学习率,warmup步数,最低比例),return float】
def lr_at(step, total, base_lr, warmup, min_ratio=0.1):
    """warmup 阶段线性升温（防止训练初期梯度过大把模型炸掉），
    之后按余弦曲线缓降到 base_lr*min_ratio（后期小步长精修）。

    为什么需要 warmup：AdamW 一开始的动量估计是空的，前几步的有效步长会异常大，
    直接从 6e-4 起步很容易一步把权重推到坏区域，之后再怎么训都救不回来。
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)  # 线性升温：第 0 步是 1/warmup，最后一步到满
    p = (step - warmup) / max(1, total - warmup)  # 进度 0→1（warmup 之后归一化）
    # 余弦从 1 平滑降到 min_ratio：0.5*(1+cos(πp)) 在 p=0 时是 1，p=1 时是 0
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
#endregion

#region 2.2 验证：算 val loss   【函数:evaluate(模型,验证loader,设备,迭代数),return float val_loss】
@torch.no_grad()  # 验证不需要梯度，省显存也更快
def evaluate(model, loader, device, iters):
    """在验证集上跑 iters 个 batch，返回按 token 数加权的平均 loss（不是 PPL）。"""
    model.eval()  # 关 dropout
    tot, n = 0.0, 0  # tot 是 loss 的加权和，n 是 token 总数
    it = iter(loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            break                 # 验证集只有这么多 batch，跑完即可
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss, _ = model(x, y)  # 前向只取 loss
        # 为什么乘 y.numel()：每个 batch 的 token 数可能不同，直接平均 batch 会让长 batch 权重偏低
        tot += loss.item() * y.numel()
        n += y.numel()
    return tot / max(n, 1)  # 兜底 1，防止 n=0 时除零
#endregion

#region 2.3 对照组：LSTM 语言模型   【类:LSTM_LM】
class LSTM_LM(nn.Module):
    """对照组：第四周那套 LSTM 语言模型（结构完全一样，只是把容量放大到与 Transformer 同量级）。
    它的存在只为了回答一个问题：同一个语料、同样的 token 预算，换成 Transformer 到底强多少？

    结构：Embedding → 多层 LSTM → Linear(V)；没有 attention，也没有位置编码（RNN 天生有序）。
    """

    def __init__(self, vocab_size, embed_dim, hidden, layers, dropout):
        super().__init__()

        class C:  # 只为让 train 循环统一接口
            pass
        self.cfg = C()  # 用一个空类伪装成 GPTConfig，这样 train 循环不用写两套分支
        self.cfg.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)  # (B,T) → (B,T,E)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        # batch_first=True 表示输入是 (B,T,E) 而不是 (T,B,E)；单层 LSTM 加 dropout 会报警告，所以判了 layers>1
        self.fc = nn.Linear(hidden, vocab_size)  # (B,T,hidden) → (B,T,V)

    def num_params(self):
        """和 GPT 同名的方法，方便 train 循环统一打印参数量。"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False):
        """接口与 GPT.forward 对齐：多出来的两个参数直接忽略（LSTM 没有 KV Cache 概念）。"""
        out, _ = self.rnn(self.drop(self.embed(idx)))   # 循环结构：必须按时间步串行推进
        logits = self.fc(self.drop(out))  # (B,T,V)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, None  # 第三个返回值恒为 None，占住 GPT 的"缓存"位置
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 3. 主流程
# ══════════════════════════════════════════════════════════════════════════════
#region 3. 主流程   【函数:main(),return None】
def main():
    #region 3.1 命令行参数   【集合:args】
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base",
                    choices=["tiny", "mini", "small", "base", "large"])  # 五档规模，改它就能做 Scaling 对照
    ap.add_argument("--arch", default="gpt", choices=["gpt", "lstm"], help="lstm = 第四周对照模型")
    ap.add_argument("--lstm_hidden", type=int, default=768)
    ap.add_argument("--lstm_layers", type=int, default=3)
    ap.add_argument("--seq_len", type=int, default=512, help="上下文长度 T")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=2, help="梯度累积步数，等效放大 batch")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--dropout", type=float, default=0.12)
    ap.add_argument("--weight_decay", type=float, default=0.1)  # AdamW 的解耦权重衰减，正则化用
    ap.add_argument("--warmup", type=int, default=120)  # 前 120 步线性升温
    ap.add_argument("--tokens", type=float, default=48e6, help="总 token 预算（决定训练多久）")
    ap.add_argument("--minutes", type=float, default=None, help="给定则按墙钟时间反推 token 预算")
    ap.add_argument("--tps", type=float, default=60000,
                    help="反推预算用的实测吞吐 token/s（eager~73k，compile~120k）")
    ap.add_argument("--eval_every", type=int, default=60)  # 每多少步验证一次
    ap.add_argument("--eval_iters", type=int, default=25)  # 每次验证跑多少个 batch
    ap.add_argument("--compile", action="store_true",
                    help="用 torch.compile 融合算子（需要 triton，小模型通常快 40~60%%）")
    ap.add_argument("--workers", type=int, default=4, help="数据加载进程数")
    ap.add_argument("--val_ratio", type=float, default=0.10)  # 10% 文档留作验证集
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--min_freq", type=int, default=1,
                    help="字符最低频次，低于它映射成 <unk>（大语料用来控制词表规模）")
    ap.add_argument("--max_vocab", type=int, default=0, help="词表上限（0=不限）")
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))  # 输出目录
    args = ap.parse_args()
    #endregion

    #region 3.2 环境准备：随机种子 + 设备 + TF32   【集合:device】
    torch.manual_seed(1337)  # 固定种子，同样的命令能复现同样的曲线
    random.seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TF32：让 fp32 矩阵乘用 19 位尾数（快 2~3 倍），精度损失对训练几乎无感
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)  # 输出目录不存在就建
    #endregion

    # ---------- 数据 ----------
    #region 3.3 数据：切分 → 建词表 → 编码 → DataLoader
    docs = load_docs(args.corpus)  # 列表，每项一篇文档
    random.shuffle(docs)                                   # 打乱文档，再做 train/val 划分
    n_val = max(1, int(len(docs) * args.val_ratio))  # 至少留 1 篇做验证
    val_docs, train_docs = docs[:n_val], docs[n_val:]  # 按"文档"切，不是按 token 切
    # 为什么按文档切而不是按 token 切：按 token 切会让同一篇文章的前半段在训练、
    # 后半段在验证，模型等于看过答案，val loss 会虚低（这就是数据泄漏）
    tok = CharTokenizer.from_text(docs, min_freq=args.min_freq, max_chars=args.max_vocab or None)
    train_ids = build_ids(train_docs, tok)  # (N_train,)
    val_ids = build_ids(val_docs, tok)  # (N_val,)
    print(f"语料: {len(docs)} 篇文档 | 训练 {len(train_docs)} 篇 / 验证 {len(val_docs)} 篇")
    print(f"token: 训练 {len(train_ids):,} · 验证 {len(val_ids):,} | 词表 V={tok.vocab_size}")

    train_ds = LMDataset(train_ids, args.seq_len)  # 训练：步长 = T，样本不重叠
    # 验证集用 1/4 步长的滑窗，样本更多、PPL 估计更稳
    val_ds = LMDataset(val_ids, args.seq_len, stride=max(1, args.seq_len // 4))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=args.workers,
                              persistent_workers=args.workers > 0,
                              prefetch_factor=4 if args.workers > 0 else None)
    # 上面几个参数解释：
    #   shuffle=True       每个 epoch 重排样本顺序
    #   drop_last=True     最后不足一个 batch 的丢掉，保证每步 tensor 形状固定
    #   pin_memory=True    锁页内存，H2D 拷贝更快
    #   persistent_workers 别每个 epoch 重建进程（重建一次要重新 fork，很贵）
    #   prefetch_factor=4  每个 worker 提前准备 4 个 batch
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=True)
    #endregion

    # ---------- 模型 ----------
    #region 3.4 建模型（可选 compile）   【集合:model, raw_model, n_params】
    cfg = scale_config(args.preset, tok.vocab_size, block_size=args.seq_len)  # 按预设拿结构超参
    cfg.dropout = args.dropout  # dropout 用命令行覆盖，不跟预设走
    if args.arch == "lstm":
        model = LSTM_LM(tok.vocab_size, cfg.d_model, args.lstm_hidden,
                        args.lstm_layers, args.dropout).to(device)  # 对照组
    else:
        model = GPT(cfg).to(device)
    raw_model = model                      # 保存/验证都用未包装的原模型（state_dict 不带 _orig_mod 前缀）
    # 为什么要留 raw_model：torch.compile 返回的是 OptimizedModule 包装，
    # 它的 state_dict key 会多出 "_orig_mod." 前缀，直接存下来以后加载会对不上
    n_params = raw_model.num_params()
    if args.compile:
        import triton  # noqa: F401  ← 没有 triton 编译不出融合核
        print(f"torch.compile: 首次编译需要约 1~2 分钟，之后每步走融合核")
        model = torch.compile(model)  # 之后的 forward 走融合后的 kernel，实测 +57%
    tokens_per_step = args.batch * args.seq_len * args.grad_accum  # 一次参数更新吃掉多少 token
    steps = max(1, int(args.tokens / tokens_per_step))  # token 预算 → 步数
    if args.minutes:
        # 时间预算模式：按 --tps 的实测吞吐反推能跑多少步（比如 eagers 73k / compile 120k）
        steps = max(20, int(args.minutes * 60 * args.tps / tokens_per_step))
        args.tokens = steps * tokens_per_step  # 反写回去，后面打印用真实的 token 数
    print(f"\n模型: arch={args.arch} preset={args.preset} | 参数量 {n_params/1e6:.2f}M")
    if args.arch == "gpt":
        print(f"      层数={cfg.n_layer} d_model={cfg.d_model} 头数={cfg.n_head} "
              f"上下文={args.seq_len}")
    else:
        print(f"      LSTM hidden={args.lstm_hidden} layers={args.lstm_layers} "
              f"上下文={args.seq_len}")
    # tokens/参数 才是判断"训练够不够"的指标，Chinchilla 建议 20
    print(f"计划: {steps} 步 × {tokens_per_step:,} token/步 = {args.tokens/1e6:.1f}M token"
          f" ≈ {args.tokens/len(train_ids):.1f} 个 epoch(训练集)\n")
    #endregion

    #region 3.5 优化器   【集合:opt】
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    # betas=(0.9,0.95)：beta2 比默认的 0.999 小，因为 LLM 训练步数少，动量需要跟得上
    # fused=True：用 CUDA 融合版 AdamW，省显存也快
    #endregion

    #region 3.6 训练循环   【集合:metrics_path, it, log_every】
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    open(metrics_path, "w").close()  # 清空旧记录，每次训练从零开始写
    best_val, best_step = float("inf"), 0  # 记录历史最好 val loss 及对应步数
    t0 = time.time()  # 墙钟起点，后面算 ETA 用
    model.train()
    it = iter(train_loader)  # 手动迭代：这样一个 epoch 结束可以无缝续上，不用管 epoch 边界
    log_every = max(1, min(20, steps // 8))  # 一共大约打印 8 次，最少 20 步一次

    for step in range(steps):
        # ① 更新学习率（每步重算，warmup 期是变的）
        lr = lr_at(step, steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        # ② 前向 + 反向，累积 grad_accum 次再更新
        opt.zero_grad(set_to_none=True)  # set_to_none 比填 0 更省，直接把梯度释放掉
        acc_loss, acc_tok = 0.0, 0  # acc_loss 是这 accum 次的平均 loss，acc_tok 是本步 token 数
        for _ in range(args.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:                       # 一个 epoch 结束，重新打乱
                it = iter(train_loader)
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # (B,T)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss, _ = model(x, y)  # bf16 前向：显存减半、速度翻倍，loss 仍在 fp32 里算
            # 除以 grad_accum：让累积后的梯度等于"大 batch"的平均梯度，而不是累加
            (loss / args.grad_accum).backward()
            acc_loss += loss.item()  # item() 会同步 GPU，但只有一次，代价可接受
            acc_tok += y.numel()
        # 梯度裁剪：把所有参数梯度的总范数压到 1.0 以内，防止偶发的大梯度毁掉权重
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 返回裁剪前的总范数，用来监控
        opt.step()  # 参数更新

        # ③ 打印进度
        tr_loss = acc_loss / args.grad_accum  # 这一步的平均训练 loss
        if step % log_every == 0 or step == steps - 1:
            el = time.time() - t0  # 已用秒数
            tps = (step + 1) * tokens_per_step / max(el, 1e-6)  # 实测吞吐 token/s
            eta = (steps - step - 1) * tokens_per_step / max(tps, 1)  # 剩余秒数 = 剩余 token / 吞吐
            print(f"step {step:>5}/{steps} | loss {tr_loss:.4f} | ppl {math.exp(tr_loss):7.2f} "
                  f"| lr {lr:.2e} | gnorm {gn.item():.2f} | {tps/1000:.1f}k tok/s "
                  f"| 已用 {el/60:.1f}min | 剩余 {eta/60:.1f}min")

        # ---------- 定期验证 ----------
        if (step + 1) % args.eval_every == 0 or step == steps - 1:
            va = evaluate(raw_model, val_loader, device, args.eval_iters)  # 用 raw_model，绕过 compile 包装
            marker = ""
            if va < best_val:
                best_val, best_step = va, step + 1
                # 只在变好时保存，所以 best.pt 永远是整个训练过程中表现最好的那份
                save_ckpt(os.path.join(args.out, "best.pt"), raw_model, tok,
                          extra={"step": step + 1, "val_loss": va, "preset": args.preset})
                marker = "  ← best，已保存"
            # 一行一条 json，方便后面画曲线、也方便直接 grep
            rec = {"step": step + 1, "tokens": (step + 1) * tokens_per_step,
                   "train_loss": tr_loss, "val_loss": va,
                   "train_ppl": math.exp(tr_loss), "val_ppl": math.exp(va), "lr": lr,
                   "params": n_params,
                   "elapsed_min": (time.time() - t0) / 60}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"        [eval] val_loss {va:.4f} | val_ppl {math.exp(va):7.2f}{marker}")
            model.train()  # evaluate 里切了 eval()，这里必须切回来，否则 dropout 被关掉
    #endregion

    #region 3.7 收尾：存 last + 打印总结
    save_ckpt(os.path.join(args.out, "last.pt"), raw_model, tok,
              extra={"step": steps, "val_loss": best_val, "preset": args.preset})
    total_min = (time.time() - t0) / 60
    print(f"\n训练完成：{total_min:.1f} 分钟 | {steps} 步 | {args.tokens/1e6:.1f}M token")
    print(f"最佳 val_loss {best_val:.4f} (ppl {math.exp(best_val):.2f}) @ step {best_step}")
    print(f"ckpt: {os.path.join(args.out, 'best.pt')}")
    #endregion

    # ---------- 训练曲线 ----------
    #region 3.8 画 loss 曲线（失败不影响训练结果）
    try:
        import matplotlib
        matplotlib.use("Agg")  # 非交互后端，无显示器也能存图
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # 让中文标签不乱码
        matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
        import matplotlib.pyplot as plt
        rows = [json.loads(l) for l in open(metrics_path, encoding="utf-8")]  # 读回刚才写的指标
        st = [r["step"] for r in rows]  # x 轴：步数
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=130)  # 左 loss、右 ppl
        ax[0].plot(st, [r["train_loss"] for r in rows], label="train")
        ax[0].plot(st, [r["val_loss"] for r in rows], label="val")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy loss"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[0].set_title("Next-Token 预训练 loss")
        ax[1].plot(st, [r["train_ppl"] for r in rows], label="train")
        ax[1].plot(st, [r["val_ppl"] for r in rows], label="val")
        ax[1].set_xlabel("step"); ax[1].set_ylabel("perplexity"); ax[1].legend(); ax[1].grid(alpha=.3)
        ax[1].set_title("Perplexity = exp(loss)")
        fig.tight_layout()
        png = os.path.join(args.out, "loss_curve.png")
        fig.savefig(png)
        print(f"loss 曲线: {png}")  # 打印曲线保存路径
    except Exception as e:
        print("画图跳过:", e)  # 没装 matplotlib 也不该让训练白跑，兜住就行
    #endregion


#endregion


#region 4. 入口
if __name__ == "__main__":
    main()
#endregion

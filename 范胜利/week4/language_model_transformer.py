"""
字符级 Transformer 语言模型训练脚本（自回归 / GPT 结构）
—— language_model.py 的 Transformer 版：同一份语料、同一套行级切分、同一口径的 PPL

【和 language_model.py 的关系】
    数据管线、切分方式、日志格式、ckpt 结构全部保持一致，只把 RNN/LSTM 换成 Transformer，
    这样两次训练的 PPL 可以直接横向比较。差异只有三处：
      1) 模型：Embedding + 可学习位置编码 + N 层 Pre-LN TransformerEncoder（带因果掩码）+ 共享权重输出头
      2) 训练：一个 batch 内 B×T 个位置并行算 loss（RNN 必须沿时间步串行）
      3) 日志：多一列「Val PPL(末位)」—— 末位是唯一「抄不到答案」的位置，防止被泄漏骗到
      4) 新增 --generate：训练脚本本身也能续写（原脚本没有生成能力）

【为什么必须有因果掩码】
    语言模型在位置 t 的标签是 x_{t+1}。如果没有掩码，位置 t 的输出能直接看到 x_{t+1}，
    模型三步就能学会「抄下一个字」，loss 掉到接近 0，但一续写就原形毕露。
    --no-mask 就是用来演示这件事的（务必同时看末位 PPL 和续写效果）。

用法:
    python language_model_transformer.py --epochs 20
    python language_model_transformer.py --epochs 20 --save tf_best.pt
    python language_model_transformer.py --epochs 20 --no-mask        # 泄漏实验
    python language_model_transformer.py --epochs 20 --no-pos         # 去掉位置编码
    python language_model_transformer.py --generate "今天天气" --ckpt tf_best.pt
    python language_model_transformer.py --corpus "corpus_general.txt" --epochs 20
"""

import argparse
import glob
import math
import os
import random
import time

# 让 MPS（Apple GPU）遇到不支持的算子时自动回退到 CPU，而不是直接报错。
# 必须在 import torch 之前设置，torch 启动后这个环境变量就不生效了
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 本文件所在目录：默认语料、默认 ckpt 都相对它定位，避免「从别的目录执行就找不到文件」
HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────── 数据（与 language_model.py 完全一致） ───────────────────────────

def load_corpus(pattern="*.txt"):
    """
    读语料：pattern 可以是单个文件，也可以是通配符（glob 展开后按文件名排序拼接）。

    和原脚本相比有两处改动：
      · sorted()：glob 本身不保证顺序，排序后多次运行读到的顺序一致，实验才可复现
      · 相对路径兜底：在当前目录找不到时，再到脚本所在目录找一遍 ——
        这样无论你从哪个目录执行 python，"--corpus corpus_general.txt" 都能命中
    """
    paths = sorted(glob.glob(pattern))
    if not paths and not os.path.isabs(pattern):
        paths = sorted(glob.glob(os.path.join(HERE, pattern)))
    text = ""
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    return text


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1       —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    Transformer 建议用 stride=seq_len：它的样本吞吐和窗口数成正比，stride=1 时
    1700 万字的语料会切出 1700 万个窗口，一轮训练就跑到天荒地老。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class TransformerLM(nn.Module):
    """
    Embedding + 可学习位置编码 + N × TransformerEncoderLayer（Pre-LN）+ 输出头（与输入共享权重）

    结构对照 GPT：
        nn.Embedding(V, d)                    ← token embedding
        + nn.Embedding(max_len, d)            ← position embedding（可学习，不是 sin/cos）
        nn.TransformerEncoder(layer × N)      ← 加因果掩码后，每层就是 GPT 的一个 decoder block
        nn.Linear(d, V, bias=False)           ← 输出 logits，权重和输入 embedding 共享（weight tying）

    causal=False 时没有掩码（BERT 那种双向注意力）：训练 loss 会崩到接近 0，但那是「抄输入」。
    use_pos=False 时不做位置编码：自注意力对位置是置换等变的，模型会退化成「词袋 + 加权」。
    """

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=512,
                 dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,      # 带 mask 时关掉嵌套张量优化，行为更直观
        )

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight   # weight tying：省参数 + 给输出头加共享先验

        # embedding 初始化成小方差（GPT-2 的做法）：默认 std=1 时共享权重的 logits 会非常大，
        # 初始 loss 直接飙到 ln(V) 的两倍多
        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx)                                  # (B, T, d)
        if self.use_pos:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)

        mask = None
        if self.causal:
            # 下三角（含对角线）0 = 可以看，右上角 -1e9 = 不能看 → 位置 t 只能看 0..t
            mask = causal_mask(idx.size(1), idx.device)

        x = self.encoder(x, mask=mask)
        # 直接返回 logits：F.cross_entropy 内部自带 log_softmax，这里再 softmax 一次会毁掉数值
        return self.head(x)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, last_only=False,
              log_every=0):
    """
    跑一个 epoch，返回 (token 加权平均 loss, ppl)。

    last_only=True 时只统计每段「最后一个位置」：
    训练窗口是 [s, s+T)，最后一个位置的输入是 x[s+T-1]，标签是 x[s+T]，
    而这个标签从来没出现在输入窗口里 —— 所以这个位置永远抄不到答案，
    它才反映模型的真实语言能力（--no-mask 时会和全位置 PPL 差一个数量级）。

    log_every > 0 时每多少个 batch 打印一次进度（当前 loss / 已用时间 / 预计剩余），
    大语料一个 epoch 要跑几百个 batch，没有这个的话屏幕会长时间没反应。
    """
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    n_batches = len(loader)

    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if last_only:
            logits, y = logits[:, -1, :], y[:, -1]
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪 1.0：Transformer 早期偶尔出大梯度（共享权重让两侧梯度叠加，更容易冲）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        if train and log_every and (i % log_every == 0 or i == n_batches):
            avg = total_loss / total_tokens
            elapsed = time.time() - t0
            eta = elapsed / i * (n_batches - i)
            print(f"    batch {i:>4}/{n_batches}  loss={avg:.4f}  ppl={math.exp(avg):>8.1f}  "
                  f"已用 {elapsed:>5.0f}s  剩余≈{eta/60:.1f}min", flush=True)

    # 一个 batch 都没跑（验证集比 batch_size 还小时，drop_last 会把它全丢掉）：
    # 此时 total_tokens=0，如果直接除 1 会得到 loss=0、PPL=1.00 的假象，必须提前返回 inf
    if total_tokens == 0:
        print("  [警告] 这个 loader 没有产生任何 token（数据太少 + drop_last），PPL 记为 inf")
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 续写 ───────────────────────────

def causal_mask(T, device):
    """
    因果掩码（float）：mask[t][j] = 0 表示 t 可以看 j，-1e9 表示不能看。

    为什么不直接用 nn.Transformer.generate_square_subsequent_mask：
      · 它内部是 triu + masked_fill 用 -inf 填出来的；
      · 在 Apple MPS 后端（torch 2.2）上，triu/masked_fill 的实现会做乘法，
        而 0 × (-inf) = nan，于是 mask[0][0] 直接变成 nan，整个训练 loss 立刻变 nan；
      · 屏蔽位改用有限的大负数 -1e9，语义完全等价（exp(-1e9) = 0，注意力权重被清零），
        CPU / CUDA 上也没有任何副作用。
    """
    idx = torch.arange(T, device=device)
    future = idx[None, :] > idx[:, None]          # True = 未来位置，需要屏蔽
    return torch.zeros(T, T, device=device).masked_fill(future, -1e9)


def pick_device(choice="auto"):
    """
    选设备：cuda → mps（Apple Silicon GPU）→ cpu。

    --device 可以强制指定（cpu / cuda / mps），主要用途是课上对比「同一份代码跑 CPU 和 GPU」。
    注意：小模型 + 短序列时 MPS 未必比 CPU 快（kernel 启动开销占大头），跑之前先用
    --epochs 1 实测一下，别默认以为 GPU 一定赢。
    """
    if choice and choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_load(path, device):
    """兼容新旧 torch：2.6 起 torch.load 默认 weights_only=True，读带 str/dict 的 ckpt 会报错。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def generate(args):
    """
    续写：把 args.generate 这段开头喂进去，逐字往外蹦 n_chars 个新字。

    为什么推理必须逐 token：生成时第 t+1 个字还不存在，必须先采样出它才能去预测 t+2。
    这里每步都把整段上文重新喂一遍（没有 KV Cache），纯粹为了代码短好读。
    """
    device = pick_device(args.device)
    path = args.ckpt
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(HERE, path)      # 从别的目录执行时到脚本目录再找一次
    # 先读进 CPU 再 .to(device)：直接 map_location="mps" 在某些 torch 版本上不稳
    ck = torch_load(path, "cpu")
    a = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]

    model = TransformerLM(
        len(char2idx), a["d_model"], a["heads"], a["layers"], a["seq_len"], 0.0,
        use_pos=not a.get("no_pos", False),
        causal=not a.get("no_mask", False),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ids = [char2idx[ch] for ch in args.generate if ch in char2idx]
    if not ids:
        print(f"开头里的字都不在字表里：{args.generate}")
        return
    # 屏蔽字符（默认换行）：语料是按行拼的，模型很容易「回车」跳到另一条无关语料上，
    # 命令行输出会变成几行乱七八糟的文字。压成 -inf 后它的概率严格为 0
    ban_ids = [char2idx[ch] for ch in args.ban if ch in char2idx]
    out = list(ids)
    for _ in range(args.n_chars):
        # 只喂最近 seq_len 个 token：位置编码只有 seq_len 行，而且训练时也只见过这么长的上下文
        logits = model(torch.tensor([out[-args.seq_len:]], device=device))[0, -1]
        # 采样这一步搬到 CPU 做：向量只有 1×V，开销可忽略，
        # 但能绕开 MPS 上可能不支持的 multinomial / 原地索引赋值，CUDA 上也没坏处
        logits = logits.float().cpu()
        if ban_ids:
            logits[ban_ids] = float("-inf")
        if args.temperature <= 0.01:
            nxt = int(logits.argmax())          # 温度≈0 → 贪心解码，结果完全确定
        else:
            nxt = int(torch.multinomial(
                F.softmax(logits / args.temperature, dim=-1), 1))
        out.append(nxt)

    print(f"输入: {args.generate}")
    print(f"续写: {''.join(idx2char[i] for i in out[len(ids):])}")


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # —— 与 language_model.py 同名的参数，保持默认值和语义一致 ——
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    # 注意：原脚本默认 stride=1（密集重叠）。Transformer 用它训练量会放大 64 倍，
    # 默认改成 seq_len=不重叠；想和原脚本完全对齐可显式传 --stride 1
    parser.add_argument("--stride",     type=int,   default=64,
                        help="切块间隔：1=密集重叠，seq_len=不重叠（Transformer 推荐）")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default=os.path.join(HERE, "*.txt"),
                        help="语料路径（支持通配符；相对路径会先试当前目录、再试脚本目录）")
    parser.add_argument("--save",       default=os.path.join(HERE, "tf_best.pt"))

    # —— Transformer 专用结构参数 ——
    # d=192 / 4 层 / 4 头 的唯一参数量约 2.56M，和原脚本 LSTM 版（embed 128 + hidden 256 + 2 层）的
    # 2.46M 接近，比 PPL 时才是「结构差异」而不是「模型大小差异」
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--heads",   type=int, default=4)
    parser.add_argument("--layers",  type=int, default=4)

    # —— 两个消融开关 ——
    parser.add_argument("--no-mask", dest="no_mask", action="store_true",
                        help="去掉因果掩码（泄漏实验）")
    parser.add_argument("--no-pos", dest="no_pos", action="store_true",
                        help="去掉位置编码")

    # —— 续写 ——
    parser.add_argument("--generate", default=None, help="给一段开头，续写文本（不训练）")
    parser.add_argument("--ckpt", default=None, help="续写用的模型，默认取 --save")
    parser.add_argument("--n_chars", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--ban", default="\n", help="续写时屏蔽这些字符（默认屏蔽换行）")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="CPU 线程数；小模型单线程通常比多线程快，也避开 OpenMP 冲突")
    parser.add_argument("--log_every", type=int, default=50,
                        help="训练时每多少个 batch 打印一次进度（0 = 只按 epoch 打印）")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="训练设备：auto = cuda → mps → cpu 依次尝试")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.generate:
        args.ckpt = args.ckpt or args.save
        generate(args)
        return

    device = pick_device(args.device)
    print(f"device: {device}  model: TRANSFORMER")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认 --corpus 路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 验证集不用 drop_last：验证集小时（比如 37 条 < batch 64）丢尾巴会丢掉整个 loader，
    # PPL 就变成「0 个 token → 1.00」的假象。run_epoch 内部按 token 加权求和，不满的 batch 不影响口径
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"样本数: 训练 {len(train_ds):,} / 验证 {len(val_ds):,}")

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_head=args.heads,
        n_layer=args.layers,
        max_len=args.seq_len,
        dropout=args.dropout,
        use_pos=not args.no_pos,
        causal=not args.no_mask,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}（d={args.d_model} 层={args.layers} 头={args.heads} "
          f"FFN={4*args.d_model} Pre-LN）  位置编码={'有' if not args.no_pos else '无'}  "
          f"因果掩码={'有' if not args.no_mask else '**无（泄漏实验）**'}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  "
          f"{'Val PPL':>10}  {'Val PPL(末位)':>14}")
    print("-" * 76)

    for epoch in range(1, args.epochs + 1):
        st = time.time()
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, log_every=args.log_every)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            _, va_ppl_last = run_epoch(model, val_loader, criterion, optimizer, device,
                                       train=False, last_only=True)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            # ckpt 存「重建模型 + 复现切分」需要的一切：
            #   model_state 权重 / char2idx、idx2char 字表（字表顺序随语料变，不能重算）
            #   args 结构超参 + 两个开关 + seed（演示层要靠种子重放同一份验证集）
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  "
              f"{va_ppl:>10.2f}  {va_ppl_last:>14.2f}"
              f"{marker}  ({time.time() - st:.0f}s)")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()

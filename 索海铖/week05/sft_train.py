"""
sft_train.py — 字符级 Transformer 语言模型的 SFT（监督微调）脚本

【这个脚本干什么】
    拿 transformer_lm.py 预训练好的 ckpt（如 tf_best.pt），在「指令-回答」格式的
    小语料（jsonl）上继续训练，让模型学会「看到指令 → 生成回答」的行为。

【SFT 数据格式（jsonl，每行一个对象）】
    {"instruction": "给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 238 数字2: 564",
     "output":     "较大的数字是 564。"}

    默认指向 week5 作业目录：
        E:/.../sft_train/sft_train.jsonl
        E:/.../sft_train/sft_val.jsonl

【实现要点（每一条都能在代码里找到对应位置）】
    1) 权重加载：用 ckpt 里存的 args 原样重建模型 + strict=True 加载，
       加载完先做一次「零训练评估」，确认权重真的加载成功；
    2) 字表冻结：直接用预训练的 char2idx，SFT 语料里的生僻字（表外字）丢弃并统计比例，
       绝不重建字表（否则预训练 embedding 全部作废）；
    3) 优化器：新开 Adam（不恢复预训练动量），学习率默认 1e-4（比预训练的 1e-3 小 10 倍），
       外加每 epoch 减半的 StepLR 衰减；
    4) 过拟合防护：小 epoch 数 + 验证集 response-PPL 选优存档 + patience 早停；
    5) 防灾难性遗忘（可选）：--replay 混入一份预训练语料的窗口做数据回放；
    6) loss 只算在「回答」上（SFT 的标准做法）：指令部分只作上下文，不作为监督信号，
       这样模型学的是「回答问题」而不是「复述问题」。

用法:
    python sft_train.py                                    # 全默认：加载 tf_best.pt 微调
    python sft_train.py --ckpt tf_best.pt --epochs 3 --lr 1e-4
    python sft_train.py --replay corpus_sample.txt         # 混入预训练语料防遗忘
    python sft_train.py --no-demo                          # 训练完不做续写演示
"""

import argparse
import json
import math           # PPL = exp(loss)
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 复用预训练脚本里的模型定义：SFT 不改结构，改的只是初始权重 / 数据 / 学习率。
# 注意 import 的是「同目录」的 transformer_lm，脚本放在一起才能跑。
from transformer_lm import TransformerLM

HERE = os.path.dirname(os.path.abspath(__file__))

# week5 作业目录下的默认语料（可用命令行参数覆盖）
DEFAULT_TRAIN = r"./sft_train/sft_train.jsonl"
DEFAULT_VAL = r"./sft_train/sft_val.jsonl"


# ─────────────────────── SFT 数据 ───────────────────────

def load_jsonl(path):
    """读 jsonl：每行一个 {"instruction": ..., "output": ...}。
    逐行解析而不是一次性 json.loads 整个文件，坏行只丢那一行，不让整个训练崩掉。"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] {os.path.basename(path)} 第 {ln} 行不是合法 JSON，已跳过")
                continue
            # instruction 允许为空（退化为纯续写样本），output 为空的行没有监督信号，直接丢
            if obj.get("output"):
                pairs.append((obj.get("instruction", "") or "", obj["output"]))
    if not pairs:
        raise FileNotFoundError(f"没读到任何有效样本: {path}")
    return pairs


def build_sample(instruction, output, char2idx, seq_len, sep="\n"):
    """
    把一条 (instruction, output) 变成 (x, y, mask) 三个 id 序列：

        拼接文本 = instruction + "\\n" + output + "\\n"
                              ↑ 分隔符          ↑ 结尾换行当「回答结束符」用

    两个刻意设计（字符级模型没有 tokenizer / 特殊符号，只能靠字符本身）：
        · instruction 和 output 之间插一个换行 —— 相当于极简版「对话模板」，
          让模型能区分「问题讲到哪、回答从哪开始」；
        · output 结尾补一个换行 —— 充当穷人版 EOS：模型学到「回答完该换行了」，
          演示续写时就有了天然的停机信号（生成到换行即止）。

    关于 loss 掩码（本脚本和纯预训练最大的差别）：
        SFT 的监督信号只应该来自 output 部分 —— 指令是「条件」，不是「答案」。
        若不掩掉，模型会花大量容量去学「复述指令」，在指令很长的样本上尤其浪费。
        x/y 与预训练一样错一位（y[i] = x[i+1] 的目标），mask[i]=True 表示
        y[i] 这个目标字属于 output（含结尾换行），只有这些位置计入 loss。

    截断策略（总长超过 seq_len+1 时）：
        · output 保头部（答案的前几个字最重要，截尾影响小）；
        · instruction 保尾部（离问题最近的上下文信息量最大）。
        顺序：先砍 instruction，instruction 砍光了才砍 output。
    """
    # ── 1. 按预训练字表逐字过滤（字表冻结：表外字直接丢，见要点 2）──
    # 注意 instruction / output 分开过滤，过滤完再算长度，掩码边界才不会错位
    p = [char2idx[c] for c in instruction if c in char2idx]
    o = [char2idx[c] for c in output + sep if c in char2idx]   # 结尾换行并入 output

    if not o:                    # 回答全部是表外字 → 这条样本没有监督信号
        return None

    # ── 2. 截断到模型见过的长度（位置编码只有 seq_len 行）──
    max_total = seq_len + 1      # x、y 各最多 seq_len 个
    if len(o) > max_total - 1:   # output 本身超长：保头
        o = o[: max_total - 1]
    budget = max_total - len(o)  # 剩余预算给 instruction
    if len(p) > budget:          # instruction 超预算：保尾（最贴近问题的部分）
        p = p[-budget:]

    ids = p + o
    if len(ids) < 2:             # 只剩 1 个字，凑不出 (x, y) 对
        return None

    x, y = ids[:-1], ids[1:]
    # y[i] 对应 ids[i+1]；ids 里第 len(p) 个起是 output，
    # 所以 y[i] 是 output ⟺ i+1 >= len(p) ⟺ i >= len(p)-1
    mask = [i >= len(p) - 1 for i in range(len(y))]
    return x, y, mask


class SFTDataset(Dataset):
    """每条 jsonl 样本 → 一个 (x, y, mask)；变长序列靠 collate_fn 补齐到批内最长。"""

    def __init__(self, pairs, char2idx, seq_len):
        self.items, self.skipped = [], 0
        oov_lines = 0
        for ins, out in pairs:
            s = build_sample(ins, out, char2idx, seq_len)
            if s is None:
                self.skipped += 1     # 回答全是表外字（或太短），只能整条丢弃
                oov_lines += 1
                continue
            self.items.append(s)
        if not self.items:
            raise RuntimeError("过滤字表后一条有效样本都不剩 —— 检查语料或预训练字表")
        n_out = sum(sum(m) for _, _, m in self.items)   # 监督 token 总数（mask 之和）
        print(f"    有效样本 {len(self.items)} 条（丢弃 {self.skipped} 条表外/过短），"
              f"监督 token 共 {n_out:,} 个")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y, m = self.items[i]
        # mask 转成 float（后面直接和 loss 逐元素相乘），collate 时 pad 位补 0
        return (torch.tensor(x), torch.tensor(y),
                torch.tensor(m, dtype=torch.float))


def make_replay_items(text, char2idx, seq_len, n):
    """
    数据回放（要点 5，防灾难性遗忘）：从预训练语料里切 n 个窗口，
    形状和 SFT 样本完全一致，只是 mask 全 1（整段都算 loss —— 通用语料没有
    「指令/回答」之分，保持普通语言模型的训练口径即可）。
    """
    ids = torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)
    items = []
    if len(ids) < seq_len + 1:
        return items
    n_win = (len(ids) - seq_len - 1) // seq_len + 1
    # 固定 seed 采样窗口：回放数据也要可复现，不能每次跑都不一样
    rng = random.Random(1234)
    for start in rng.sample([i * seq_len for i in range(n_win)], min(n, n_win)):
        chunk = ids[start: start + seq_len + 1]
        x, y = chunk[:-1], chunk[1:]
        items.append((x, y, torch.ones(len(y))))     # mask 全 1：整窗口计 loss
    return items


def collate(batch):
    """把变长的 (x, y, mask) 补齐成三个 (B, T_max) 张量。
    pad 用 id=0、mask=0：
        · pad 只会加在序列末尾，因果掩码保证前面的真实位置看不到 pad，注意力不受污染；
        · pad 位置的 loss 权重是 0，不会进梯度。"""
    xs, ys, ms = zip(*batch)
    T = max(len(x) for x in xs)
    X = torch.zeros(len(xs), T, dtype=torch.long)
    Y = torch.zeros(len(xs), T, dtype=torch.long)
    M = torch.zeros(len(xs), T)
    for i, (x, y, m) in enumerate(zip(xs, ys, ms)):
        X[i, : len(x)], Y[i, : len(y)], M[i, : len(m)] = x, y, m
    return X, Y, M


# ─────────────────────── 训练 / 评估 ───────────────────────

def masked_loss(model, x, y, m):
    """只对 mask=1 的位置算交叉熵（SFT 核心）：先逐位置算，再按 token 加权平均。"""
    logits = model(x)                                     # (B, T, V)
    # reduction="none" 保留每个位置的 loss，便于用 mask 挑出「回答」位置
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           y.reshape(-1), reduction="none")   # (B*T,)
    loss = loss * m.reshape(-1)                           # pad 和指令位置清零
    return loss.sum(), m.sum()                            # 返回总和与有效 token 数


@torch.no_grad()
def evaluate(model, loader, device):
    """验证：返回 (loss, ppl)。只统计 mask=1 的位置（即「回答」token）的 PPL。
    @torch.no_grad() + model.eval() 缺一不可：前者不建计算图省显存，
    后者关掉 dropout —— 否则同一个模型每次评出来的 PPL 都在漂。"""
    model.eval()
    total, tokens = 0.0, 0.0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        s, n = masked_loss(model, x, y, m)
        total += float(s)
        tokens += float(n)
    loss = total / max(tokens, 1.0)
    return loss, math.exp(loss)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── 要点 1：用 ckpt 里的 args 原样重建模型，strict 加载 ──
    ck = torch.load(args.ckpt, map_location=device)
    pre = ck["args"]                        # 预训练时的全部超参（含 d_model/heads/layers/seq_len）
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]   # 要点 2：字表原封不动
    # 防呆：无因果掩码的泄漏实验 ckpt（如 tf_best.pt / tf_nomask.pt）只会「抄输入」，
    # 没有真正的生成能力，拿它 SFT 只会训出复读机（本仓库里 tf_best.pt 恰恰是泄漏版，
    # 正常基线是 tf_full.pt / tf_mask.pt）。这里直接拦住，比训完看生成结果再排查快得多
    if pre.get("no_mask", False):
        raise SystemExit(
            f"错误：{args.ckpt} 是 --no-mask 泄漏实验的产物（args.no_mask=True），"
            f"\n它只会抄输入、不会生成，不能作为 SFT 的起点。"
            f"\n请改用带因果掩码的正常 ckpt，例如：--ckpt {os.path.join(HERE, 'tf_full.pt')}")
    # 不能随便换超参 new 一个模型 —— 结构对不上 load_state_dict 直接报形状错误；
    # 两个消融开关（no_mask / no_pos）也必须按预训练时的状态还原
    model = TransformerLM(len(char2idx), pre["d_model"], pre["heads"], pre["layers"],
                          pre["seq_len"], args.dropout,
                          use_pos=not pre.get("no_pos", False),
                          causal=not pre.get("no_mask", False)).to(device)
    # strict=True（默认）：一个 key 都不许少/多，防「悄悄漏加载半套权重」
    model.load_state_dict(ck["model_state"], strict=True)
    print(f"device: {device}  已加载预训练权重 {args.ckpt}"
          f"（{sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数，"
          f"d={pre['d_model']} x {pre['layers']} 层，seq_len={pre['seq_len']}）")

    # ── 数据 ──
    print(f"训练集: {args.train}\n验证集: {args.val}")
    seq_len = pre["seq_len"]                # 序列长度由预训练模型决定（位置编码只有这么多行）
    print("[训练集]"); tr = SFTDataset(load_jsonl(args.train), char2idx, seq_len)
    print("[验证集]"); va = SFTDataset(load_jsonl(args.val), char2idx, seq_len)

    items = list(tr.items)
    if args.replay:
        # 要点 5：混入预训练语料做数据回放（默认按 1:1 的量）
        with open(args.replay, encoding="utf-8", errors="ignore") as f:
            pre_text = f.read()
        n_replay = int(len(items) * args.replay_ratio)
        replay = make_replay_items(pre_text, char2idx, seq_len, n_replay)
        items += replay
        print(f"[回放]   混入预训练语料窗口 {len(replay)} 条（--replay {args.replay}）")
    random.shuffle(items)                   # 回放样本和 SFT 样本打混，模型不会「分段学」
    tr_ds = _ListDataset(items)

    # 小语料不需要 drop_last（batch 本来就少，再丢就没了）；验证集不打乱，可复现
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ── 要点 1 的收尾：零训练评估，确认权重加载成功 ──
    # 这个数没有「应该等于多少」的基准（换了语料），它的作用是：
    # 若 load 失败 / 结构错位，这里通常会烂得离谱（PPL 上千），一眼就能看出来
    l0, p0 = evaluate(model, va_loader, device)
    print(f"\n[SFT 前] 验证集 回答PPL = {p0:.2f}（loss {l0:.4f}）—— 微调前的起点\n")

    # ── 要点 3：新开 Adam + 小学习率（预训练 1e-3 → 这里默认 1e-4）──
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 小语料几个 epoch 就过拟合，学习率先小再一路减半收敛，比固定 lr 稳得多
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    best, bad = float("inf"), 0
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValPPL(回答)':>13} {'lr':>9} {'tok/s':>8}")
    print("-" * 55)
    for ep in range(1, args.epochs + 1):
        # train()/eval() 成对出现：训练开 dropout（正则），评估关（确定性）
        model.train()
        st, seen, tot = time.time(), 0.0, 0.0
        for x, y, m in tr_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            s, n = masked_loss(model, x, y, m)
            loss = s / n                     # batch 内按有效 token 平均
            opt.zero_grad()
            loss.backward()
            # 梯度裁剪和预训练保持一致：weight-tied 输出头梯度叠加，偶发大梯度裁一下
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(s.detach()); seen += float(n)   # detach：只取数值，不碰计算图
        sched.step()
        _, v_ppl = evaluate(model, va_loader, device)
        mark = ""
        if v_ppl < best:
            # 要点 4：按验证集选优存档（不是存最后一个 epoch —— 那个往往已经过拟合）
            best, bad, mark = v_ppl, 0, " *"
            # ckpt 记 stage="sft" 和本阶段超参，日后能追溯这个权重是哪来的；
            # 结构超参原样抄预训练的，保证 transformer_lm.py 的 generate 还能直接加载它
            merged = dict(pre)
            merged.update({"stage": "sft", "sft_train": args.train, "sft_val": args.val,
                           "sft_lr": args.lr, "sft_epochs": args.epochs, "seed": args.seed})
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": merged}, args.save)
        else:
            bad += 1
        print(f"{ep:>6} {tot/max(seen,1):>10.4f} {v_ppl:>13.2f} "
              f"{sched.get_last_lr()[0]:>9.1e} {seen/max(time.time()-st,1e-6):>8,.0f}{mark}")
        if bad >= args.patience:            # 要点 4：早停 —— val PPL 连续 patience 轮不降
            print(f"验证 PPL 连续 {args.patience} 轮无提升，早停")
            break

    print(f"\n最佳验证 回答PPL: {best:.2f}（SFT 前 {p0:.2f}）  已保存 {args.save}")


class _ListDataset(Dataset):
    """把 [(x,y,mask), ...] 列表包成 Dataset（torch 要求喂 DataLoader 的得是 Dataset）。
    列表元素可能是 list（SFT 样本）也可能是 tensor（回放样本），统一成 tensor。"""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y, m = self.items[i]
        return (x if torch.is_tensor(x) else torch.tensor(x),
                y if torch.is_tensor(y) else torch.tensor(y),
                m if torch.is_tensor(m) else torch.tensor(m, dtype=torch.float))


# ─────────────────────── 训练后演示 ───────────────────────

@torch.no_grad()
def demo(args):
    """从验证集挑两条指令做续写演示：训练完直观看看模型「会不会答题」。
    生成逻辑和 transformer_lm.generate 一致：逐 token 串行（无 KV Cache），
    喂最近 seq_len 个字，到换行（我们设计的「回答结束符」）就停。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.save, map_location=device)
    c, char2idx, idx2char = ck["args"], ck["char2idx"], ck["idx2char"]
    model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"],
                          c["seq_len"], 0.0,
                          use_pos=not c.get("no_pos", False),
                          causal=not c.get("no_mask", False)).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()                            # 关 dropout：同一个开头每次生成结果才稳定

    pairs = load_jsonl(args.val)[: args.demo_n]
    for ins, ref in pairs:
        # 演示时也按训练的拼接格式：指令 + 换行，让模型从「回答的第一个字」开始续写
        prompt = "".join(ch for ch in ins if ch in char2idx) + "\n"
        ids = [char2idx[ch] for ch in prompt]
        out = list(ids)
        for _ in range(args.n_chars):
            logits = model(torch.tensor([out[-c["seq_len"]:]], device=device))[0, -1]
            if args.temperature <= 0.01:
                nxt = int(logits.argmax())                # 温度≈0：贪心，结果确定
            else:
                nxt = int(torch.multinomial(
                    F.softmax(logits / args.temperature, dim=-1), 1))
            out.append(nxt)
            if idx2char[nxt] == "\n":                     # 学到的「回答结束符」：换行即停
                break
        gen = "".join(idx2char[i] for i in out[len(ids):])
        print(f"\n指令: {ins[:60]}{'…' if len(ins) > 60 else ''}"
              f"\n生成: {gen.rstrip()}"
              f"\n参考: {ref.rstrip()}")


def main():
    ap = argparse.ArgumentParser(description="字符级 Transformer 的 SFT 微调")
    ap.add_argument("--train", default=DEFAULT_TRAIN, help="训练集 jsonl 路径")
    ap.add_argument("--val", default=DEFAULT_VAL, help="验证集 jsonl 路径")
    # 本仓库各 ckpt 的身份（默认用 tf_full.pt —— 有因果掩码的正常基线）：
    #   tf_full.pt           正常基线（mask + pos，全量语料）★ SFT 默认起点
    #   tf_mask.pt           正常基线（另一份语料路径训的）
    #   tf_best.pt           ⚠ 泄漏实验产物（no_mask=True），不能用于 SFT
    #   tf_nomask.pt / tf_no-mask-600w.pt   泄漏实验（no_mask=True）
    #   tf_no-pos-600w.pt    去位置编码消融（no_pos=True）
    ap.add_argument("--ckpt", default=os.path.join(HERE, "tf_full.pt"),
                    help="预训练 checkpoint（从这里加载初始权重，必须是带因果掩码的版本）")
    ap.add_argument("--save", default=os.path.join(HERE, "tf_sft.pt"),
                    help="SFT 最优权重保存路径（默认另存新文件，绝不覆盖预训练 ckpt）")
    # 要点 3：学习率默认 1e-4，比预训练的 1e-3 小一个量级
    ap.add_argument("--lr", type=float, default=1e-4)
    # 要点 4：小语料小 epoch；早停 patience
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="SFT 阶段保持 0.1 别关 —— 小语料上最便宜的正则")
    # 要点 5：数据回放（不传就不混）
    ap.add_argument("--replay", default=None,
                    help="可选：预训练语料路径，混入做数据回放防遗忘")
    ap.add_argument("--replay_ratio", type=float, default=1.0,
                    help="回放窗口数 = 训练样本数 × 该比例")
    ap.add_argument("--seed", type=int, default=42)
    # 演示参数
    ap.add_argument("--no-demo", dest="demo", action="store_false",
                    help="训练结束后不做续写演示")
    ap.add_argument("--demo_n", type=int, default=2, help="演示取几条验证样本")
    ap.add_argument("--n_chars", type=int, default=80, help="演示最多续写多少字")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    train(args)
    if args.demo:
        print("\n" + "═" * 60 + "\n续写演示（加载刚保存的最优 SFT 权重）\n" + "═" * 60)
        demo(args)


if __name__ == "__main__":
    main()

"""
demo_transformer_continuation.py —— 课堂演示：同一段开头，让 LSTM / RNN / Transformer 并排续写

【这个演示想说明什么】
    三个模型训练在**同一份语料、同一套行级切分、同一个验证集**上，唯一差别是结构。
    并排看它们的续写，能直观回答「Transformer 到底强在哪 / 小语料上强不强」。

    演示同时给出两组数字：
      1) 验证集 PPL（全位置 + 末位）—— 和训练脚本同一口径，用 ckpt 里的 seed 重放同一份切分
      2) 相邻重复率 / 最长连续重复 —— 把「气气气气」这种退化量化，不用人眼数

    顺带还能演示两件反直觉的事（如果目录里有 --no-mask 训出来的 ckpt）：
      · 无 mask 的模型全位置 PPL 只有 1.x，看起来强得离谱
      · 但它一续写就在复读，末位 PPL 也差得多 —— 因为它训练时「抄得到答案」

【依赖】
    language_model_transformer.py  提供 TransformerLM / CharDataset / run_epoch / load_corpus
    language_model.py              提供 LM（RNN / LSTM 那套），用来加载旧 ckpt

用法:
    # 1) 三个模型各训一个（已有 best_model.pt / best_model_rnn.pt 的可以直接用）
    python language_model.py --model lstm --epochs 20 --save best_model.pt
    python language_model.py --model rnn  --epochs 20 --save best_model_rnn.pt
    python language_model_transformer.py --epochs 20 --save tf_best.pt

    # 2) 一组 prompt 并排看
    python demo_transformer_continuation.py

    # 3) 课上一句一句输入（回车结束）
    python demo_transformer_continuation.py --interactive

    # 4) 固定输出、方便反复讲同一个例子
    python demo_transformer_continuation.py --prompt "今天天气" --temperature 0 --n_chars 40

    # 5) 只比 Transformer 的两个版本
    python demo_transformer_continuation.py --ckpts tf_best.pt tf_nomask.pt
"""

import argparse
import glob
import os
import random
import unicodedata

# 让 MPS 遇到不支持的算子时回退到 CPU（必须在 import torch 之前）
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 演示层不另起一套模型/数据口径，直接复用训练脚本里的实现
from language_model import LM
from language_model_transformer import (
    CharDataset, TransformerLM, load_corpus, pick_device, run_epoch, torch_load,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# 默认要比的三个模型：LSTM（原脚本默认）、RNN、Transformer
DEFAULT_CKPTS = ["best_model.pt", "best_model_rnn.pt", "tf_best.pt"]


# ─────────────────────────── 加载 ───────────────────────────

def load_any(path, device):
    """
    按 ckpt 里记录的 args 还原模型（RNN / LSTM / Transformer 都能认）。

    关键是「结构超参从 ckpt 里读、不写死」：
    写死 d_model / hidden_dim 的话，换一个 ckpt 就会 load_state_dict 形状不匹配。
    """
    # 先读进 CPU 再 .to(device)：直接 map_location="mps" 在某些 torch 版本上不稳
    ck = torch_load(path, "cpu")
    a = ck["args"]
    V = len(ck["char2idx"])

    if a.get("model") in ("rnn", "lstm"):
        model = LM(
            vocab_size=V, embed_dim=a["embed_dim"], hidden_dim=a["hidden_dim"],
            num_layers=a["num_layers"], model_type=a["model"], dropout=a["dropout"],
        )
        kind = a["model"].upper()
    else:
        model = TransformerLM(
            V, a["d_model"], a["heads"], a["layers"], a["seq_len"], 0.0,
            use_pos=not a.get("no_pos", False),
            causal=not a.get("no_mask", False),
        )
        kind = "TRANSFORMER"
        if a.get("no_pos"):
            kind += "（无位置编码）"
        if a.get("no_mask"):
            kind += "（无 mask）"

    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()      # 关 dropout，否则同一个开头每次续写都不一样
    return {"path": path, "kind": kind, "model": model, "ck": ck, "args": a}


def _disp_width(s):
    """中文/全角字符占 2 列：用显示宽度算对齐，中英混排的表格才不会歪"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _ljust(s, width):
    return s + " " * max(0, width - _disp_width(s))


def tag(entry):
    """把「结构 / 参数量 / 文件」打到输出上，避免课上把几个模型念串"""
    n = sum(p.numel() for p in entry["model"].parameters())
    return _ljust(entry["kind"], 24) + f"｜{n/1e6:.2f}M 参数｜{os.path.basename(entry['path'])}"


# ─────────────────────────── 续写 ───────────────────────────

@torch.no_grad()
def generate(entry, prompt, n_chars, temperature, top_k, seed, ban=""):
    """
    逐字续写 n_chars 个字。推理必须一个字一个字来（第 t+1 个字要先采样出来，
    才能当第 t+2 个字的前文），这正是「训练可并行、推理只能串行」的分水岭。
    """
    ck = entry["ck"]
    c2i, i2c = ck["char2idx"], ck["idx2char"]
    seq_len = entry["args"]["seq_len"]
    device = next(entry["model"].parameters()).device

    ids = [c2i[ch] for ch in prompt if ch in c2i]
    if not ids:
        return "（开头里的字都不在字表里）"
    # ban：把禁用字符的 logit 压成 -inf，让它的概率严格为 0 ——
    # 是「从来没被采样到」，而不是「生成了再过滤」（后者会让输出变短）
    ban_ids = [c2i[ch] for ch in ban if ch in c2i]

    # 固定采样种子：几个模型用同一个 seed 走采样，随机性来源一致，对比才公平
    torch.manual_seed(seed)
    out = list(ids)
    for _ in range(n_chars):
        # 只喂最近 seq_len 个 token：位置编码只有 seq_len 行，训练时也只见过这么长的上下文
        logits = entry["model"](torch.tensor([out[-seq_len:]], device=device))[0, -1]
        # 采样搬到 CPU：1×V 的向量，开销可忽略，还能绕开 MPS 上不支持的算子
        logits = logits.float().cpu()
        if ban_ids:
            logits[ban_ids] = float("-inf")        # 贪心和采样都屏蔽，行为一致
        if temperature <= 0.01:
            # 温度≈0 等价于贪心解码：结果完全确定，适合课上反复讲同一个例子
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature          # <1 更尖锐（保守）、>1 更平坦（发散）
            if top_k and top_k > 0:
                # top-k 截断：砍掉长尾，避免蹦出莫名其妙的一个字
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        out.append(nxt)

    return "".join(i2c[i] for i in out[len(ids):])


def repeat_stats(text):
    """
    返回（相邻重复率, 最长连续重复片段），用来量化「气气气气」这种退化。

    相邻重复率 = 相邻两个字相同的比例。正常中文大约 0.0x（叠词「人人/看看」有限），
    退化时能到 0.9 以上 —— 课上不用再盯着屏幕数「们」有多少个。
    """
    if not text:
        return 0.0, "—"
    rep = sum(1 for a, b in zip(text, text[1:]) if a == b) / max(1, len(text) - 1)
    best, cur, best_ch = 1, 1, text[0]
    for a, b in zip(text, text[1:]):
        if a == b:
            cur += 1
            if cur > best:
                best, best_ch = cur, b
        else:
            cur = 1
    # 阈值取 3：中文里两个字连用太常见（「人人」「常常」），连续 3 个以上才像卡住了
    return rep, (f"{best_ch}×{best}" if best >= 3 else "—")


# ─────────────────────────── PPL 复算 ───────────────────────────

def resolve_corpus(cand):
    """把 ckpt 里记录的语料路径解析成「本机真实存在」的一份；支持通配符。"""
    if not cand:
        return None
    p = cand if os.path.isabs(cand) else os.path.join(HERE, cand)
    return p if glob.glob(p) else None


@torch.no_grad()
def ppl_of(entry, corpus, device, batch_size=128):
    """
    按训练时的同一套划分复算 PPL：先 random.seed(ckpt 里的 seed) 再打乱同样的行，
    才能拿到「当初那个验证集」，否则 PPL 和训练日志没法比。
    """
    a = entry["args"]
    text = load_corpus(corpus)
    lines = text.splitlines()
    random.seed(a.get("seed", 42))
    random.shuffle(lines)
    split = int(len(lines) * (1 - a.get("val_ratio", 0.05)))
    ds = CharDataset("\n".join(lines[split:]), entry["ck"]["char2idx"],
                     a["seq_len"], a.get("stride", 1))
    # batch_size 不影响 PPL（内部按 token 求和再除）；shuffle=False 保证几个模型用的是同一份数据。
    # drop_last=False：验证集小时不能丢尾巴，否则可能一个 batch 都不剩，PPL 会被算成 1.00 的假象
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    if len(ds) == 0:
        return float("inf"), float("inf"), 0
    criterion = nn.CrossEntropyLoss()
    _, full = run_epoch(entry["model"], dl, criterion, None, device, train=False)
    _, last = run_epoch(entry["model"], dl, criterion, None, device,
                        train=False, last_only=True)
    return full, last, len(ds)


# ─────────────────────────── 输出 ───────────────────────────

def show(prompt, outs):
    """并排打印多个模型的续写，并在每行下面附上复读统计"""
    print("\n" + "─" * 78)
    print(f"输入：{prompt}")
    width = max(len(o["label"]) for o in outs)
    for o in outs:
        # 换行替换成空格：续写里带换行会把表格对齐搞乱
        body = o["text"].replace("\n", " ")
        rep, run = repeat_stats(body)
        print(f"  {o['label']:<{width}} │ {body}")
        print(f"  {'':<{width}} │ 相邻重复率 {rep:.2f}　最长连续重复 {run}")
    print("─" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help=f"要比的 ckpt 列表（默认：{' '.join(DEFAULT_CKPTS)}）")
    ap.add_argument("--prompt", nargs="*", default=["我们来到", "今天天气", "从前有座山"])
    ap.add_argument("--n_chars", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.65,
                    help="0 = 贪心（完全确定）")
    ap.add_argument("--top_k", type=int, default=40, help=">0 时只从 top-k 里采样")
    ap.add_argument("--ban", default="\n",
                    help="生成时屏蔽这些字符（默认屏蔽换行，避免跳去另一条语料行）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=None, help="算 PPL 用的语料，默认取 ckpt 里记录的")
    ap.add_argument("--no_ppl", action="store_true", help="跳过 PPL 计算（更快）")
    ap.add_argument("--interactive", action="store_true", help="进入交互模式，逐句输入")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="推理设备：auto = cuda → mps → cpu 依次尝试")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    # 找 ckpt：默认那几个，缺了就说明一句、跳过（至少得有一个才能演示）
    want = args.ckpts if args.ckpts else DEFAULT_CKPTS
    paths = []
    for p in want:
        full = p if os.path.isabs(p) else os.path.join(HERE, p)
        if os.path.exists(full):
            paths.append(full)
        else:
            print(f"  [跳过] 找不到 {p}")
    if not paths:
        raise FileNotFoundError(
            "一个模型都没找到。先训练：\n"
            "  python language_model.py --epochs 20 --save best_model.pt\n"
            "  python language_model.py --model rnn --epochs 20 --save best_model_rnn.pt\n"
            "  python language_model_transformer.py --epochs 20 --save tf_best.pt"
        )

    entries = [load_any(p, device) for p in paths]
    print("\n模型")
    for e in entries:
        print(f"  {tag(e)}")

    # PPL：语料依次尝试 --corpus → ckpt 里记录的 → 目录里的常见文件名
    if not args.no_ppl:
        corpus = resolve_corpus(args.corpus)
        if corpus is None:
            for e in entries:
                corpus = resolve_corpus(e["args"].get("corpus"))
                if corpus:
                    break
        if corpus is None:
            for name in ("corpus_general.txt", "corpus_sample.txt"):
                if os.path.exists(os.path.join(HERE, name)):
                    corpus = os.path.join(HERE, name)
                    break

        if corpus is None:
            print("\n[提示] 找不到语料，跳过 PPL 复算（只演示续写）。"
                  "用 --corpus 指定训练时那份语料即可。")
        else:
            print(f"\n验证集 PPL（语料 {os.path.basename(corpus)}，同一套 seed 切分）")
            print(f"  {_ljust('模型', 24)}{'全位置':>10}{'末位':>10}{'验证样本':>10}")
            for e in entries:
                full, last, n = ppl_of(e, corpus, device)
                print(f"  {_ljust(e['kind'], 24)}{full:>10.2f}{last:>10.2f}{n:>10,}")
            print("  说明：末位是唯一「抄不到答案」的位置；两者差很多时，说明模型在利用泄漏。")

    print(f"\n生成长度 {args.n_chars} 字｜temperature={args.temperature}｜"
          f"top_k={args.top_k or '不限'}｜屏蔽字符 {args.ban!r}｜seed={args.seed}")

    def run(prompt):
        outs = [{"label": e["kind"],
                 "text": generate(e, prompt, args.n_chars, args.temperature,
                                  args.top_k, args.seed, args.ban)}
                for e in entries]
        show(prompt, outs)

    if args.interactive:
        print("\n交互模式：输入开头（回车看各模型续写），直接回车退出。")
        while True:
            try:
                p = input("\n开头> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not p:
                break
            run(p)
    else:
        for p in args.prompt:
            run(p)


if __name__ == "__main__":
    main()

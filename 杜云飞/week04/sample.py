"""
sample.py —— 采样实验台：把课程 04 节的每一种策略都真跑一遍，肉眼对比差别。
================================================================================
对应《第五周》知识点：
    · 04 节  Greedy / Beam Search / Temperature / Top-K / Top-P
    · 04 节  logits → 概率分布 → 采样（--show-probs 直接打印这个分布）
    · 06 节  TTFT 首包延迟（--stream 流式输出并计时）
    · 02 节  自回归生成过程（一步步把新 token 拼回输入）
    · 加餐    KV Cache：为什么真实推理离不开它

用法：
    python sample.py                                  # 默认对比全部策略
    python sample.py --prompt "他推开门，屋里没有人"   # 换开头
    python sample.py --mode temperature --stream       # 只跑温度、流式输出
    python sample.py --mode beam --beam 4
    python sample.py --show-probs                      # 看 next-token 概率分布长什么样
    python sample.py --bench                           # KV Cache 加速比
    python sample.py --interactive                     # 交互式续写（你打字，它接）
================================================================================
"""

#region 0. 导入库与路径   【模块:标准库 argparse/os/time + torch + model】
import argparse
import os
import time

import torch

from model import CharTokenizer, GPTConfig, GPT, load_ckpt, EOS

HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
DEFAULT_CKPT = os.path.join(HERE, "ckpt_novel", "best.pt")  # 默认加载小说版权重
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════════════
#region 1. 工具：打印 next-token 概率分布   【函数:show_probs(模型,分词器,提示词,显示个数,温度),return None】
@torch.no_grad()
def show_probs(model, tok, prompt, topn=12, temperature=1.0):
    """把 prompt 喂进去，打印"下一个 token 的概率分布"——这就是课程里那张柱状图。

    这个函数不采样，只是把 softmax 的结果摊开给你看：
    temperature 越大，柱子越矮越平（模型越"犹豫"），熵也越大。
    """
    # (1, T) 的输入张量；设备跟着模型走，否则会报 device mismatch
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    logits, _, _ = model(ids)  # (1,T,V)
    logits = logits[0, -1].float()  # (V,) 只留最后一个位置，转 fp32 便于精确算概率
    if temperature != 1.0:
        logits = logits / temperature  # 温度只在 softmax 之前起作用
    probs = torch.softmax(logits, dim=-1)  # (V,) 和为 1 的概率分布
    top = torch.topk(probs, topn)  # values (topn,), indices (topn,)
    print(f"\n—— 下一个 token 的 top-{topn} 概率（temperature={temperature}）——")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        c = tok.itos[i]  # id 反查字符
        # 特殊 token 用尖括号包一下，否则 <pad> 会被当成普通文本看不清
        c = {"<pad>": "⟨pad⟩", "<unk>": "⟨unk⟩", "<eos>": "⟨EOS⟩"}.get(c, c)
        bar = "█" * max(1, int(p * 60))  # 用方块长度表示概率，至少画 1 格（免得 0 概率看不见）
        print(f"  {c:>6}  {p:6.2%}  {bar}")
    print(f"  （词表 V={tok.vocab_size}，剩下 {(1-top.values.sum().item()):.1%} 的概率分散在其余 {tok.vocab_size-topn} 个 token 上）")
    # 熵：分布越平熵越大。加 1e-9 是为了防止 log(0) 变成 -inf
    print(f"  —— 熵 = {-(probs * (probs + 1e-9).log()).sum().item():.2f} nats，"
          f"熵越大越「犹豫」。temperature↑ 熵↑，temperature↓ 熵↓\n")
#endregion

#region 2. 束搜索 Beam Search   【函数:beam_search(模型,分词器,提示词,生成长度,beam宽,设备),return (文本, 平均对数概率)】
@torch.no_grad()
def beam_search(model, tok, prompt, n_tokens=60, beam=4, device="cpu"):
    """束搜索：同时保留 beam 条最优路径，每步扩展后按累计 log 概率排序、裁剪。
    课程里的定位：翻译/摘要这类"有相对标准答案"的任务用它，创意写作不用。

    这里加了一个必要的小补丁：**禁止重复 2-gram**。因为 beam 追求的是"整体概率最高"，
    而语言模型最容易的高概率路径就是原地打转（"管管管管…"）——这就是课程说的
    "beam 容易产出平均答案、缺乏惊喜"，工程上必须靠 no-repeat-ngram 之类的手段兜住。

    为什么用对数概率相加：概率连乘会下溢成 0，取 log 后变成相加，数值上安全。
    """
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)  # (1,T0)
    beams = [(0.0, ids)]                      # (累计对数概率, 序列)，一开始只有一条
    for _ in range(n_tokens):
        cands = []  # 本步所有候选：(分数, 序列)，数量 = beam × beam
        for score, seq in beams:
            ctx = seq[:, -model.cfg.block_size:]  # 超长就截断，位置编码不能越界
            logits, _, _ = model(ctx)  # (1, len(ctx), V)
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)  # (V,) 对数概率
            # 禁用已经出现过的 2-gram，防止 beam 原地打转
            s = seq[0].tolist()
            if len(s) >= 2:
                # zip(s[-60:-1], s[-59:]) 取最近 60 个 token 的所有相邻二元组
                for a, b in zip(s[-60:-1], s[-59:]):
                    if logp[b].item() > -1e9:  # 已经禁过就不用再禁（避免反复赋值）
                        logp[b] = -1e9  # 用 -1e9 而不是 -inf，防止全部被禁后 nan
            topv, topi = torch.topk(logp, beam)  # 每条路径取前 beam 个下一步
            for v, i in zip(topv.tolist(), topi.tolist()):
                # 累计分数 = 父路径分数 + 这一步的 log 概率（即相乘取对数）
                cands.append((score + v, torch.cat([seq, torch.tensor([[i]], device=device)], dim=1)))
        cands.sort(key=lambda t: -t[0])        # 按累计分数排序
        beams = cands[:beam]  # 只留最好的 beam 条，其余丢弃
        # 所有路径都以 EOS 结尾就提前收工
        if all(b[1][0, -1].item() == EOS for b in beams):
            break
    best_score, best_seq = beams[0]  # 分数最高的那条
    # 除以长度得到"平均每 token 的对数概率"，便于跨长度比较（否则长句必然分数低）
    return tok.decode(best_seq[0].tolist()), best_score / max(1, n_tokens)
#endregion

#region 3. 工具：截断文本方便并排对比   【函数:fmt(文本,长度上限),return 字符串】
def fmt(text, limit=110):
    """把文本折成两行，方便并排对比。"""
    t = text.replace("\n", "⏎")  # 换行符换成可见符号，否则表格会散
    return t[:limit] + ("…" if len(t) > limit else "")
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
#region 4. 主流程：六种策略逐个跑   【函数:main(),return None】
def main():
    #region 4.1 命令行参数   【集合:args】
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt",
                    default="老掌柜把茶碗往前一推，说道：“")  # 默认给小说式开头，这一段模型最熟
    ap.add_argument("--n", type=int, default=120, help="最多生成多少个字")
    ap.add_argument("--mode", default="all",
                    choices=["all", "greedy", "sample", "temperature", "top_k", "top_p", "beam"])
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.15, help="重复惩罚系数")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--stream", action="store_true", help="流式输出并统计 TTFT")
    ap.add_argument("--show-probs", action="store_true")  # 只打印概率分布
    ap.add_argument("--bench", action="store_true")  # 只跑 KV Cache 基准
    ap.add_argument("--interactive", action="store_true")  # 只进交互式续写
    args = ap.parse_args()
    #endregion

    #region 4.2 加载模型   【集合:model, tok, extra】
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"找不到 {args.ckpt}，先跑 python train.py")
    model, tok, extra = load_ckpt(args.ckpt, device)  # 三件套：模型 + 分词器 + 训练指标
    info = f"参数量 {model.num_params()/1e6:.2f}M · 层数 {model.cfg.n_layer} · d_model {model.cfg.d_model}"
    if "val_loss" in extra:
        import math
        info += f" · val_ppl {math.exp(extra['val_loss']):.1f}"  # loss 换回 PPL，更直观
    print(f"加载 {os.path.basename(os.path.dirname(args.ckpt))}/{os.path.basename(args.ckpt)}：{info}\n")
    #endregion

    #region 4.3 三个"单独模式"：看分布 / 测缓存 / 交互
    if args.show_probs:
        show_probs(model, tok, args.prompt)
        return

    if args.bench:
        bench_kv(model, tok, args.prompt, device)
        return

    if args.interactive:
        interactive(model, tok, device, args)
        return
    #endregion

    eos = {"stop_at_eos": True}  # 后面所有实验都开启"遇到 EOS 就停"，否则测的是空转

    # ── 1. Greedy：每步 argmax，完全确定 ──────────────────────────────────
    #region 4.4 ① 贪心解码
    if args.mode in ("all", "greedy"):
        print("① Greedy（贪心）：每步取概率最大的 token → 确定、可复现，但容易重复/死循环")
        for trial in range(2):
            t0 = time.time()
            out = model.generate(torch.tensor([tok.encode(args.prompt)], device=device),
                                 max_new_tokens=args.n, greedy=True, **eos)
            # 跑两次是为了证明"完全一样"——贪心没有任何随机性，种子都不需要
            print(f"   [第{trial+1}次 {(time.time()-t0):.1f}s] {args.prompt}{fmt(tok.decode(out))}")
        print()
    #endregion

    # ── 2. 纯温采样：只用 temperature，不截断 ─────────────────────────────
    #region 4.5 ② 只调温度
    if args.mode in ("all", "temperature"):
        print("② 只调 Temperature：T↓ 分布尖锐（保守），T↑ 分布平坦（发散、易跑偏）")
        for T in (0.5, 1.0, 1.5):
            torch.manual_seed(7)  # 固定种子，保证三组对比只有温度这一个变量不同
            out = model.generate(torch.tensor([tok.encode(args.prompt)], device=device),
                                 max_new_tokens=args.n, temperature=T, top_k=None, top_p=None, **eos)
            # top_k/top_p 都设 None：这是关键，否则温度的影响会被截断掩盖
            print(f"   T={T:<4} {fmt(tok.decode(out))}")
        print()
    #endregion

    # ── 3. Top-K ─────────────────────────────────────────────────────────
    #region 4.6 ③ Top-K
    if args.mode in ("all", "top_k"):
        print("③ Top-K：只在高概率的 K 个里选，切掉长尾垃圾（但 K 是固定数，不适配分布形态）")
        for k in (5, 50, 200):
            torch.manual_seed(7)
            out = model.generate(torch.tensor([tok.encode(args.prompt)], device=device),
                                 max_new_tokens=args.n, temperature=0.9, top_k=k, top_p=None, **eos)
            # K=5 太保守会复读，K=200 等于没截断，50 是常用折中
            print(f"   K={k:<4} {fmt(tok.decode(out))}")
        print()
    #endregion

    # ── 4. Top-P（核采样）────────────────────────────────────────────────
    #region 4.7 ④ Top-P 核采样
    if args.mode in ("all", "top_p"):
        print("④ Top-P（核采样）：候选数量随分布形状自动伸缩——这才是现在的默认做法")
        for p in (0.5, 0.9, 0.99):
            torch.manual_seed(7)
            out = model.generate(torch.tensor([tok.encode(args.prompt)], device=device),
                                 max_new_tokens=args.n, temperature=0.9, top_k=None, top_p=p, **eos)
            # P=0.5 只留最核心的几个候选，P=0.99 几乎不截断
            print(f"   P={p:<5} {fmt(tok.decode(out))}")
        print()
    #endregion

    # ── 5. 组合（工业界默认配方）+ 流式 ──────────────────────────────────
    #region 4.8 ⑤ 工业配方（温度 + K + P + 重复惩罚），可流式
    if args.mode in ("all", "sample"):
        print(f"⑤ 工业配方 temperature={args.temperature} + top_k={args.top_k} + top_p={args.top_p}"
              f" + 重复惩罚 {args.rep}")
        if args.stream:
            # ── 流式：一边生成一边打印，同时记录首字延迟 TTFT ──
            print(f"   流式输出（首字延迟 TTFT 会被记录）：\n   ", end="", flush=True)
            print(args.prompt, end="", flush=True)
            t0 = time.time()
            ttft = [None]  # 用列表存，才能在内层函数里改（闭包不能重新绑定外层变量）
            first = []  # 收集所有已生成的字符，用来算字/秒
            def on_tok(i):
                """每生成一个 token 就被 generate 回调一次，这就是流式的实现方式。"""
                if ttft[0] is None:
                    ttft[0] = time.time() - t0  # 只有第一个 token 的耗时才是 TTFT
                ch = tok.decode([i])  # 单个 id → 单个字
                first.append(ch)
                print(ch, end="", flush=True)  # flush 必须开，否则会攒在缓冲区里看不出"流"
            ids = torch.tensor([tok.encode(args.prompt)], device=device)
            model.generate(ids, max_new_tokens=args.n, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep,
                           on_token=on_tok, **eos)
            dt = time.time() - t0
            # TTFT 为什么重要：用户感知的是"多久看到第一个字"，不是总时长
            print(f"\n   [TTFT {ttft[0]*1000:.0f} ms | 共 {len(first)} 字 | {len(first)/dt:.1f} 字/秒]")
        else:
            for trial in range(2):
                torch.manual_seed(7 + trial)  # 换种子，看同一组参数下的两种可能
                out = model.generate(torch.tensor([tok.encode(args.prompt)], device=device),
                                     max_new_tokens=args.n, temperature=args.temperature,
                                     top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep, **eos)
                print(f"   种子{trial+1} {fmt(tok.decode(out))}")
        print()
    #endregion

    # ── 6. Beam Search ──────────────────────────────────────────────────
    #region 4.9 ⑥ Beam Search
    if args.mode in ("all", "beam"):
        print(f"⑥ Beam Search（beam={args.beam}）：保留 {args.beam} 条路径，平均分最高者胜出")
        for beam in (1, args.beam):
            t0 = time.time()
            text, avg = beam_search(model, tok, args.prompt, args.n, beam, device)
            # beam=1 时等价于贪心，用来做对照；平均对数概率越接近 0 说明越"自信"
            print(f"   beam={beam} [{(time.time()-t0):.1f}s 平均对数概率 {avg:.3f}] {fmt(text)}")
        print()
    #endregion
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache 基准：prefill + decode 的真实差别
# ══════════════════════════════════════════════════════════════════════════════
#region 5. KV Cache 基准   【函数:bench_kv(模型,分词器,提示词,设备),return None】
@torch.no_grad()
def bench_kv(model, tok, prompt, device):
    """KV Cache 到底能加速多少？答案是：看你能不能把 GPU 喂饱。
    关掉 EOS 提前停止（不然测的是"谁停得快"），生成 128 个 token 计时。"""
    print("KV Cache：生成第 t 个字时，能不能不重算前面 t-1 个字？\n")
    print(f"{'prompt长度':>10} {'batch':>6} {'无缓存':>10} {'有缓存':>10} {'加速比':>8}")
    print("-" * 50)
    long_text = prompt * 40  # 复制 40 遍，凑出足够长的文本供切片
    for plen in (16, 128, 400):
        base_ids = torch.tensor([tok.encode(long_text[:plen])], device=device)  # (1, plen)
        for bs in (1, 8):
            ids = base_ids.repeat(bs, 1)  # (bs, plen) 复制成 batch
            row = []  # 收集 [无缓存耗时, 有缓存耗时]，单位毫秒
            for uc in (False, True):
                # 先空跑 8 个 token 预热：第一次调用要初始化 CUDA kernel，不能计入计时
                model.generate(ids, max_new_tokens=8, use_cache=uc, stop_at_eos=False)  # 预热
                torch.cuda.synchronize() if device == "cuda" else None  # 等 GPU 真跑完，否则计时会是 0
                t0 = time.time()
                model.generate(ids, max_new_tokens=128, temperature=0.9,
                               use_cache=uc, stop_at_eos=False)
                torch.cuda.synchronize() if device == "cuda" else None  # 核函数是异步的，必须同步后再读表
                row.append((time.time() - t0) * 1000)  # 秒 → 毫秒
            print(f"{plen:>10} {bs:>6} {row[0]:>9.0f}ms {row[1]:>9.0f}ms {row[0]/row[1]:>7.1f}x")
    # 这张表的解读是本次实验最反直觉的地方，直接写进输出里
    print("""
   怎么读这张表（这是最容易被讲错的一点）：
   · batch=1 时，缓存反而略慢。因为 15M 的小模型根本喂不饱这张显卡，
     每步耗时由 kernel 启动开销决定，跟"算了多少 token"无关；缓存还要多付
     torch.cat 拼接的代价。
   · batch=8、prompt 400 时，缓存快 2~3 倍。这时算力真的被用起来了，
     "每步只算 1 个 token"才开始省钱。
   · 放到 7B 模型 + 长上下文 + 高并发，这个差距会变成 10 倍以上——
     这就是 vLLM / TensorRT-LLM 把 KV Cache 当第一优化项的原因。
   真实推理还有一个必然后果：缓存要显存。序列越长、并发越高，KV Cache 越大，
   所以才有 MQA / GQA 这些"少存几份 K/V"的技术。
""")
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 交互式续写
# ══════════════════════════════════════════════════════════════════════════════
#region 6. 交互式续写：命令行版流式   【函数:interactive(模型,分词器,设备,args),return None】
def interactive(model, tok, device, args):
    """读一行、续一段、再读一行。空行退出。"""
    print("交互式续写（空行退出）。模型是纯预训练 Base Model：它只会「接龙」，不会回答你的问题——")
    print("这正好验证课程 03 节：知识来自预训练，问答的形式要等 SFT。\n")
    while True:
        try:
            s = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl+C / Ctrl+D 都能正常退出，不抛栈
        if not s:
            break
        print("AI > " + s, end="", flush=True)  # 先把用户输入原样回显，再往后接
        ids = torch.tensor([tok.encode(s)], device=device)
        model.generate(ids, max_new_tokens=args.n, temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep,
                       on_token=lambda i: print(tok.decode([i]), end="", flush=True))  # lambda 里直接打印，就是流式
        print("\n")
#endregion


#region 7. 入口
if __name__ == "__main__":
    main()
#endregion

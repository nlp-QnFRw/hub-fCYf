"""
compare.py -- 把 GPT（单向）和 BERT（双向）的阶段指标放到一张表里

    python compare.py

读两份 CSV：
    runs/metrics.csv        train.py 产出      （next-token PPL）
    runs_bert/metrics.csv   bert_train.py 产出 （MLM loss / 命中率 / 伪 PPL）

先说清楚这两个数字**不是同一个任务**：
    GPT  预测下一个字：只能用左边，损失是 -log P(下一个字 | 左边)
    BERT 填空：左右都能看，损失是 -log P(被遮的字 | 左右两边)
所以 BERT 的伪 PPL 天然更低 —— 它偷看了未来。这张表的意义是展示
「单向 vs 双向」的差距和各自的收敛过程，而不是宣布谁更厉害。
要比生成，BERT 根本不能上桌：它没有「下一个字」的概念。

两张表：
    表一「按训练步数对齐」—— 两次训练第 N 步时各自的指标，看谁收敛得快；
    表二「成本与起点」    —— 参数量、初始状态、看过的 token、耗时。
                             两边 token 预算差 5 倍、任务还不同，硬做同预算对齐只会骗人。
"""

from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GPT_DIR = os.path.join(HERE, "runs")
BERT_DIR = os.path.join(HERE, "runs_bert")


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v, digits: int = 2) -> str:
    if v in ("", None):
        return "—"
    try:
        f = float(v)
    except ValueError:
        return str(v)
    if f != f:
        return "—"
    return f"{f:.{digits}f}"


def gpt_step_config() -> tuple[int, int]:
    path = os.path.join(GPT_DIR, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f).get("train", {})
        return int(cfg.get("batch_size", 32)), int(cfg.get("block_size", 256))
    return 32, 256


def count_gpt_params() -> int | None:
    """直接数 checkpoint 里的参数，避免手算公式。"""
    import torch
    from model import GPT, GPTConfig
    path = os.path.join(GPT_DIR, "best.pt")
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu")
    return GPT(GPTConfig.from_dict(ck["config"])).num_params


def count_bert_params() -> int | None:
    path = os.path.join(BERT_DIR, "final.pt")
    if not os.path.exists(path):
        return None
    import torch
    from bert_model import BertConfig, BertForMaskedLM
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return BertForMaskedLM(BertConfig(**ck["config"])).num_params


def main():
    gpt = read_csv(os.path.join(GPT_DIR, "metrics.csv"))
    bert = read_csv(os.path.join(BERT_DIR, "metrics.csv"))
    if not gpt and not bert:
        raise SystemExit("两份 metrics.csv 都没有，先跑 train.py 和 bert_train.py")
    if not gpt:
        print("提醒：没有 runs/metrics.csv，GPT 那几列会是空的")
    if not bert:
        print("提醒：没有 runs_bert/metrics.csv，BERT 那几列会是空的")

    bs, blk = gpt_step_config()
    for r in gpt:
        r["tokens"] = int(r["step"]) * bs * blk
    gpt_by_step = {int(r["step"]): r for r in gpt}
    bert_by_step = {int(r["step"]): r for r in bert}

    header = ("步数", "GPT token", "GPT PPL", "BERT token", "BERT loss", "BERT 命中率", "BERT 伪PPL")
    rows = []
    for s in sorted(set(gpt_by_step) | set(bert_by_step)):
        g, b = gpt_by_step.get(s), bert_by_step.get(s)
        rows.append([
            str(s),
            f"{g['tokens'] / 1e6:.2f}M" if g else "—",
            num(g["val_ppl"]) if g else "—",
            f"{int(b['tokens_seen']) / 1e6:.2f}M" if b else "—",
            num(b["val_loss"], 4) if b else "—",
            f"{float(b['mask_acc']) * 100:.1f}%" if b else "—",
            num(b["pseudo_ppl"]) if b else "—",
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print("\n表一 · 按训练步数对齐")
    print("-" * len(line))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print("-" * len(line))

    # ── 表二：成本与起点
    g_params, b_params = count_gpt_params(), count_bert_params()
    g_last = gpt[-1] if gpt else None
    b_last = bert[-1] if bert else None
    b_first = bert[0] if bert else None
    g_sec = max((float(r["sec"]) for r in gpt), default=0.0)
    b_sec = max((float(r["sec"]) for r in bert), default=0.0)

    cost = [
        ["GPT（本项目）",
         f"{g_params / 1e6:.2f}M" if g_params else "—",
         "随机初始化",
         f"PPL {num(g_last['val_ppl'])}" if g_last else "—",
         f"{int(g_last['tokens']) / 1e6:.2f}M" if g_last else "—",
         f"{g_sec:.0f}s"],
        ["BERT（bert-base-chinese）",
         f"{b_params / 1e6:.2f}M" if b_params else "—",
         "预训练权重" if b_first else "—",
         f"伪PPL {num(b_last['pseudo_ppl'])}" if b_last else "—",
         f"{int(b_last['tokens_seen']) / 1e6:.2f}M" if b_last else "—",
         f"{b_sec:.0f}s"],
    ]
    header2 = ("模型", "参数量", "起点", "终点指标", "看过 token", "训练耗时")
    w2 = [max(len(header2[i]), *(len(r[i]) for r in cost)) for i in range(len(header2))]
    line2 = "  ".join(h.ljust(w) for h, w in zip(header2, w2))
    print("\n表二 · 成本与起点（任务不同，不要横向比大小）")
    print("-" * len(line2))
    print(line2)
    print("-" * len(line2))
    for r in cost:
        print("  ".join(c.ljust(w) for c, w in zip(r, w2)))
    print("-" * len(line2))
    print("  GPT 是从零学一门语言；BERT 是在已经会中文的基础上补这门语料的领域知识。")
    print("  参数量差 29 倍、token 预算差 5 倍，两个数字放一起只说明「起点不同」。")

    out = os.path.join(BERT_DIR, "comparison.md")
    os.makedirs(BERT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 阶段对比：单向 GPT vs 双向 BERT\n\n")
        f.write("同一份语料、同一个验证集。两个指标任务不同，只用于观察趋势与量纲。\n\n")
        f.write("## 表一 · 按训练步数对齐\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")

        if b_first and b_last:
            f.write("\n## 关键变化\n\n")
            f.write(f"- BERT 阶段 0（原版 bert-base-chinese，未训练）伪 PPL "
                    f"{num(b_first['pseudo_ppl'])}、命中率 {float(b_first['mask_acc']) * 100:.1f}%；"
                    f"第 {b_last['step']} 步（{int(b_last['tokens_seen']) / 1e6:.2f}M token）"
                    f"降到伪 PPL {num(b_last['pseudo_ppl'])}、命中率 "
                    f"{float(b_last['mask_acc']) * 100:.1f}%，"
                    f"伪 PPL 相对下降 "
                    f"{(1 - float(b_last['pseudo_ppl']) / float(b_first['pseudo_ppl'])) * 100:.1f}%\n")
            f.write(f"- 这一段的收益全部来自「领域语料」：结构没动，只是在本项目语料上继续做 MLM。\n")
        if g_last:
            f.write(f"- GPT 从随机初始化（PPL≈字表大小 {3325}）训到 "
                    f"PPL {num(g_last['val_ppl'])}，全靠这 "
                    f"{int(g_last['tokens']) / 1e6:.2f}M token 自己学。\n")

        f.write("\n## 表二 · 成本与起点\n\n")
        f.write("| " + " | ".join(header2) + " |\n")
        f.write("|" + "|".join(["---"] * len(header2)) + "|\n")
        for r in cost:
            f.write("| " + " | ".join(r) + " |\n")

        f.write("\n> BERT 的伪 PPL 天生比 GPT 的 PPL 低，因为它能同时看到左右两侧；\n"
                "> 这正说明「双向编码器不适合做生成」——它没有「下一个字」的概念。\n")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()

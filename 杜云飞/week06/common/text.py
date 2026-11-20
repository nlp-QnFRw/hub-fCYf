#region 0. 模块说明
"""
文本构造与截断策略的**唯一实现**。BERT / LLM / TF-IDF 三条线共用同一套规则，
否则"max_len 消融"比的是两种不同的输入构造，结论无意义。

为什么需要三种策略（THUCNews 正文 700~1500 字，max_len=256 只覆盖 17~37%）：
  head      : 标题 + 正文开头。最朴素，作为基准。
  head_tail : 标题 + 开头 + 结尾。新闻的导语与结语各自携带大量类别线索，
              中间段落多为细节铺陈，砍中间通常比砍尾巴划算。
  title_only: 只留标题。成本最低，用来量化"标题到底有多能打"
              （如果标题就能到 85%，长文本消融的收益要重新解释）。

中文几乎 1 字 = 1 token，所以**字符预算 ≈ token 预算**，这里按字符裁剪，
再由 tokenizer 的 max_length 做兜底，避免超长样本被二次截断而丢掉尾部。

使用方式：
  from common.text import build_text, TRUNCATIONS
  s = build_text({"title": "…", "content": "…"}, mode="head_tail", max_len=256)
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

#endregion

#region 2. 全局配置
TRUNCATIONS: tuple[str, ...] = ("head", "head_tail", "title_only")
SEP: str = "。"  # 拼接标题与正文的分隔符，用句号避免把两个句子粘成一个
ELLIPSIS: str = "……"  # head_tail 中间被砍掉的标记，让模型知道这里断了
#endregion

#region 3. 函数：按策略构造输入文本
def build_text(rec: dict, mode: str = "head_tail", max_len: int = 256) -> str:
    """把一条样本（title + content）按指定策略压到约 max_len 个字符。

    参数：
      rec     : 至少包含 "title" / "content" 两个键（可为空串）
      mode    : "head" | "head_tail" | "title_only"
      max_len : 目标字符预算（约等于 token 数）；内部会预留 2 个位置给 [CLS]/[SEP]
    返回：
      str，长度约 <= max_len
    """
    if mode not in TRUNCATIONS:
        raise ValueError(f"未知截断策略 {mode}，可选 {TRUNCATIONS}")

    title = (rec.get("title") or "").strip()
    content = (rec.get("content") or "").strip()
    budget = max(8, max_len - 2)  # 给特殊符号留位

    if mode == "title_only":
        return title[:budget]

    # head / head_tail 都以标题打头：标题是类别最强的单句信号，成本为 0，没有理由丢掉
    head_txt = f"{title}{SEP}" if title else ""
    rest = budget - len(head_txt)
    if rest <= 0:
        return head_txt[:budget]

    if mode == "head":
        return head_txt + content[:rest]

    # head_tail：开头 60% + 结尾 40%（导语的信息密度高于结语，所以前段多分一点）
    n_head = int(rest * 0.6)
    n_tail = rest - n_head - len(ELLIPSIS)
    if n_tail <= 0 or len(content) <= rest:
        return head_txt + content[:rest]  # 正文本来就够短，不需要砍
    return head_txt + content[:n_head] + ELLIPSIS + content[-n_tail:]


def build_input(rec: dict, mode: str = "head_tail", max_len: int = 256,
                max_chars: int | None = None) -> str:
    """TF-IDF / LLM 路线用的薄封装：允许用 max_chars 覆盖 max_len（字数与 token 数解耦）。

    参数：
      max_chars : 给定则优先使用（TF-IDF 直接吃字符，不需要考虑 token）
    """
    return build_text(rec, mode=mode, max_len=max_chars or max_len)
#endregion

#region 4. 自检
if __name__ == "__main__":
    demo = {"title": "中超-国安3:1逆转申花", "content": "开场第15分钟申花先进一球" + "，随后国安连入三球完成逆转" * 80}
    for m in TRUNCATIONS:
        s = build_text(demo, mode=m, max_len=64)
        print(f"{m:<11} len={len(s):>4}  {s[:60]}…")
    assert len(build_text(demo, "head_tail", 64)) <= 64
    assert "……" in build_text(demo, "head_tail", 64)
    print("截断策略自检通过 ✓")
#endregion

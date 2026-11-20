#region 0. 模块说明
"""
生成式路线的 **prompt 与解析的唯一实现**（本地 LLM 与 API LLM 共用）。

为什么必须共用：
  "本地 Qwen zero-shot" 与 "DeepSeek zero-shot" 如果不是同一套 prompt、同一套解析，
  两者的差距就同时包含"模型能力差"和"prompt 差"，结论无法归因。
  共用之后，唯一的变量是模型本身——这才是可比的口径。

三个设计决定（都有代价，报告里要写明）：
  1. **固定编号白名单**：把 14 个类别编号列出，要求只输出类名。
     代价是 prompt 变长（约 +80 token），收益是"标签空间闭合"，大幅降低编造类名的概率。
  2. **三档解析**：strict（去标点后全等）/ lenient（包含即算，取最长匹配）/ abstain（无匹配）。
     主表口径 = 全样本计入分母（失败算错），失败率单列。
  3. **few-shot 示例按类分层轮转**：不是随机抽，保证 N=1/3/5 时覆盖尽可能多的类，
     否则"3-shot 比 1-shot 差"可能只是示例恰好选了两个相似类。

使用方式：
  from common.prompt import build_messages, parse_label
  msgs = build_messages(rec, shots=2, max_chars=600)
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import re

from common.labels import LABELS
from common.text import build_text

#endregion

#region 2. 全局配置
SYSTEM_PROMPT: str = (
    "你是中文新闻分类助手。只能从给定类别中选择一个，不要解释，不要输出标点或多余文字。"
)

# 固定的 14 类清单（编号 + 类名），写死在 prompt 里，让标签空间闭合
CLASS_LIST: str = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(LABELS))

# few-shot 示例池：每类 1 条，短文本（省 token），按类轮转选取
FEW_SHOT_POOL: list[dict] = [
    {"label": "财经", "text": "央行今日宣布下调存款准备金率 0.5 个百分点，释放长期资金约 1 万亿元，"
                              "以支持实体经济发展。"},
    {"label": "彩票", "text": "双色球第 2024150 期开奖，红球 03 07 12 18 25 31，蓝球 09，"
                              "一等奖开出 8 注，单注奖金 720 万元。"},
    {"label": "房产", "text": "多地优化限购政策，二手房成交量环比上涨，开发商加大推盘力度，"
                              "房贷利率降至历史低位。"},
    {"label": "股票", "text": "A股三大指数集体收涨，沪指涨 1.2% 收复 3100 点，"
                              "券商板块领涨，两市成交额突破 9000 亿元。"},
    {"label": "家居", "text": "小户型装修如何扩容？设计师建议用浅色系搭配多功能家具，"
                              "橱柜与卫浴尽量做嵌入式。"},
    {"label": "教育", "text": "教育部发布通知，要求各地做好高考志愿填报指导工作，"
                              "严禁炒作高分考生和升学率。"},
    {"label": "科技", "text": "华为发布新一代旗舰手机，搭载自研芯片与全新操作系统，"
                              "影像系统由多颗摄像头组成。"},
    {"label": "社会", "text": "当地警方通报，一起交通事故造成两人受伤，伤者已送医救治，"
                              "事故原因正在调查中。"},
    {"label": "时尚", "text": "本季时装周主打复古风，宽松廓形与低饱和配色成为潮流，"
                              "搭配一只亮色单肩包即可提升整体造型。"},
    {"label": "时政", "text": "国务院总理主持召开国务院常务会议，研究部署进一步扩大内需、"
                              "促进消费的政策措施。"},
    {"label": "体育", "text": "中超第 20 轮，北京国安主场 3:1 逆转上海申花，"
                              "张玉宁梅开二度，球队取得联赛三连胜。"},
    {"label": "星座", "text": "本周白羊座运势回升，工作上容易得到领导认可，"
                              "感情方面适合主动表达；注意水逆期间的沟通误会。"},
    {"label": "游戏", "text": "该游戏新版本上线，新增两个副本与一套装备，"
                              "玩家反馈平衡性调整后体验更好。"},
    {"label": "娱乐", "text": "这部悬疑电影上映首周票房破 5 亿，导演与主演现身首映礼，"
                              "口碑持续走高。"},
]

PUNCT = " \t\r\n，,。.、；;：:！!？?\"'“”‘’（）()【】[]《》<>-—_"
#endregion

#region 3. 函数：构造消息
def pick_shots(n: int, exclude_label: str | None = None) -> list[dict]:
    """按类轮转取 n 条示例（尽量覆盖不同类别）。

    参数：
      n             : shot 数（0 表示 zero-shot）
      exclude_label : 当前样本的真实类别；若不为 None，**优先避开同类**，
                      否则模型会倾向复制示例标签（尤其小模型）
    """
    if n <= 0:
        return []
    pool = [e for e in FEW_SHOT_POOL if e["label"] != exclude_label] or FEW_SHOT_POOL
    return [pool[i % len(pool)] for i in range(n)]


def build_messages(rec: dict, shots: int = 0, max_chars: int = 600,
                   truncation: str = "head_tail", shot_style: str = "turn") -> list[dict]:
    """构造 chat messages。

    参数：
      rec        : {"title", "content", "label"?}
      shots      : few-shot 条数
      max_chars  : 输入正文的字符预算（提示词成本的第一大来源，必做消融）
      truncation : 截断策略（与 BERT 侧保持同一套口径）
      shot_style : "turn"   每个示例都是独立的一轮 user/assistant（标准对话式）
                   "inline" 所有示例压缩进**同一条** user 消息（`文本 → 类别` 逐行）
    返回：
      [{"role": "system", ...}, ...few-shot 对..., {"role": "user", ...}]

    为什么要有 shot_style：
      0.5B 级模型在多轮示例后会"接着往下写正文"（实测 5-shot 有 93% 输出是正文续写，
      而不是类名）。inline 把示例压成几行 `文本 → 类别`，模型只需续写最后一个类名，
      格式负担小得多。两者对比才能分清"few-shot 没用"和"多轮格式不适合小模型"。
    """
    body = build_text(rec, mode=truncation, max_len=max_chars)
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    examples = pick_shots(shots, rec.get("label"))

    if shot_style == "inline" and examples:
        # 示例与待分类文本放进同一条 user 消息，避免多轮结构
        demo = "\n".join(f"新闻：{ex['text']}\n类别：{ex['label']}" for ex in examples)
        msgs.append({"role": "user", "content":
                     f"可选类别：\n{CLASS_LIST}\n\n下面是几个示例：\n{demo}\n\n"
                     f"现在分类这条新闻：\n{body}\n\n"
                     f"只输出一个类别名（从上面 14 个里选），不要解释。"})
        return msgs

    for ex in examples:
        msgs.append({"role": "user", "content": _user_text(ex["text"])})
        msgs.append({"role": "assistant", "content": ex["label"]})

    msgs.append({"role": "user", "content": _user_text(body)})
    return msgs


def _user_text(body: str) -> str:
    """拼装单条用户消息（类别清单 + 待分类文本 + 输出约束）。"""
    return (f"可选类别：\n{CLASS_LIST}\n\n"
            f"待分类新闻：\n{body}\n\n"
            f"只输出一个类别名（从上面 14 个里选），不要解释。")
#endregion

#region 4. 函数：解析输出
def parse_label(text: str, mode: str = "lenient") -> tuple[str | None, str]:
    """把模型输出解析成类名。

    参数：
      text : 模型原始输出
      mode : "strict"（去标点后必须全等）| "lenient"（包含即算，取最长匹配）
    返回：
      (类名 或 None, 状态 "ok"|"abstain")
    """
    if not text:
        return None, "abstain"
    t = text.strip()
    if mode == "strict":
        core = t.strip(PUNCT)
        return (core, "ok") if core in LABELS else (None, "abstain")

    # lenient：取输出中出现的、最长的类名（避免"体育"命中"体育彩票"这类包含关系）
    hits = [name for name in LABELS if name in t]
    if not hits:
        # 兜底：模型可能只输出编号，如 "11"
        m = re.search(r"\b(\d{1,2})\b", t)
        if m and 1 <= int(m.group(1)) <= len(LABELS):
            return LABELS[int(m.group(1)) - 1], "ok"
        return None, "abstain"
    return max(hits, key=len), "ok"
#endregion

#region 5. 自检
if __name__ == "__main__":
    rec = {"title": "中超-国安3:1逆转申花", "content": "张玉宁梅开二度。" * 30, "label": "体育"}
    msgs = build_messages(rec, shots=2, max_chars=120)
    print(f"messages 条数 {len(msgs)}（1 system + 2×2 few-shot + 1 user）")
    print("最后一条 user 前 120 字：\n", msgs[-1]["content"][:120], "...")
    print("示例类别（应避开体育）：", [m["content"] for m in msgs[2:-1] if len(m["content"]) < 6])
    for out in ("体育", "体育。", "  11 ", "这是一条体育新闻", "足球"):
        print(f"  解析 {out!r:<16} -> {parse_label(out)}")
    assert parse_label("体育。", "strict") == ("体育", "ok")
    assert parse_label("这是一条体育新闻")[0] == "体育"
    assert parse_label("足球")[0] is None
    print("prompt / 解析自检通过 ✓")
#endregion

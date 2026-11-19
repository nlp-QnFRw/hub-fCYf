"""
方法一：规则 / 关键词（L0）

两档规则，对应现实里的两种做法：
  naive    —— 正文里出现类别名就判该类（原项目的做法，用来暴露这种规则的脆弱）
  keywords —— 手工维护一张领域词表（球员/比赛→体育，股价/涨停→财经…）

为什么值得把 keywords 这档写出来：
  生产里最先上线的往往就是它——零成本、可解释、出问题能立刻改。
  它决定了「模型至少要赢它多少才值得上」，所以这条线不是陪跑。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits, subset
from common.labels import CLASSES
from common.metrics import classification_report, pred_distribution, to_markdown_table

# 手工词表：每类挑 8~12 个高区分度词。刻意不追求完备，用来量化「人工规则的天花板」。
KEYWORDS = {
    "体育": ["球员", "比赛", "联赛", "球队", "主帅", "进球", "赛季", "奥运", "冠军", "球迷"],
    "娱乐": ["明星", "电影", "导演", "主演", "演唱会", "综艺", "票房", "剧组", "歌手", "恋情"],
    "家居": ["装修", "家具", "客厅", "户型", "软装", "瓷砖", "衣柜", "地板", "厨房", "卫浴"],
    "房产": ["楼市", "房价", "开发商", "楼盘", "购房", "二手房", "土地出让", "限购", "房贷", "万科"],
    "教育": ["学生", "高考", "学校", "老师", "考研", "招生", "课程", "教育部", "留学", "校园"],
    "时尚": ["时尚", "穿搭", "妆容", "品牌", "秀场", "单品", "时装", "美妆", "造型", "奢侈品"],
    "时政": ["国务院", "总书记", "省委", "人大", "外交", "政策", "会议", "部长", "主席", "政府"],
    "游戏": ["玩家", "游戏", "手游", "版本", "副本", "电竞", "主机", "steam", "关卡", "装备"],
    "科技": ["手机", "芯片", "华为", "发布会", "技术", "互联网", "人工智能", "5G", "屏幕", "系统"],
    "财经": ["股价", "涨停", "市值", "营收", "净利润", "券商", "上市公司", "融资", "财报", "板块"],
}


def classify_naive(text: str, fallback: int) -> int:
    for i, name in enumerate(CLASSES):
        if name in text:
            return i
    return fallback


def classify_keywords(text: str, fallback: int) -> int:
    """命中词数最多的一类胜出；一个都没命中就退化成多数类"""
    best_i, best_n = fallback, 0
    for i, name in enumerate(CLASSES):
        n = sum(1 for w in KEYWORDS[name] if w in text)
        if n > best_n:
            best_i, best_n = i, n
    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部验证集")
    ap.add_argument("--out", default="outputs/rule_pred.json")
    args = ap.parse_args()

    train, val, meta = load_splits()
    samples = subset(val, args.n) if args.n else val

    from collections import Counter
    fallback = Counter(d["label"] for d in train).most_common(1)[0][0]

    y_true = [d["label"] for d in samples]
    results, reports = {}, {}
    for mode, fn in (("naive", classify_naive), ("keywords", classify_keywords)):
        y_pred = [fn(d["text"], fallback) for d in samples]
        r = classification_report(y_true, y_pred)
        r["method"] = f"rule_{mode}"
        r["pred_distribution"] = pred_distribution(y_pred)
        reports[mode] = {"report": r,
                         "preds": [{"text": d["text"][:60], "true": d["label"], "pred": p}
                                   for d, p in zip(samples, y_pred)]}
        results[mode] = r

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "modes": reports}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"评测 {len(samples)} 条")
    print(to_markdown_table({"规则·类名匹配": results["naive"], "规则·关键词表": results["keywords"]}))
    print(f"→ 写入 {out}")


if __name__ == "__main__":
    main()

#region 0. 模块说明
"""
L0：规则 / 关键词基线。

作用不是"赢"，而是给所有后续方法一个**下限**：连关键词都不如的方法，
说明它在这个任务上学到的不是类别本身。同时它能量化一个常被忽略的事实——
新闻类别的头部信号（"涨停"→股票、"中超"→体育）有多强，剩下的百分点才是模型的价值。

打分方式：
  score[c] = 2×标题命中数 + 1×正文命中数（标题权重高，因为标题是人工拟的类别浓缩）
  并列时取 LABELS 中靠前的类（避免随机性）；全 0 落 "社会"（THUCNews 的默认大类）。

置信度：
  用 softmax(score / T) 得到伪概率（T=1.5），**这不是校准过的概率**，
  只用于 ECE 与队列路由的演示，报告里必须注明。

使用方式：
  python rule/classify_rule.py
  python rule/classify_rule.py --n 500 --demo     # 冒烟
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import append_index, new_run_id, write_meta, write_preds, write_train_log  # noqa: E402
from common.data import load_split  # noqa: E402
from common.labels import LABELS  # noqa: E402
from common.paths import ROOT  # noqa: E402
from common.seed import set_seed  # noqa: E402

#endregion

#region 2. 全局配置：关键词表 + 权重
# 关键词按"命中即几乎确定类别"的程度挑选，避免用泛词（"公司"、"记者"这种会全类命中）
KEYWORDS: dict[str, tuple[str, ...]] = {
    "财经": ("GDP", "央行", "财政", "货币政策", "通胀", "CPI", "汇率", "经济增速", "发改委", "税收"),
    "彩票": ("双色球", "大乐透", "开奖", "福彩", "体彩", "中奖", "奖池", "号码走势"),
    "房产": ("楼市", "房价", "房地产", "限购", "房贷", "开发商", "土地出让", "二手房", "商品房"),
    "股票": ("A股", "沪指", "深成指", "涨停", "跌停", "个股", "上证指数", "创业板", "券商", "收盘"),
    "家居": ("装修", "家具", "家装", "建材", "橱柜", "沙发", "软装", "地板", "卫浴"),
    "教育": ("高考", "考研", "招生", "学校", "留学", "教师", "教育部", "录取", "课程", "学生"),
    "科技": ("手机", "芯片", "人工智能", "5G", "互联网", "苹果公司", "华为", "操作系统", "算法", "发布会"),
    "社会": ("警方", "事故", "案件", "居民", "社区", "网友", "城管", "救援", "现场", "通报"),
    "时尚": ("穿搭", "时装", "妆容", "奢侈品", "口红", "潮流", "时装周", "护肤", "单品"),
    "时政": ("国务院", "主席", "总理", "外交部", "人大", "政协", "政策", "会议", "签署", "访问"),
    "体育": ("中超", "NBA", "世界杯", "球队", "联赛", "进球", "冠军", "比分", "赛季", "球员"),
    "星座": ("星座", "运势", "水逆", "白羊座", "金牛座", "双子座", "狮子座", "天蝎座", "桃花运"),
    "游戏": ("游戏", "玩家", "手游", "主机", "副本", "电竞", "新版本", "装备", "Steam"),
    "娱乐": ("明星", "电影", "综艺", "演唱会", "导演", "演员", "票房", "绯闻", "剧组", "专辑"),
}
TITLE_WEIGHT: float = 2.0
SOFTMAX_T: float = 1.5
DEFAULT_CLASS: str = "社会"
#endregion

#region 3. 函数：打分与预测
def score_record(rec: dict) -> np.ndarray:
    """返回长度 14 的分数向量（顺序同 LABELS）。"""
    title = rec.get("title", "") or ""
    content = rec.get("content", "") or ""
    scores = np.zeros(len(LABELS), dtype=np.float64)
    for i, name in enumerate(LABELS):
        kw = KEYWORDS.get(name, ())
        if not kw:
            continue
        s = sum(title.count(k) for k in kw) * TITLE_WEIGHT + sum(content.count(k) for k in kw)
        scores[i] = s
    return scores


def predict_one(rec: dict) -> tuple[str, np.ndarray, float]:
    """返回 (预测类名, 伪概率向量, 该条的 delay_ms 不在这里算)。

    全 0 分时落 DEFAULT_CLASS，且伪概率接近均匀——这在 ECE 上会被记为"高置信错误"，
    是我们想要暴露的现象。
    """
    s = score_record(rec)
    if s.sum() <= 0:
        probs = np.ones(len(LABELS)) / len(LABELS)
        return DEFAULT_CLASS, probs, 0.0
    z = (s - s.max()) / SOFTMAX_T
    e = np.exp(z)
    probs = e / e.sum()
    return LABELS[int(s.argmax())], probs, float(probs.max())
#endregion

#region 4. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="L0 规则/关键词基线")
    ap.add_argument("--n", type=int, default=None, help="限制测试集条数（冒烟）")
    ap.add_argument("--demo", action="store_true", help="只跑 20 条，打印样例")
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    set_seed(42)
    if args.demo:
        args.n = 20
    run_id = args.run_id or new_run_id("t1", "rule", "keyword")
    t0 = time.time()

    results: dict[str, list[dict]] = {}
    prob_dump: dict[str, np.ndarray] = {}
    for split in ("test_nat_10k", "test_bal"):
        recs = load_split(split, limit=args.n)
        preds: list[dict] = []
        probs_list: list[np.ndarray] = []
        for r in recs:
            ts = time.time()
            pred, probs, conf = predict_one(r)
            preds.append({
                "id": r["id"], "gold": r["label"], "pred": pred,
                "raw": f"score_max={conf:.3f}", "status": "ok",
                "latency_ms": (time.time() - ts) * 1000,
            })
            probs_list.append(probs)
        results[split] = preds
        prob_dump[split] = np.vstack(probs_list) if probs_list else np.zeros((0, len(LABELS)))
        n_ok = sum(1 for p in preds if p["pred"] == p["gold"])
        print(f"  {split}: {len(preds)} 条，acc={n_ok / max(1, len(preds)):.4f}")

    for split, preds in results.items():
        write_preds(run_id, split, preds)

    np.savez_compressed(ROOT / "runs" / run_id / "probs.npz",
                        **{k: v for k, v in prob_dump.items()})  # 供 ECE 使用

    if args.demo:
        print("\n[Demo 模式] 前 10 条：")
        for p, rec in list(zip(results["test_bal"], load_split("test_bal", limit=10)))[:10]:
            print(f"  真值 {p['gold']:<3} 预测 {p['pred']:<3} | {(rec.get('title') or rec['content'])[:40]}")
        print("[Demo 模式] 本轮为演示，产物已落盘但规模极小，不可用于对比")

    write_meta(run_id, tier="t1", method="rule", variant="keyword",
               backbone="规则表", n_train=0, seed=42,
               hyperparams={"title_weight": TITLE_WEIGHT, "softmax_T": SOFTMAX_T,
                            "n_keywords": {k: len(v) for k, v in KEYWORDS.items()}},
               elapsed_s=round(time.time() - t0, 1), device="cpu", peak_mem_gb=0.0, cost_usd=0.0)
    write_train_log(run_id, [])
    append_index(run_id, {"method": "rule"})

    print(f"\nrun_id = {run_id}")
    print(f"下一步：python eval/evaluate.py --run {run_id}")


if __name__ == "__main__":
    main()
#endregion

#region 0. 模块说明
"""
L1：TF-IDF + 线性分类器。20 秒~几分钟就能出结果，是判断"这任务到底难不难"的最快探针。

两组对照（都是单变量消融）：
  特征粒度：char_wb(1,3)（不依赖分词，中文短句鲁棒） vs word（jieba 分词）
  分类器  ：LogisticRegression（有概率，可算 ECE） vs LinearSVC（更快，无概率）

为什么用 char_wb 而不是 word 作默认：
  中文新闻里的实体（"双色球""沪指"）在分词器眼里可能被切碎，字 n-gram 反而稳；
  而且不需要额外依赖。word 作对照，用来回答"分词到底值不值"。

口径注意：
  - 训练文本截断到 max_chars（默认 1500 字），TF-IDF 对长度不敏感但内存敏感；
  - LinearSVC 没有 predict_proba，**不硬凑 decision_function 当概率**（规范 §3.9），
    该 run 的 ECE 记为 None。

使用方式：
  python tfidf/train_tfidf.py                        # char + LR，100k
  python tfidf/train_tfidf.py --feature word --clf svm
  python tfidf/train_tfidf.py --n_train 5000 --demo
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
from common.text import build_text  # noqa: E402

#endregion

#region 2. 全局配置
MAX_FEATURES: int = 50_000
MIN_DF: int = 3
MAX_CHARS: int = 1500
#endregion

#region 3. 函数：文本预处理
def make_texts(recs: list[dict], feature: str, max_chars: int) -> list[str]:
    """按特征类型准备文本：char 直接用原文，word 先分词再用空格连接。"""
    raw = [build_text(r, mode="head", max_len=max_chars) for r in recs]
    if feature == "char":
        return raw
    try:
        import jieba  # 延迟导入：没装也能跑 char 分支
    except ImportError:
        raise SystemExit("word 特征需要 jieba：pip install jieba")
    jieba.setLogLevel(20)  # 关掉 "Building prefix dict" 之类的噪音
    return [" ".join(jieba.cut(t)) for t in raw]


def build_vectorizer(feature: str):
    """构造 TfidfVectorizer。char_wb 对边界更敏感，适合中文无空格。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if feature == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                               max_features=MAX_FEATURES, min_df=MIN_DF, sublinear_tf=True)
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                           max_features=MAX_FEATURES, min_df=MIN_DF, sublinear_tf=True)
#endregion

#region 4. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="L1 TF-IDF + 线性分类器")
    ap.add_argument("--feature", choices=["char", "word"], default="char")
    ap.add_argument("--clf", choices=["lr", "svm"], default="lr")
    ap.add_argument("--n_train", type=int, default=100_000)
    ap.add_argument("--max_chars", type=int, default=MAX_CHARS)
    ap.add_argument("--n_test", type=int, default=None, help="限制测试条数（冒烟）")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    set_seed(42)
    if args.demo:
        args.n_train, args.n_test = 5000, 500
        print("[Demo 模式] 5000 训练 / 500 测试，产物不可用于对比")

    tag = f"{args.feature}-{args.clf}-{args.n_train // 1000}k"
    run_id = args.run_id or new_run_id("t1", "tfidf", tag)
    t0 = time.time()

    train = load_split(f"sub_{args.n_train // 1000}k") if args.n_train <= 50_000 else load_split("train_100k")
    if args.n_train < len(train):
        train = train[: args.n_train]
    print(f"训练集 {len(train)} 条")

    print("预处理文本 ...")
    x_train_txt = make_texts(train, args.feature, args.max_chars)
    y_train = np.array([r["label_id"] for r in train])

    print("拟合 TF-IDF ...")
    vec = build_vectorizer(args.feature)
    X = vec.fit_transform(x_train_txt)
    print(f"特征矩阵 {X.shape}，稀疏度 {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.4f}%")

    print(f"训练 {args.clf} ...")
    t_fit = time.time()
    if args.clf == "lr":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs", n_jobs=-1)
    else:
        from sklearn.svm import LinearSVC

        clf = LinearSVC(C=0.5)
    clf.fit(X, y_train)
    fit_s = time.time() - t_fit
    print(f"拟合耗时 {fit_s:.1f}s")

    probs_dump: dict[str, np.ndarray] = {}
    for split in ("test_nat_10k", "test_bal"):
        recs = load_split(split, limit=args.n_test)
        Xt = vec.transform(make_texts(recs, args.feature, args.max_chars))
        t_inf = time.time()
        pred_ids = clf.predict(Xt)
        infer_s = time.time() - t_inf
        per = infer_s / max(1, len(recs)) * 1000

        if args.clf == "lr":
            probs = clf.predict_proba(Xt)
            # 保证概率矩阵列顺序与 LABELS 一致（clf.classes_ 是 label_id 的升序）
            full = np.zeros((len(recs), len(LABELS)))
            for j, c in enumerate(clf.classes_):
                full[:, int(c)] = probs[:, j]
            probs_dump[split] = full
        else:
            probs_dump[split] = np.zeros((0, 0))  # 无概率，evaluate 时 ECE 记 None

        preds = [{
            "id": r["id"], "gold": r["label"], "pred": LABELS[int(p)],
            "raw": f"clf={args.clf}", "status": "ok", "latency_ms": per,
        } for r, p in zip(recs, pred_ids)]
        write_preds(run_id, split, preds)
        acc = float(np.mean([p["pred"] == p["gold"] for p in preds]))
        print(f"  {split}: {len(preds)} 条  acc={acc:.4f}  {per:.3f} ms/条")

    np.savez_compressed(ROOT / "runs" / run_id / "probs.npz", **probs_dump)

    if args.demo:
        print("\n[Demo 模式] 预测样例：")
        for p in load_split("test_bal", limit=5):
            print(f"  真值 {p['label']:<3} | {(p.get('title') or p['content'])[:40]}")

    write_meta(run_id, tier="t1", method="tfidf", variant=tag,
               backbone=f"tfidf-{args.feature}", n_train=len(train), seed=42,
               hyperparams={"feature": args.feature, "clf": args.clf, "C": 4.0 if args.clf == "lr" else 0.5,
                            "max_features": MAX_FEATURES, "min_df": MIN_DF,
                            "ngram": "(1,3)" if args.feature == "char" else "word",
                            "max_chars": args.max_chars, "n_features": int(X.shape[1])},
               elapsed_s=round(time.time() - t0, 1), device="cpu", peak_mem_gb=0.0, cost_usd=0.0)
    write_train_log(run_id, [{"stage": "fit", "elapsed_s": round(fit_s, 1),
                              "n_features": int(X.shape[1]), "n_train": len(train)}])
    append_index(run_id, {"method": "tfidf"})

    print(f"\nrun_id = {run_id}")
    print(f"下一步：python eval/evaluate.py --run {run_id}")


if __name__ == "__main__":
    main()
#endregion

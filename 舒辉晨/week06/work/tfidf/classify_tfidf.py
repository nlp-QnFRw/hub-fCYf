"""
方法二：TF-IDF + 线性分类器（L1）

在长文本上它的表现通常比短标题更好——因为词面证据变多了。
两个关键选择：
  - 字 1-2gram 而非分词：省掉 jieba 依赖，中文新闻里「股/楼/球/赛」这类字自带信号；
  - sublinear_tf + min_df=3：长文档里词频差异极大，取 log 能压住高频词的支配。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits, subset
from common.metrics import classification_report, to_markdown_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ngram", default="1,2")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_features", type=int, default=400000)
    ap.add_argument("--out", default="outputs/tfidf_pred.json")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    train, val, meta = load_splits()
    samples = subset(val, args.n) if args.n else val

    lo, hi = (int(x) for x in args.ngram.split(","))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi),
                          max_features=args.max_features, min_df=3, sublinear_tf=True)

    t0 = time.time()
    Xtr = vec.fit_transform([d["text"] for d in train])
    Xte = vec.transform([d["text"] for d in samples])
    clf = LinearSVC(C=args.C).fit(Xtr, [d["label"] for d in train])
    y_pred = clf.predict(Xte).tolist()
    fit_s = round(time.time() - t0, 1)

    y_true = [d["label"] for d in samples]
    rep = classification_report(y_true, y_pred)
    rep.update({"method": "tfidf_svm", "fit_seconds": fit_s, "ngram": args.ngram,
                "n_features": int(Xtr.shape[1]), "C": args.C})

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {**meta, "n_eval": len(samples)}, "report": rep,
                               "preds": [{"text": d["text"][:60], "true": d["label"], "pred": p}
                                         for d, p in zip(samples, y_pred)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {len(samples)} 条 | 特征 {Xtr.shape[1]:,} 维 | 训练+推理 {fit_s}s | "
          f"ngram={args.ngram} C={args.C}")
    print(to_markdown_table({"TF-IDF+SVM": rep}))
    print(f"→ 写入 {out}")


if __name__ == "__main__":
    main()

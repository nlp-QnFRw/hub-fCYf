"""
数据量实验：训练集越小，「微调」是不是越值钱？

做法：固定验证集不变，只改训练集比例，比较 TF-IDF 与预训练模型微调的 Macro-F1。
原项目在 TNEWS **短标题**上观察到「数据越少，微调优势越大」——TF-IDF 只能复用训练集里
出现过的字面特征，而微调还有预训练知识垫底。本任务换成 900 字正文，结论是否成立由数据说：
实测差距随数据量**扩大**（15% 时 −0.14 点，100% 时 +0.62 点），说明该规律依赖任务形态。

用法：
  python eval/scan_scale.py --model rbt3                 # TF-IDF + rbt3，约 15 分钟
  python eval/scan_scale.py --model none                 # 只扫 TF-IDF，约 1 分钟
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import load_splits

OUT = ROOT / "outputs"


def run_tfidf(train, val, cfg):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.metrics import f1_score

    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=400000,
                          min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform([d["text"] for d in train])
    Xte = vec.transform([d["text"] for d in val])
    clf = LinearSVC(C=cfg["C"]).fit(Xtr, [d["label"] for d in train])
    pred = clf.predict(Xte)
    return round(float(f1_score([d["label"] for d in val], pred, average="macro")), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rbt3", choices=["rbt3", "macbert", "none"])
    ap.add_argument("--ratios", default="0.15,0.3,0.6,1.0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="outputs/scan.json")
    args = ap.parse_args()

    train, val, meta = load_splits()
    cfg = {"C": 1.0}
    rows = []
    for r in [float(x) for x in args.ratios.split(",")]:
        n = int(len(train) * r)
        sub = train[:n]
        t0 = time.time()
        tf = run_tfidf(sub, val, cfg)
        row = {"ratio": r, "n_train": n, "tfidf_f1": tf, "tfidf_s": round(time.time() - t0, 1)}

        if args.model != "none":
            tag = f"scan_{args.model}_{int(r*100)}"
            cmd = [sys.executable, str(ROOT / "bert" / "train_bert.py"), "--model", args.model,
                   "--n_train", str(n), "--epochs", str(args.epochs), "--tag", tag]
            print(f"\n>>> {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, cwd=str(ROOT))
            rep = json.loads((OUT / f"{tag}_pred.json").read_text(encoding="utf-8"))["report"]
            row[f"{args.model}_f1"] = rep["macro_f1"]
            row["gap"] = round(rep["macro_f1"] - tf, 4)
        rows.append(row)
        print(f"[scan] ratio={r} n={n} tfidf={tf} {args.model}={row.get(f'{args.model}_f1', '-')}")

    p = ROOT / args.out
    p.write_text(json.dumps({"meta": meta, "model": args.model, "rows": rows},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {p}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

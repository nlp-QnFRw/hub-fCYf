"""生成数据概况表（写进 data/DATA.md），顺便确认最大长度没有超过模型截断阈值。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DATA_PROCESSED  # noqa: E402
from src.data import load_labels, load_splits  # noqa: E402


def main() -> None:
    out = ["# 数据概况（自动生成）", ""]
    out += ["| 数据集 | split | 条数 | 字符数 | 实体数 | 平均字数 | 平均实体/条 | 无实体条数 | 最长句 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ds in ("resume", "cluener", "msra"):
        splits = load_splits(ds)
        for split, rows in splits.items():
            n = len(rows)
            chars = sum(len(r["text"]) for r in rows)
            ents = sum(len(r["entities"]) for r in rows)
            empty = sum(1 for r in rows if not r["entities"])
            mx = max(len(r["text"]) for r in rows)
            out.append(f"| {ds} | {split} | {n} | {chars} | {ents} | {chars/max(1,n):.1f} | "
                       f"{ents/max(1,n):.2f} | {empty} | {mx} |")
        out.append("")

    for ds in ("resume", "cluener", "msra"):
        labels = load_labels(ds)
        splits = load_splits(ds)
        cnt = {lab: Counter() for lab in labels}
        for split, rows in splits.items():
            for r in rows:
                for e in r["entities"]:
                    cnt[e["label"]][split] += 1
        out += [f"## {ds} 各类实体数（{len(labels)} 类）", "",
                "| 标签 | train | dev | test | 合计 |", "| --- | ---: | ---: | ---: | ---: |"]
        for lab in sorted(labels, key=lambda l: -sum(cnt[l].values())):
            tot = sum(cnt[lab].values())
            out.append(f"| {lab} | {cnt[lab]['train']} | {cnt[lab]['dev']} | {cnt[lab]['test']} | {tot} |")
        out.append("")

    (DATA_PROCESSED / "DATA.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:10]))
    print(f"\n-> {DATA_PROCESSED / 'DATA.md'}")


if __name__ == "__main__":
    main()

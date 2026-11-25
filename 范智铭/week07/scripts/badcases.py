"""挑几组「同一个句子、不同方法对错不同」的例子，给报告做定性分析。

    python scripts/badcases.py --dataset cluener --top 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import OUTPUTS, REPORTS, read_jsonl  # noqa: E402
from src.data import load_labels, load_splits  # noqa: E402


def entset(row: dict) -> set:
    return {(e["start"], e["end"], e["label"]) for e in row.get("entities", [])}


def load_pred(ds: str, method: str) -> dict[str, dict]:
    if method.startswith("llm"):
        p = OUTPUTS / ds / "llm"
        cands = sorted(p.glob(f"pred_{method.split('_')[1]}_*.jsonl"))
        cands = [c for c in cands if "smoke" not in c.name and "mock" not in c.name]
        if not cands:
            return {}
        path = cands[-1]
    else:
        path = OUTPUTS / ds / method / "pred.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in read_jsonl(path)}


def highlight(text: str, spans: list[tuple[int, int]], mark: str = "【") -> str:
    out, last = [], 0
    for s, e in sorted(spans):
        out.append(text[last:s])
        out.append(f"{mark}{text[s:e]}】")
        last = e
    out.append(text[last:])
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cluener")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max_chars", type=int, default=120)
    args = ap.parse_args()

    gold = {r["id"]: r for r in load_splits(args.dataset)["test"]}
    methods = ["bilstm", "bert", "roberta", "llm_zero", "llm_few"]
    preds = {m: load_pred(args.dataset, m) for m in methods}
    preds = {m: p for m, p in preds.items() if p}
    print(f"可用方法: {list(preds)}")

    lines = [f"# 定性对比样例（{args.dataset}）", ""]
    for a in preds:
        for b in preds:
            if a >= b:
                continue
            # 找 A 全对、B 至少错一个实体的句子
            cases = []
            for rid, g in gold.items():
                if rid not in preds[a] or rid not in preds[b]:
                    continue
                gs = entset(g)
                if not gs:
                    continue
                pa, pb = entset(preds[a][rid]), entset(preds[b][rid])
                if gs <= pa and gs - pb and len(g.get("entities", [])) >= 2:
                    cases.append((rid, g, pa, pb))
            for rid, g, pa, pb in cases[:args.top]:
                text = g["text"]
                if len(text) > args.max_chars:
                    continue
                lines += [f"## {a} 对 / {b} 错 · `{rid}`", "", "```text", text, "```", ""]
                lines.append("金标：" + "、".join(f"{e['label']}「{text[e['start']:e['end']]}」"
                                                for e in g["entities"]))
                wrong = [e for e in g["entities"] if (e["start"], e["end"], e["label"]) not in pb]
                lines.append(f"{b} 漏/错：" + "、".join(f"{e['label']}「{text[e['start']:e['end']]}」"
                                                      for e in wrong))
                extra = [e for e in pb if e not in gs]
                if extra:
                    lines.append(f"{b} 多标：" + "、".join(f"{lab}「{text[s:t]}」"
                                                        for s, t, lab in extra))
                lines.append("")
    out = REPORTS / f"cases_{args.dataset}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out}（{len(lines)} 行）")


if __name__ == "__main__":
    main()

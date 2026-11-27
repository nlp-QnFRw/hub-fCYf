# -*- coding: utf-8 -*-
"""
verify_processed.py — 逐行校验派生数据（课程纪律：转换产物必须可逐条反查）

检查项：
  1. 每行 text[start:end] == span.text（触发词 + 全部论元）；
  2. sent_id 在文件内唯一（评测键 id|sent_id 不碰撞，课程坑 #1）；
  3. 事件类型 ∈ event_schema.json，且角色 ∈ 该类型的 schema 角色集；
  4. train/dev/test 三个文件之间 sent_id 不重复；
  5. (id, text) 粒度的事件数统计与 build_stats.json 一致。
任何一项失败都打印明细并以非零退出（fail-fast，不静默吞）。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    schema = json.load(open(PROC / "event_schema.json", encoding="utf-8"))
    stats = json.load(open(PROC / "build_stats.json", encoding="utf-8"))

    global_ids = set()
    for split in ["train", "dev", "test"]:
        path = PROC / f"{split}_extract.jsonl"
        n_rows, n_span_err = 0, 0
        ids = set()
        sents = Counter()
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                r = json.loads(line)
                n_rows += 1
                if r["sent_id"] in ids:
                    fail(f"{path.name}:{ln} sent_id 重复: {r['sent_id']}")
                if r["sent_id"] in global_ids:
                    fail(f"{path.name}:{ln} sent_id 跨 split 重复: {r['sent_id']}")
                ids.add(r["sent_id"])
                global_ids.add(r["sent_id"])
                text = r["text"]
                ev = r["event"]
                if ev["type"] not in schema:
                    fail(f"{path.name}:{ln} 类型不在 schema: {ev['type']}")
                t = ev["trigger"]
                if text[t["start"]:t["end"]] != t["text"]:
                    n_span_err += 1
                    print(f"  [span_err] {path.name}:{ln} trigger {t}")
                for a in ev["args"]:
                    if text[a["start"]:a["end"]] != a["text"]:
                        n_span_err += 1
                        print(f"  [span_err] {path.name}:{ln} arg {a}")
                    if a["role"] not in schema[ev["type"]]:
                        fail(f"{path.name}:{ln} 角色 {a['role']} 不在 {ev['type']} 的 schema 中")
                sents[(r["id"], text)] += 1
        if n_rows != stats["splits"][split]["n_rows"]:
            fail(f"{split} 行数 {n_rows} != build_stats {stats['splits'][split]['n_rows']}")
        if n_span_err:
            fail(f"{split} 有 {n_span_err} 处 span 校验失败")
        print(f"[OK] {split}: rows={n_rows} unique_sents={len(sents)} span_errors=0")

    # dev_n200 与 eval_ids 一致性
    n200 = [json.loads(l) for l in open(PROC / "dev_n200.jsonl", encoding="utf-8")]
    eval_ids = json.load(open(PROC / "eval_ids.json", encoding="utf-8"))
    if len(eval_ids) != 200:
        fail(f"eval_ids 应为 200 句, 实际 {len(eval_ids)}")
    import hashlib
    keys = {(e["id"], e["text_md5"]) for e in eval_ids}
    n200_keys = {(r["id"], hashlib.md5(r["text"].encode()).hexdigest()) for r in n200}
    if keys != n200_keys:
        fail("dev_n200.jsonl 与 eval_ids.json 不一致")
    print(f"[OK] dev_n200: sents=200 rows={len(n200)}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

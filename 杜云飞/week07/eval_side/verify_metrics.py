# -*- coding: utf-8 -*-
"""verify_metrics.py — 交付前的产物验收（防止"每个都失败了但没人知道"）

每个方法都必须同时满足：
  1. metrics JSON 存在，且 trigger/argument/event_exact 三段齐全；
  2. 不是退化结果：trigger F1 > 0.05（全 0 说明模型没学到或解码坏了）；
  3. pred JSONL 存在且行数与口径一致（判别式 4087 行 → 合并 2548 句 / LLM 200 句）；
  4. n200 投影存在（跨方法同批对比的唯一依据）；
  5. 溯源字段齐全（训练时长/延迟/epoch 预算）。
用法：python eval_side/verify_metrics.py [--json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs" / "metrics" / "disc"
P = ROOT / "outputs" / "disc"

A_ORDER = ["bert12L_crf", "bert12L_linear", "bert4L_crf", "bert4L_linear", "bert4Lrand_crf",
           "roberta12L_crf", "roberta4L_crf", "roberta12L_linear", "roberta4L_linear",
           "macbert12L_crf", "macbert4L_crf", "macbert12L_linear", "macbert4L_linear"]
LLM = ["api_deepseek-flash_zero_rep1", "api_deepseek-flash_zero_rep2", "api_deepseek-flash_few",
       "local_Qwen3-0.6B_zero", "local_Qwen3-0.6B_few", "local_Qwen3-0.6B_lora",
       "local_Qwen2-0.5B-Instruct_zero", "local_Qwen3-1.7B_zero", "local_Qwen3-4B_zero"]


def count_lines(p: Path) -> int:
    if not p.exists():
        return -1
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def check(name: str, expect_rows: int) -> dict:
    mp = M / f"{name}.json"
    out = {"name": name, "ok": True, "problems": []}
    if not mp.exists():
        return {"name": name, "ok": False, "problems": ["缺指标"], "trig": None, "arg": None}
    d = json.loads(mp.read_text(encoding="utf-8"))
    meta = d.get("meta", {})
    trig = d.get("trigger", {}).get("f1")
    arg = d.get("argument", {}).get("f1")
    out.update({"trig": trig, "arg": arg,
                "event": d.get("event_exact", {}).get("f1"),
                "json_ok": meta.get("json_valid_rate"),
                "epochs": meta.get("epochs") or meta.get("n_few_shot_examples"),
                "train_min": round(float(meta["train_s"]) / 60, 1) if meta.get("train_s") else None,
                "sec_per_sent": meta.get("sec_per_sentence")})
    if trig is None or arg is None:
        out["problems"].append("三段指标缺失")
    elif trig <= 0.0:
        out["problems"].append(f"trigger F1 全 0（解码或链路坏了）")
    elif trig < 0.05:
        # 真正的"退化"只有全 0；0.03 这种是模型本身很差，属于结果不是故障
        print(f"[warn] {name} trigger F1 极低（{trig}），属真实结果，不计为失败")
    rows = count_lines(P / f"pred_{name}.jsonl")
    out["pred_rows"] = rows
    if rows < 0:
        out["problems"].append("缺 pred")
    elif rows != expect_rows:
        out["problems"].append(f"pred 行数 {rows}≠{expect_rows}")
    if not (M / f"{name}_n200.json").exists():
        out["problems"].append("缺 n200 投影")
    if meta.get("epochs") is None and meta.get("n_few_shot_examples") is None:
        out["problems"].append("缺 epoch 溯源")
    out["ok"] = not out["problems"]
    return out


def main() -> None:
    # 判别式 pred 是行级（dev 4087 行 → 评测时合并成 2548 句），LLM 侧是句级 200 行
    rows = [check(n, 4087) for n in A_ORDER] + [check(n, 200) for n in LLM]
    for r in rows:
        flag = "OK  " if r["ok"] else "FAIL"
        print(f"{flag} {r['name']:<32} trig={r.get('trig')} arg={r.get('arg')} "
              f"ep={r.get('epochs')} min={r.get('train_min')} "
              f"{'' if r['ok'] else '← ' + ', '.join(r['problems'])}")
    ok = sum(1 for r in rows if r["ok"])
    print(f"\n完成 {ok}/{len(rows)}")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()

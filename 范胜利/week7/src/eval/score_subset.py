# -*- coding: utf-8 -*-
"""
在「本次提供的 sample 数据」上，用同一套评测口径重算所有方法的指标。

两种来源分开标注，绝不混为一谈：
  A. 本次本地重训（`outputs/predictions/local/`，或新训练产生的 `src/outputs/disc/`）
     —— 在 sample 上真正训练/推理；
  B. 已保存的参考预测（`outputs/predictions/disc/`、`outputs/predictions/llm/`）
     —— 全量数据上训练好的模型输出，这里只把它投影到 sample 的同一批句子上重新评分，
     用于和 PPT 主表对得上。

统一子集：
  LLM 链路只评测固定 n=200 抽样（outputs/llm/eval_ids.json / gold_n200.jsonl），
  为了让「规则/BiLSTM/BERT/两段式/LLM」同批可比，这里默认取
  sample dev ∩ LLM n200 = 200 句，所有方法都在同一批 200 句上算指标。

用法：
  python eval/score_subset.py                      # 默认 sample + reference
  python eval/score_subset.py --all_dev            # 不投影，直接在 sample dev 300 上评
产物：
  outputs/metrics/sample_subset/<name>.json
  outputs/sample_subset.csv（不再写 .md；学习/结果类 md 统一放桌面合集）

对齐规则（本项目最容易错的地方）：
  DuEE-Fin 的 `id` 是**公告 id**，一篇公告的多句会共用同一个 id；评测键必须是
  `id + sent_id`。如果只按 id 对齐，后一句会覆盖前一句，gold-vs-gold 都只有 0.93。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/eval/ -> week7/
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "common"))

from evaluate_events import evaluate  # noqa: E402

DATA = ROOT / "data" / "sample"
REF = ROOT / "reference"
OUT = ROOT / "outputs"
PRED_DISC = OUT / "predictions" / "disc"
PRED_LLM = OUT / "predictions" / "llm"
LOCAL_PRED = OUT / "predictions" / "local"

# (name, 显示名, 预测文件, 来源说明)
METHODS = [
    ("rule", "L0 规则（词典+正则）", PRED_DISC / "pred_rule.jsonl", "reference"),
    ("bilstm_crf", "L1 字符级 BiLSTM-CRF", PRED_DISC / "pred_bilstm_crf.jsonl", "reference"),
    ("bert_linear_256", "L2 BERT+Linear", PRED_DISC / "pred_bert_linear_256.jsonl", "reference"),
    ("bert_crf_256", "L2 BERT+CRF", PRED_DISC / "pred_bert_crf_256.jsonl", "reference"),
    ("bert_crf_256_hard", "L2 BERT+CRF+解码期硬约束", PRED_DISC / "pred_bert_crf_256_hard.jsonl", "reference"),
    ("pipeline", "L2 两段式（检测→类型→论元）", PRED_DISC / "pred_pipeline.jsonl", "reference"),
    ("local_zero_n200", "L3 本地 Qwen3-0.6B zero-shot", PRED_LLM / "pred_local_zero_n60.jsonl", "reference"),
    ("api_zero_n200", "L3 API zero-shot", PRED_LLM / "pred_api_zero_n200.jsonl", "reference"),
    ("api_few_n200", "L3 API few-shot", PRED_LLM / "pred_api_few_n200.jsonl", "reference"),
    ("sft_lora_n200", "L4 LoRA SFT", PRED_LLM / "pred_sft_lora_n200.jsonl", "reference"),
]

# 本次本地重训产物（存在才纳入）
LOCAL = [
    ("rule_sample", "L0 规则（sample 重训）"),
    ("bilstm_crf_sample", "L1 BiLSTM-CRF（sample 重训）"),
    ("bert_linear_sample", "L2 BERT+Linear（sample 重训）"),
    ("bert_crf_sample", "L2 BERT+CRF（sample 重训）"),
    ("bert_crf_sample_hard", "L2 BERT+CRF+硬约束（sample 重训）"),
]


def load_jsonl(path):
    """读 JSONL；空行跳过，保留原始顺序。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key(row):
    """统一评测键：优先 id|sent_id；没有 sent_id 的旧数据才退化为 id。"""
    return f'{row.get("id", "")}|{row.get("sent_id", "")}' if row.get("sent_id") else str(row.get("id"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(DATA / "dev_extract.jsonl"))
    ap.add_argument("--keys", default=str(REF / "llm" / "gold_n200.jsonl"),
                    help="限定评测子集的 gold 文件（默认 LLM 固定 n=200 抽样）")
    ap.add_argument("--all_dev", action="store_true", help="不投影，评 sample dev 全部 300 句")
    args = ap.parse_args()

    # gold_all 是 sample dev 300；默认再用 LLM 的固定 n=200 抽样做交集，
    # 保证规则/BiLSTM/BERT/两段式/LLM 在同一批句子上比较。
    gold_all = load_jsonl(Path(args.gold))
    if args.all_dev:
        gold = gold_all
        subset = "sample_dev_all"
    else:
        keys_path = Path(args.keys)
        keys = {key(r) for r in load_jsonl(keys_path)} if keys_path.exists() else None
        gold = [r for r in gold_all if keys is None or key(r) in keys]
        subset = "sample_dev_x_llm_n200"
    gold_keys = {key(r) for r in gold}

    out_dir = OUT / "metrics" / "sample_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(name, display, pred_path, source):
        """评测一个预测文件，并把一行结果追加到 `rows`。

        参数：
          name/display：机器名与展示名；
          pred_path：预测 JSONL，必须带 id+sent_id；
          source：`reference`（全量模型复评）或 `local_sample`（本次 sample 重训）。

        为什么要做“覆盖率回退”：
          LLM 的本地 0.6B 只跑了 n=60，而统一子集是 200 句。如果直接把缺的 140 句
          当“全漏”，n=60 的 F1 会被虚假拉低。因此覆盖率 <80% 时，只在该方法
          实际覆盖的句子上评，并在结果里保留真实 n。
        """
        if not pred_path.exists():
            return
        preds = load_jsonl(pred_path)
        preds = [p for p in preds if key(p) in gold_keys]
        if not preds:
            return
        # 小样本方法（如本地 0.6B 只跑 n=60）不能把未预测的 140 句当“全漏”，
        # 否则会把 n=60 的分数虚假拉低；覆盖率 <80% 时退回到它自己的覆盖子集。
        if len(preds) < 0.8 * len(gold):
            covered = {key(p) for p in preds}
            gold_eval = [g for g in gold if key(g) in covered]
        else:
            gold_eval = gold
        res = evaluate(gold_eval, preds, verbose=False)
        res["method"] = name
        res["display_name"] = display
        res["source"] = source
        # 本地重训若完全没收敛（全 O），不混进方法对比表；在报告 4.3 单独说明
        if source == "local_sample" and res["trigger"]["f1"] == 0 and res["argument"]["f1"] == 0:
            print(f"[skip] {display} 本地重训未收敛（全 O），不纳入同批对比")
            return
        res["subset"] = subset
        (out_dir / f"{name}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({
            "method": name, "display_name": display, "source": source,
            "n": res["num_samples"], "covered": len(preds),
            "trigger_f1": res["trigger"]["f1"],
            "argument_f1": res["argument"]["f1"],
            "event_f1": res["event_exact"]["f1"],
            "type_macro_f1": res.get("type_macro_f1"),
        })

    for name, display, path, source in METHODS:
        add(name, display, path, source)
    for name, display in LOCAL:
        # 新训练优先用 src/outputs/disc；交付包里没有训练产物时退回 predictions/local
        fresh = SRC / "outputs" / "disc" / f"pred_{name}.jsonl"
        add(name, display, fresh if fresh.exists() else LOCAL_PRED / f"pred_{name}.jsonl",
            "local_sample")

    rows.sort(key=lambda r: r["argument_f1"], reverse=True)
    import csv
    csv_path = OUT / "sample_subset.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
        w.writeheader()
        w.writerows(rows)

    # markdown
    L = [f"> **本文档主题：** 10 种方法在 {subset} 上的同口径复评结果（n={len(gold)}）\n",
         "> **初学者导读：** 这是结果表，不用逐行读；先看方法名、n、论元 F1、事件 F1 四列。\n"
         "> **怎么用：** 选型看论元/事件 F1；写报告直接引用这张表；n=60 的行不能和 n=200 的行直接比。\n"
         "> **对应章节：** 初学者教程第 4 章「评测」\n"
         "> **文档级别：** 📊 结果",
         f"# 同口径对比（{subset}，n={len(gold)}）\n",
         "| 方法 | 来源 | n | 触发词 F1 | 论元 F1 | 事件级 F1 | 类型 macro F1 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        src = "本地重训" if r["source"] == "local_sample" else "已保存参考预测（全量训练，sample 子集复评）"
        L.append(f"| {r['display_name']} | {src} | {r['n']} | {r['trigger_f1']:.4f} | "
                 f"{r['argument_f1']:.4f} | {r['event_f1']:.4f} | "
                 f"{(r['type_macro_f1'] or 0):.4f} |")
    # 不写 .md：结果类 Markdown 统一放在桌面 week7_学习文档合集.md，不进 GitHub。
    # 终端仍打印同一张表，CSV 仍是机器可读的权威结果。

    print("\n".join(L))
    print(f"\n已写入 {csv_path}")


if __name__ == "__main__":
    main()

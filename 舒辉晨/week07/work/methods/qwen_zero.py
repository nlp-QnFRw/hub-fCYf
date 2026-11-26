"""换模型：本地 Qwen3-0.6B zero/few-shot 做事件抽取（生成式路线）。

复用 week7 的 prompt 模板与解析器（event_prompt.py），保证和 week7 的 LLM 链路同口径；
只改「跑在哪批数据、产物写在哪」，不另写 prompt 或解析逻辑。
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (DATA_ROOT, LLM_GEN_BATCH, LLM_MAX_NEW_TOKENS, LLM_N,  # noqa: E402
                    MODEL_DIRS, OUT, PRETRAIN_ROOTS, SEED, WEEK7_DATA)
from common.bridge import (  # noqa: E402
    batch_messages, build_messages, generate_batch, load_causal_lm,
    load_jsonl, make_pred_row, parse_model_output, resolve_model_path,
    resolve_torch_dtype,
)
from common.eval_local import run_eval, sample_rows  # noqa: E402


def data_dir(variant: str) -> Path:
    return WEEK7_DATA if variant == "sent" else DATA_ROOT / variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sent", "heldout", "doc"], default="sent")
    ap.add_argument("--mode", choices=["zero", "few"], default="zero")
    ap.add_argument("--n", type=int, default=LLM_N)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_length", type=int, default=2048,
                    help="prompt+生成总长度上限；few-shot prompt 约 1500 token，1280 会截断 system schema")
    args = ap.parse_args()

    ddir = data_dir(args.variant)
    schema_path = ddir / "event_schema.json"
    raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # event_prompt 用的是生成式链路的 schema 包装：{"event_types": ..., "enums": ...}
    schema = {"event_types": {k: v for k, v in raw_schema.items() if not k.startswith("__")},
              "enums": raw_schema.get("__enums__", {})}
    all_rows = load_jsonl(str(ddir / "dev_extract.jsonl"))
    rows = sample_rows(all_rows, args.n, SEED)
    name = args.tag or f"qwen_{args.mode}_{args.variant}_n{len(rows)}"

    ids_path = OUT / f"eval_ids_{args.variant}_n{len(rows)}.json"
    ids_path.write_text(json.dumps([r["id"] for r in rows], ensure_ascii=False, indent=1),
                        encoding="utf-8")
    # 生成式只跑 60 条：gold 必须同步缩到这 60 条，否则 870 条缺预测会被算成 FN，
    # 得到的是「没跑」的数字，不是模型效果（week7 的 n=200 投影就是这个道理）。
    gold_path = OUT / f"gold_{args.variant}_n{len(rows)}.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model_path = resolve_model_path(MODEL_DIRS["qwen"], extra_roots=PRETRAIN_ROOTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[local] {model_path} | device={device} | n={len(rows)} mode={args.mode}", flush=True)
    model, tokenizer = load_causal_lm(str(model_path), device,
                                      resolve_torch_dtype(device, "auto"))
    msgs = [build_messages(r["text"], schema, few_shot=(args.mode == "few")) for r in rows]
    t0 = time.time()
    outs = generate_batch(model, tokenizer, msgs, device,
                          max_new_tokens=LLM_MAX_NEW_TOKENS, batch_size=LLM_GEN_BATCH,
                          max_length=args.max_length)
    gen_s = time.time() - t0

    preds, stats = [], Counter()
    for row, out in zip(rows, outs):
        evs, info = parse_model_output(out["text"], row["text"], schema)
        for k, v in info.items():
            if isinstance(v, bool) or isinstance(v, int):
                stats[k] += int(bool(v))
        preds.append(make_pred_row(row, evs))
    with open(OUT / "logs" / f"raw_{name}.jsonl", "w", encoding="utf-8") as f:
        for row, out in zip(rows, outs):
            f.write(json.dumps({"id": row["id"], "raw": out["text"],
                                "latency_s": out.get("latency_s", 0)}, ensure_ascii=False) + "\n")

    meta = {
        "backend": "local", "model": "Qwen3-0.6B", "model_path": str(model_path),
        "mode": args.mode, "n": len(rows), "seed": SEED, "gen_s": round(gen_s, 1),
        "prompt_max_length": args.max_length,
        "ms_per_sentence": round(gen_s * 1000 / len(rows), 1),
        "parse_stats": dict(stats), "eval_ids": str(ids_path), "gold_subset": str(gold_path),
    }
    res = run_eval(name, preds, gold_path, meta=meta)
    print(f"{name}: argument F1={res['argument']['f1']:.4f} "
          f"event F1={res['event_exact']['f1']:.4f} | json_ok={stats.get('json_ok', 0)}/{len(rows)}",
          flush=True)


if __name__ == "__main__":
    main()

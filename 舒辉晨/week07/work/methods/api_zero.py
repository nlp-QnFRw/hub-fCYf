"""换 API：OpenAI 兼容的云端大模型 zero/few-shot 抽取。

用标准库 urllib 发请求，不依赖 openai 包（py312 环境没装）。
**没有 key 就明确记 NOT_RUN**，不写估算值；设置 LLM_API_KEY / DEEPSEEK_API_KEY 后重跑即可。
可选环境变量：LLM_BASE_URL（默认 DeepSeek）、LLM_MODEL（默认 deepseek-v4-flash）。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_ROOT, LLM_N, OUT, SEED, WEEK7_DATA  # noqa: E402
from common.bridge import build_messages, load_jsonl, make_pred_row, parse_model_output  # noqa: E402
from common.eval_local import record_skip, run_eval, sample_rows  # noqa: E402

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def call_api(base_url: str, key: str, model: str, messages: list, max_tokens: int) -> dict:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens, "enable_thinking": False}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage") or {}
    return {"text": choice, "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)}


def data_dir(variant: str) -> Path:
    return WEEK7_DATA if variant == "sent" else DATA_ROOT / variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sent", "heldout", "doc"], default="sent")
    ap.add_argument("--mode", choices=["zero", "few"], default="zero")
    ap.add_argument("--n", type=int, default=LLM_N)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    name = args.tag or f"api_{args.mode}_{args.variant}_n{args.n}"
    key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        record_skip(name, "未提供 LLM_API_KEY / DEEPSEEK_API_KEY；本路未运行，无实测数字")
        return

    ddir = data_dir(args.variant)
    raw_schema = json.loads((ddir / "event_schema.json").read_text(encoding="utf-8"))
    schema = {"event_types": {k: v for k, v in raw_schema.items() if not k.startswith("__")},
              "enums": raw_schema.get("__enums__", {})}
    rows = sample_rows(load_jsonl(str(ddir / "dev_extract.jsonl")), args.n, SEED)
    gold_path = OUT / f"gold_{args.variant}_n{len(rows)}.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    print(f"[api] {base_url} / {model} / n={len(rows)}", flush=True)
    outs, stats, errors = [], Counter(), []
    t0 = time.time()
    for i, row in enumerate(rows):
        msgs = build_messages(row["text"], schema, few_shot=(args.mode == "few"))
        try:
            outs.append(call_api(base_url, key, model, msgs, 512))
        except (urllib.error.URLError, KeyError, ValueError) as e:
            errors.append({"id": row["id"], "error": str(e)[:200]})
            outs.append({"text": "", "prompt_tokens": 0, "completion_tokens": 0})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    elapsed = time.time() - t0
    preds = []
    for row, out in zip(rows, outs):
        evs, info = parse_model_output(out["text"], row["text"], schema)
        for k, v in info.items():
            if isinstance(v, (bool, int)):
                stats[k] += int(bool(v))
        preds.append(make_pred_row(row, evs))
    meta = {"backend": "api", "model": model, "base_url": base_url, "mode": args.mode,
            "n": len(rows), "elapsed_s": round(elapsed, 1), "errors": errors,
            "parse_stats": dict(stats)}
    res = run_eval(name, preds, gold_path, meta=meta)
    print(f"{name}: argument F1={res['argument']['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()

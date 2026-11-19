"""
方法五：本地大模型零样本（L4，不训练）

用 Qwen3-0.6B 直接生成类别名。它回答的是「手上一条标注都没有时能做到什么」。

两个必须写对的细节：
  1. `enable_thinking=False` 要在 chat template 阶段传，否则模型先吐 <think> 段落，
     输出格式全乱、解析失败率飙升；
  2. 解析用「包含 + 最长匹配」而不是严格相等——模型常多吐一个句号或"类别："前缀，
     严格相等会把可用的答案也判成失败。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits, subset
from common.labels import CLASSES
from common.metrics import classification_report, pred_distribution, to_markdown_table
from common.model_paths import qwen_path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM = ("你是一个中文新闻分类助手。请把给定的新闻内容归到以下类别之一，"
          "只输出类别名称，不要输出任何解释。可选类别：" + "、".join(CLASSES))


def parse_label(text: str) -> int:
    text = text.strip()
    hits = [c for c in CLASSES if c in text]
    if not hits:
        return -1
    return CLASSES.index(max(hits, key=len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--out", default="outputs/zero_shot_pred.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = qwen_path()
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    _, val, meta = load_splits()
    samples = subset(val, args.n)
    preds, raws = [], []
    t0 = time.time()
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["text"][:600]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws.append(raw)
        preds.append(parse_label(raw))
        if i % 25 == 0:
            print(f"  {i}/{len(samples)} {time.time()-t0:.0f}s", flush=True)

    y_true = [d["label"] for d in samples]
    rep = classification_report(y_true, preds)
    rep.update({"method": "qwen3_zero_shot",
                "sec_per_item": round((time.time() - t0) / len(samples), 2),
                "pred_distribution": pred_distribution(preds), "n_eval": len(samples)})

    out = ROOT / args.out
    out.write_text(json.dumps({"meta": {**meta, "n_eval": len(samples)}, "report": rep,
                               "preds": [{"text": d["text"][:60], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown_table({"Qwen3-0.6B zero-shot": rep}))
    print(f"覆盖类别 {rep['pred_distribution']['used_classes']}/{len(CLASSES)} | "
          f"{rep['sec_per_item']}s/条 → {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
extract_local.py — B 组：离线大模型 zero/few-shot 抽取（Qwen 家族批量贪心生成）

本机 8GB 显存适配：
  - 0.5B/0.6B/1.7B：bf16 全量上卡；4B：bf16 + device_map=auto（剩余层卸载到 32GB 内存，
    牺牲速度换能跑——这正是"离线模型选型必须看本机配置"的实测素材）；
  - 批量解码 + 左 padding（课程坑 #27：Windows WDDM 下逐 token 解码每步 ~200 个 CUDA kernel，
    批量能把 kernel 的活变多，实测 4~6 倍提速）；
  - Qwen3 系统一 enable_thinking=False（坑 #5：思考链会污染 JSON）。

用法：
  python llm/extract_local.py --model Qwen3-1.7B [--few_shot] [--n 200]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "llm"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from model_paths import resolve_model_path  # noqa: E402
from prompt import build_messages  # noqa: E402
from extract_common import load_dev_sentences, load_schema, run_extract  # noqa: E402


def render(tokenizer, messages, disable_thinking: bool) -> str:
    """Qwen3 用 enable_thinking=False 关思考链；Qwen2 无此参数，直接渲染。"""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("chat_template_kwargs", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="模型目录名（D:\\AI_learn\\Model 下）：Qwen2-0.5B-Instruct / Qwen3-0.6B / Qwen3-1.7B / Qwen3-4B")
    ap.add_argument("--few_shot", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--device_map", default="auto", help="4B 建议 auto（内存兜底）；小模型 cuda")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--adapter", default="",
                    help="LoRA 适配器目录（sft_lora.py 产出）；空=基座 zero/few-shot")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    schema = load_schema()
    sentences = load_dev_sentences(args.n)
    model_path = resolve_model_path(args.model)

    tok = AutoTokenizer.from_pretrained(str(model_path))
    disable_thinking = "Qwen3" in args.model
    # 左 padding（decoder-only 批量生成必须；右 padding 会让模型从 pad 位置续写）
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    if args.model == "Qwen3-4B" or args.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16).to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[adapter] 已加载 LoRA 适配器：{args.adapter}", flush=True)
    model.eval()

    prompt_tokens = completion_tokens = 0

    @torch.no_grad()
    def generate_fn(batch_texts: list[str]) -> list[str]:
        nonlocal prompt_tokens, completion_tokens
        prompts = [render(tok, build_messages(t, schema, args.few_shot), disable_thinking)
                   for t in batch_texts]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        prompt_tokens += int(enc["input_ids"].numel())
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        comp = out[:, enc["input_ids"].shape[1]:]
        completion_tokens += int(comp.numel())
        return tok.batch_decode(comp, skip_special_tokens=True)

    tag = "_lora" if args.adapter else ("_few" if args.few_shot else "_zero")
    run_name = args.run_name or f"local_{args.model}{tag}"
    meta = {
        "display_name": (f"本地 {args.model} LoRA-SFT" if args.adapter else
                         f"本地 {args.model} {'few-shot' if args.few_shot else 'zero-shot'}"),
        "adapter": args.adapter or None,
        "model": args.model, "model_path": str(model_path),
        "few_shot": bool(args.few_shot),
        "n_few_shot_examples": 9 if args.few_shot else 0,
        "gen_batch": args.gen_batch, "max_new_tokens": args.max_new_tokens,
        "dtype": "bf16", "device_map": args.device_map,
        "disable_thinking": disable_thinking,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    res = run_extract(schema, sentences, generate_fn, run_name, meta,
                      batch_size=args.gen_batch)
    meta["prompt_tokens"] = prompt_tokens
    meta["completion_tokens"] = completion_tokens
    # 回写 token 统计（evaluate_predictions 已落盘，这里追加）
    mf = ROOT / "outputs" / "metrics" / "disc" / f"{run_name}.json"
    data = json.loads(mf.read_text(encoding="utf-8"))
    data["meta"].update({k: meta[k] for k in ("prompt_tokens", "completion_tokens")})
    mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"token 合计 prompt={prompt_tokens} completion={completion_tokens}")


if __name__ == "__main__":
    main()

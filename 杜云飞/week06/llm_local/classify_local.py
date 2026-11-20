#region 0. 模块说明
"""
L3-a：本地小模型 zero/few-shot 分类（Qwen2-0.5B-Instruct / Qwen3-0.6B）。

它在报告里的角色是**诊断**，不是"参赛"：
  用 0.5B 的通用模型直接做 14 分类，准确率大概率明显低于微调后的 BERT。
  但我们真正要回答的是——"生成式路线在**不训练**的情况下能到哪一步""标签空间不闭合
  会浪费多少条""few-shot 到底是帮忙还是添乱"。这些问题的答案，决定了 L4（LoRA）值不值得做。

工程要点：
  1. `apply_chat_template(..., tokenize=False)` 显式关掉 tokenize（transformers 5.x 里
     不传会返回 dict，后面 `[:max_len]` 直接炸）；
  2. 生成必须 `padding_side="left"`，否则 PAD 会挤在答案前面，解码出来的前缀全是 PAD；
  3. `do_sample=False, num_beams=1` 保证可复现；
  4. 解码时只取 `input_len:` 之后的部分，避免把 prompt 里的类名误当答案；
  5. Qwen3 必须 `enable_thinking=False`（否则输出里带思考段，解析全失败）；
  6. 三档解析口径，主表用"全样本计入分母"，失败率单列。

使用方式：
  python llm_local/classify_local.py --shots 0 --variant qwen2-0shot
  python llm_local/classify_local.py --model qwen3-0.6b --shots 2 --thinking off
  python llm_local/classify_local.py --demo
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import append_index, new_run_id, write_meta, write_preds, write_train_log  # noqa: E402
from common.data import load_split  # noqa: E402
from common.paths import has_local, model_path  # noqa: E402
from common.prompt import build_messages, parse_label  # noqa: E402
from common.seed import set_seed  # noqa: E402

#endregion

#region 2. 全局配置
BATCH_SIZE: int = 16
MAX_NEW_TOKENS: int = 8  # 类名最长 2 个字，8 个 token 足够；给大了会显著变慢
#endregion

#region 3. 函数：加载模型
def load_model(alias: str, thinking: str = "off"):
    """加载 tokenizer 与模型。

    参数：
      alias    : qwen2-0.5b / qwen3-0.6b
      thinking : 仅 Qwen3 有意义，off 时传 enable_thinking=False
    返回：
      (tokenizer, model, device)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not has_local(alias):
        raise SystemExit(f"本地没有 {alias}：python scripts/04_fetch_backbones.py --only {alias}")
    path = model_path(alias)
    tk = AutoTokenizer.from_pretrained(path)
    # Qwen2 的 pad_token 是 <|endoftext|>，与 eos(<|im_end|>) 不同；生成时用 pad 兜底批量对齐
    if tk.pad_token_id is None:
        tk.pad_token_id = tk.eos_token_id
    tk.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                 attn_implementation="sdpa")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"[模型] {alias} on {device} | pad={tk.pad_token!r} eos={tk.eos_token!r}")
    return tk, model, device


def render_prompt(tk, rec: dict, shots: int, max_chars: int, truncation: str,
                  thinking: str, shot_style: str = "turn") -> str:
    """把 messages 渲染成纯文本 prompt。

    Qwen3 需要在渲染时关闭思考：`chat_template_kwargs` 是 transformers 5.x 的传参方式，
    老版本不支持就退回普通渲染（此时 Qwen3 的输出会带思考段，解析失败率会上升——
    这个现象本身就是报告里"模板不一致导致指标腰斩"的证据）。
    """
    msgs = build_messages(rec, shots=shots, max_chars=max_chars, truncation=truncation,
                          shot_style=shot_style)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if thinking == "off":
        try:
            return tk.apply_chat_template(msgs, chat_template_kwargs={"enable_thinking": False}, **kwargs)
        except (TypeError, ValueError):
            pass
    return tk.apply_chat_template(msgs, **kwargs)
#endregion

#region 4. 函数：批量生成
@torch.inference_mode()
def generate_batch(tk, model, prompts: list[str], device) -> tuple[list[str], float]:
    """批量生成，返回 (解码后的新文本列表, 总耗时秒)。"""
    enc = tk(prompts, return_tensors="pt", padding=True, truncation=True,
             max_length=1024).to(device)
    t0 = time.time()
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tk.pad_token_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    new_tokens = out[:, enc["input_ids"].shape[1]:]  # 只取新生成部分
    texts = tk.batch_decode(new_tokens, skip_special_tokens=True)
    return texts, dt
#endregion

#region 5. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="本地 LLM zero/few-shot 分类")
    ap.add_argument("--model", default="qwen2-0.5b")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--shot_style", default="turn", choices=["turn", "inline"],
                    help="turn=多轮示例；inline=示例压进同一条 user 消息（小模型更稳）")
    ap.add_argument("--splits", nargs="+", default=["test_bal", "test_nat_10k"])
    ap.add_argument("--n", type=int, default=None, help="限制每个 split 的条数")
    ap.add_argument("--max_chars", type=int, default=600)
    ap.add_argument("--truncation", default="head_tail")
    ap.add_argument("--thinking", default="off", choices=["off", "on"], help="仅 Qwen3 生效")
    ap.add_argument("--parse", default="lenient", choices=["strict", "lenient"])
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    if args.demo:
        args.n, args.splits = 20, ["test_bal"]
        print("[Demo 模式] 只跑 20 条，产物不可用于对比")

    variant = args.variant or f"{args.model}-{args.shots}shot-{args.max_chars}c"
    run_id = args.run_id or new_run_id("t1" if args.shots in (0,) else "t3", "llm-local", variant)
    tk, model, device = load_model(args.model, args.thinking)

    t_all = time.time()
    for split in args.splits:
        recs = load_split(split, limit=args.n)
        preds: list[dict] = []
        n_fail = 0
        t_split = time.time()
        for i in range(0, len(recs), args.batch_size):
            chunk = recs[i: i + args.batch_size]
            prompts = [render_prompt(tk, r, args.shots, args.max_chars, args.truncation,
                                     args.thinking, args.shot_style) for r in chunk]
            texts, dt = generate_batch(tk, model, prompts, device)
            per_ms = dt / max(1, len(chunk)) * 1000
            for r, txt in zip(chunk, texts):
                pred, status = parse_label(txt, mode=args.parse)
                if pred is None:
                    n_fail += 1
                preds.append({"id": r["id"], "gold": r["label"], "pred": pred,
                              "raw": txt.replace("\n", " "), "status": status,
                              "latency_ms": per_ms})
            done = min(i + args.batch_size, len(recs))
            if done % (args.batch_size * 8) == 0 or done == len(recs):
                acc = sum(1 for p in preds if p["pred"] == p["gold"]) / max(1, len(preds))
                print(f"  {split} {done}/{len(recs)}  acc(含失败)={acc:.4f}  "
                      f"失败 {n_fail}  {done / max(1e-6, time.time() - t_split):.1f} 条/s", flush=True)

        write_preds(run_id, split, preds)
        acc = sum(1 for p in preds if p["pred"] == p["gold"]) / max(1, len(preds))
        print(f"  {split}: {len(preds)} 条  acc={acc:.4f}  解析失败 {n_fail} "
              f"({n_fail / max(1, len(preds)) * 100:.1f}%)")

    if args.demo:
        ps = load_split("test_bal", limit=5)
        for p, r in zip(load_split("test_bal", limit=5), ps):
            pass
        print("[Demo 模式] 20 条演示结束，产物已落盘但规模极小")

    write_meta(run_id, tier="t1" if args.shots == 0 else "t3", method="llm-local",
               backbone=args.model, n_train=0, seed=42,
               hyperparams={"shots": args.shots, "shot_style": args.shot_style,
                            "max_chars": args.max_chars,
                            "truncation": args.truncation, "parse": args.parse,
                            "thinking": args.thinking, "batch_size": args.batch_size,
                            "max_new_tokens": MAX_NEW_TOKENS},
               elapsed_s=round(time.time() - t_all, 1), device=str(device),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2)
               if torch.cuda.is_available() else 0.0, cost_usd=0.0)
    write_train_log(run_id, [])
    append_index(run_id, {"method": "llm-local", "model": args.model, "shots": args.shots})

    print(f"\nrun_id = {run_id}")
    print(f"下一步：python eval/evaluate.py --run {run_id}")


if __name__ == "__main__":
    main()
#endregion

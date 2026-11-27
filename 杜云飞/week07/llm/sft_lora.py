# -*- coding: utf-8 -*-
"""sft_lora.py — B 组补强：Qwen3-0.6B LoRA 指令微调（FewFC train → 本地小模型能不能救）

为什么必须补这一条：
  Qwen3-0.6B zero-shot 的 trigger F1 只有 0.087，失败模式是"schema 没进脑子"（36.8% 编造
  schema 外类型），不是边界偏差。这正是"本地小模型 + 好 prompt 不行，那就微调"的选型分叉点：
  如果 SFT 后能到 0.4+，本地部署就有意义；如果还是 0.1，就老老实实走 API/判别式。
  这条数字是简历上"我能把开源小模型调到可用"的直接证据。

数据构造的三个纪律：
  1. **按句聚合**：train_extract 是"行=一个事件"，同一句多个事件会变成"同输入多目标"互相
     打架。这里按 id 聚合，目标是一次输出该句全部事件（顺带教模型 30.7% 的多事件句）。
  2. **目标格式 = 推理时的解析契约**（prompt.py 的 {"events":[{type,trigger.text,arguments}]}），
     训练/推理同一套 schema，评测器不用改一行。
  3. **只对 assistant 段算 loss**：system+user 的 token 全部 -100，否则模型会去学复述 prompt。

用法：
  python llm/sft_lora.py                       # 全量 3552 句，2 epoch
  python llm/sft_lora.py --limit 40 --epochs 1 # 冒烟
  python llm/extract_local.py --model Qwen3-0.6B --adapter outputs/sft/qwen3_0.6b_lora \
         --run_name local_Qwen3-0.6B_lora      # 评测（走同一条 n200 口径）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "llm"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from model_paths import resolve_model_path  # noqa: E402
from prompt import build_system_prompt  # noqa: E402
from log_utils import TrainLogger  # noqa: E402

PROC = ROOT / "data" / "processed"


def render(tok, messages) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tok.apply_chat_template(messages, chat_template_kwargs={"enable_thinking": False}, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def target_json(events: list[dict]) -> str:
    out = {"events": [{"type": e["type"],
                       "trigger": {"text": e["trigger"]["text"]},
                       "arguments": [{"role": a["role"], "argument": a["text"]}
                                     for a in (e.get("args") or [])]}
                      for e in events]}
    return json.dumps(out, ensure_ascii=False)


def build_samples(schema: dict) -> list[dict]:
    """train_extract → 按句聚合的 (prompt_text, target_text) 样本。"""
    by_sent: dict[str, dict] = {}
    with open(PROC / "train_extract.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ev = r.get("event")
            if not ev:
                continue
            g = by_sent.setdefault(r["id"], {"id": r["id"], "text": r["text"], "events": []})
            g["events"].append(ev)
    system = build_system_prompt(schema)
    samples = []
    for g in by_sent.values():
        msg = [{"role": "system", "content": system},
               {"role": "user", "content": f"句子：{g['text']}"}]
        samples.append({"id": g["id"], "prompt": msg, "target": target_json(g["events"]),
                        "n_events": len(g["events"])})
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_dir", default="outputs/sft/qwen3_0.6b_lora")
    ap.add_argument("--run_name", default="sft_qwen3_0.6b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    schema = json.loads((PROC / "event_schema.json").read_text(encoding="utf-8"))
    samples = build_samples(schema)
    if args.limit:
        samples = samples[:args.limit]
    model_path = resolve_model_path(args.model)
    print(f"[data] {len(samples)} 句（含多事件 {sum(1 for s in samples if s['n_events'] > 1)} 句）", flush=True)

    tok = AutoTokenizer.from_pretrained(str(model_path))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    eos = tok.eos_token or ""

    # 预先 tokenize（1536 token 的样本 ×3552 只有几十 MB，省掉训练时的重复渲染）
    feats, n_trunc = [], 0
    for s in samples:
        p_ids = tok(render(tok, s["prompt"]), add_special_tokens=False)["input_ids"]
        t_ids = tok(s["target"] + eos, add_special_tokens=False)["input_ids"]
        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        if len(ids) > args.max_len:
            n_trunc += 1
            ids, labels = ids[:args.max_len], labels[:args.max_len]
        feats.append((ids, labels))
    lens = sorted(len(x[0]) for x in feats)
    print(f"[data] token 长度 中位 {lens[len(lens)//2]} / p95 {lens[int(len(lens)*0.95)]} / "
          f"max {lens[-1]}（截断 {n_trunc} 条）", flush=True)

    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] 可训练 {trainable/1e6:.2f}M / 总 {total/1e6:.1f}M ({trainable/total:.2%})", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(feats) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(total_steps * 0.03))

    def lr_at(step: int) -> float:
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return args.lr * 0.5 * (1 + __import__("math").cos(__import__("math").pi * p))

    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=int(trainable),
                         notes=f"FewFC SFT LoRA r={args.lora_r} alpha={args.lora_alpha} "
                               f"on {args.model}（按句聚合 {len(feats)} 句）")
    order = list(range(len(feats)))
    rng = __import__("random").Random(args.seed)
    t0 = time.perf_counter()
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        ep_loss, ep_n = 0.0, 0
        opt.zero_grad()
        for i in range(0, len(order), args.batch_size):
            chunk = order[i:i + args.batch_size]
            m = max(len(feats[j][0]) for j in chunk)
            # pin_memory + non_blocking：pageable 的 .cuda() 是隐式同步，本机驱动状态下
            # 一次同步能放大到秒级，实测把 SFT 拖成 27s/步（详见 disc/train.py stage_dataset 注释）
            ids = torch.full((len(chunk), m), tok.pad_token_id, dtype=torch.long, pin_memory=True)
            lab = torch.full((len(chunk), m), -100, dtype=torch.long, pin_memory=True)
            am = torch.zeros((len(chunk), m), dtype=torch.long, pin_memory=True)
            for k, j in enumerate(chunk):
                L = len(feats[j][0])
                ids[k, :L] = torch.tensor(feats[j][0])
                lab[k, :L] = torch.tensor(feats[j][1])
                am[k, :L] = 1
            ids = ids.cuda(non_blocking=True)
            lab = lab.cuda(non_blocking=True)
            am = am.cuda(non_blocking=True)
            out = model(input_ids=ids, attention_mask=am, labels=lab)
            loss = out.loss
            (loss / args.grad_accum).backward()
            ep_loss += float(loss.detach())
            ep_n += 1
            if ep_n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(global_step)
                opt.step()
                opt.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    logger.log_step(epoch=epoch, step=global_step,
                                    loss=ep_loss / ep_n, lr=lr_at(global_step))
                    print(f"  epoch {epoch} step {global_step}/{total_steps} "
                          f"loss={ep_loss / ep_n:.4f} "
                          f"({(time.perf_counter()-t0)/max(global_step,1):.2f}s/step)", flush=True)
        print(f"epoch {epoch} 完成 loss={ep_loss / max(ep_n, 1):.4f} "
              f"用时 {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        logger.log_epoch(epoch=epoch, train_loss=ep_loss / max(ep_n, 1),
                         elapsed=time.perf_counter() - t0)

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tok.save_pretrained(str(save_dir))
    logger.finalize(total_s=time.perf_counter() - t0, epochs=args.epochs,
                    n_samples=len(feats), total_steps=global_step,
                    trainable_params=int(trainable), n_truncated=n_trunc,
                    save_dir=str(save_dir))
    print(f"适配器已保存 → {save_dir}", flush=True)


if __name__ == "__main__":
    main()

#region 0. 导入库
import argparse
import math
import random
import sys

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from common import (ART_DIR, DATA_DIR, OUT_DIR, SRC_VAL, ANSWER_END, EOS_IDS, IM_END, PAD_ID,
                    build_prompt, char_f1, count_question_marks, get_tokenizer, is_degenerate,
                    load_base, read_jsonl, rouge_l_char, save_json)
#endregion

#region 1. 常量
N_SAMPLES = 10          # 报告里展示多少组对照
SEED = 42               # 固定种子，保证报告可复现
#endregion

#region 2. 函数：teacher-forcing 指标（不生成，最便宜）   【函数:eval_ppl,return dict】
@torch.inference_mode()
def eval_ppl(model, ds, collator):
    """只在回答段算 loss。返回：dict(loss, ppl, token_acc)。"""
    loss_sum, acc, n = 0.0, 0, 0
    for batch in DataLoader(ds, batch_size=8, collate_fn=collator):
        batch = {k: v.cuda() for k, v in batch.items()}
        logits = model(**batch).logits[:, :-1].float()               # (B, L-1, V)，错一位预测下一个
        tgt = batch["labels"][:, 1:]                                 # (B, L-1)
        mask = tgt != -100                                           # 只保留回答段
        sel = logits[mask]                                           # (N_ans, V)
        loss_sum += F.cross_entropy(sel, tgt[mask], reduction="sum").item()
        acc += (sel.argmax(-1) == tgt[mask]).sum().item()
        n += mask.sum().item()
    return {"loss": loss_sum / n, "ppl": math.exp(loss_sum / n), "token_acc": acc / n, "answer_tokens": n}
#endregion

#region 3. 函数：批量生成（左 padding + 双 eos）   【函数:eval_gen,return list[dict]】
@torch.inference_mode()
def eval_gen(model, tok, records, batch_size=8):
    """贪心解码。返回：list[dict]，含 text（截到 <|im_end|> 之前）与 stopped（是否自己收尾）。"""
    tok.padding_side = "left"                                        # 批量生成必须左 padding，否则结果错位
    prompts = [build_prompt(r["instruction"]) for r in records]
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=64, do_sample=False,
                             eos_token_id=EOS_IDS, pad_token_id=PAD_ID)   # eos 是 151643，必须显式给
        for seq in gen:
            raw = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=False)
            text = raw.split(ANSWER_END)[0].split(IM_END)[0].strip()    # 截掉终止符
            outs.append({"stopped": ANSWER_END in raw or IM_END in raw,   # 先记「有没有主动收尾」
                         "text": text})
    tok.padding_side = "right"                                       # 复位，别污染后面的调用
    return outs
#endregion

#region 4. 函数：把生成结果折算成 7 个指标   【函数:score,return dict】
def score(records, outs):
    """统计行为指标。返回：dict，全部为 0~1 的比例或均值。"""
    n, preds = len(outs), [o["text"] for o in outs]
    f1 = [char_f1(p, r["output"]) for p, r in zip(preds, records)]
    rl = [rouge_l_char(p, r["output"]) for p, r in zip(preds, records)]
    return {
        "stop_rate": sum(o["stopped"] for o in outs) / n,                        # 有没有学会收尾
        "multi_q_rate": sum(1 for p in preds if count_question_marks(p) >= 2) / n,   # 还在继续列问题？
        "len_in_range": sum(1 for p in preds if 6 <= len(p) <= 80) / n,          # 长度落进数据分布
        "char_f1": sum(f1) / n,
        "rouge_l": sum(rl) / n,
        "repeat_rate": sum(1 for p in preds if is_degenerate(p)) / n,            # 复读退化
        "echo_rate": sum(1 for p, r in zip(preds, records) if char_f1(p, r["instruction"]) > 0.8) / n,
        "avg_len": sum(len(p) for p in preds) / n,
    }
#endregion

#region 5. 主流程：同一份权重先量 base，再挂 adapter 量 lora
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    tok = get_tokenizer()
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    val_ds = load_from_disk(str(DATA_DIR / "val"))
    records = read_jsonl(SRC_VAL)[: args.n]

    model = load_base()
    result = {}
    for tag in ("base", "lora"):
        if tag == "lora":
            model = PeftModel.from_pretrained(model, str(OUT_DIR))    # 不重载底座，省一次 3 秒
        ppl = eval_ppl(model, val_ds, collator)
        print(f"[{tag}] answer_ppl={ppl['ppl']:.2f} token_acc={ppl['token_acc']:.3f}")
        outs = eval_gen(model, tok, records)
        result[tag] = {**ppl, **score(records, outs), "outs": outs}
        print(f"[{tag}] {({k: round(v, 3) for k, v in result[tag].items() if k not in ('outs', 'answer_tokens')})}")

    # 固定种子抽 10 条做三列对照
    idx = random.Random(SEED).sample(range(len(records)), N_SAMPLES)
    samples = [{"instruction": records[i]["instruction"], "reference": records[i]["output"],
                "base": result["base"]["outs"][i]["text"], "lora": result["lora"]["outs"][i]["text"]}
               for i in idx]

    # 随机基线：把预测打乱后重算 char_f1，它衡量「毫不相干的两句话能有多像」
    shuffled = random.Random(0).sample([o["text"] for o in result["lora"]["outs"]], len(records))
    baseline = sum(char_f1(p, r["output"]) for p, r in zip(shuffled, records)) / len(records)
    print(f"随机配对基线 char_f1 = {baseline:.3f}")

    save_json({"n": len(records), "base": result["base"], "lora": result["lora"],
               "random_baseline": baseline, "samples": samples}, ART_DIR / "eval.json")
    print(f"评估完成，结果写入 {ART_DIR / 'eval.json'}")
    return 0
#endregion

if __name__ == "__main__":
    sys.exit(main())

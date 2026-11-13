#region 0. 导入库
import json
import sys
from collections import Counter

from datasets import Dataset

from common import (DATA_DIR, ART_DIR, SRC_TRAIN, SRC_VAL, ANSWER_END, encode_example,
                    get_tokenizer, read_jsonl, save_json)
#endregion

#region 1. 常量   【常量:MAX_LEN 单条最大 token 数】
MAX_LEN = 512          # 数据实测最长约 160 token，512 足够，且 8GB 显存扛得住
#endregion

#region 2. 函数：列表 → 可直接喂给 Trainer 的 Dataset   【函数:build(records,tok),return Dataset】
def build(records, tok):
    """批量编码。返回：Dataset，列为 input_ids / labels / attention_mask。"""
    rows = [encode_example(tok, r["instruction"], r["output"], MAX_LEN) for r in records]
    return Dataset.from_dict({k: [r[k] for r in rows] for k in ("input_ids", "labels", "attention_mask")})
#endregion

#region 3. 函数：把 loss mask 渲染成人眼可读的文本   【函数:visualize(tok,row),return str】
def visualize(tok, row):
    """-100 段标 [MASK]、参与 loss 段标 [LEARN]。返回：str。"""
    out, prev = [], None
    for tid, lab in zip(row["input_ids"], row["labels"]):
        cur = lab == -100                                        # 当前 token 是否被 mask
        if cur != prev:
            out.append("\n[MASK] " if cur else "\n[LEARN] ")      # 只在切换处打标记
            prev = cur
        out.append(tok.decode([tid]))
    return "".join(out)
#endregion

#region 4. 主流程：编码 + 统计 + 自检
def main():
    tok = get_tokenizer()
    train_raw, val_raw = read_jsonl(SRC_TRAIN), read_jsonl(SRC_VAL)
    print(f"原始数据: train={len(train_raw)} val={len(val_raw)}")

    train_ds, val_ds = build(train_raw, tok), build(val_raw, tok)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(DATA_DIR / "train"))
    val_ds.save_to_disk(str(DATA_DIR / "val"))

    # 长度统计：确认不会截断，也确认回答段没被截空
    lens = sorted(len(x) for x in train_ds["input_ids"])
    n_answer = [sum(1 for t in lb if t != -100) for lb in train_ds["labels"]]     # 每条实际参与 loss 的 token 数
    stats = {"train": len(train_raw), "val": len(val_raw), "max_len": MAX_LEN,
             "token_len": {"p50": lens[len(lens) // 2], "p95": lens[int(len(lens) * .95)], "max": lens[-1]},
             "answer_tokens": {"min": min(n_answer), "mean": round(sum(n_answer) / len(n_answer), 1)},
             "truncated": sum(1 for x in lens if x >= MAX_LEN)}
    save_json(stats, ART_DIR / "data_stats.json")
    print(f"长度分位: {stats['token_len']}  截断条数: {stats['truncated']}")

    # mask 自检：三条判据全过才继续，否则训练一定错
    checks = []
    for idx in (0, 1, 2):
        text = visualize(tok, train_ds[idx])
        checks.append(text)
        mask_part = text.split("[LEARN]")[0].replace("[MASK]", "")
        learn_part = text.split("[LEARN]")[-1]
        q, a = train_raw[idx]["instruction"], train_raw[idx]["output"]
        ok_mask = q.replace(" ", "") in mask_part.replace(" ", "")               # 提问段确实没学
        ok_end = learn_part.rstrip().endswith(ANSWER_END)                        # 回答段确实收尾
        print(f"  样本{idx}: 提问段被mask={ok_mask} 回答段以endoftext结尾={ok_end}")
        assert ok_mask and ok_end, "mask 自检失败"
        checks[idx] = f"===== 样本 {idx} =====\n原始提问: {q}\n原始回答: {a}\n{text}\n"

    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ART_DIR / "mask_check.txt", "w", encoding="utf-8") as f:
        f.write("".join(checks))
    print(f"自检通过，明细见 {ART_DIR / 'mask_check.txt'}")
    return 0
#endregion

if __name__ == "__main__":
    sys.exit(main())

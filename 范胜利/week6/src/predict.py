# -*- coding: utf-8 -*-
"""
predict.py —— 用训练好的模型做推理（单条 / 批量 / 文件）

【前置：模型权重从哪来】
    train.py 默认只落指标、不落权重（对比实验不需要 400MB 的 ckpt）。
    要得到可推理的权重，训练时加 --save_ckpt：

        python src/train.py --backbone bert-base-chinese --save_ckpt
        → outputs/ckpt_bert-base-chinese.pt

    ckpt 里只存 state_dict + 训练配置 + 基座名，不含分词器；
    推理时按 ckpt 里记的 backbone 重新定位基座目录（本地没有会自动下载）。

【用法】
    # 单条
    python src/predict.py --ckpt outputs/ckpt_bert-base-chinese.pt \
        --text "二手房首付都付了，卖家现在不想卖了怎么办？"

    # 多条（命令行里给多个 --text，或直接给一个 txt / json / jsonl）
    python src/predict.py --ckpt outputs/ckpt_chinese-roberta-wwm-ext.pt \
        --file inputs.txt --out preds.json

    # 拿验证集抽查 20 条，看预测和真实标签对不对
    python src/predict.py --ckpt outputs/ckpt_bert-base-chinese.pt \
        --file data/val.json --limit 20

【输出】
    表格式：文本摘要 | 预测类别 | 置信度 | Top3 | 主题组 | 是否投研相关
    --out 会给一份 json，字段固定，方便下游服务直接用。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401  ← 必须第一行：torch 要先于 numpy 加载（见 _bootstrap.py）

import torch
import torch.nn.functional as F

from config import backbone_spec, ensure_backbone, resolve_device
from labels import GROUP_MAP, LABELS, NUM_LABELS
from model import BackboneClassifier

# 投研口径：金融/产业算相关，宏观弱相关，其余是噪音（与第六周 labels.py 一致）
RELEVANT_GROUPS = {"金融", "产业"}


def load_ckpt(path: Path):
    """读 ckpt，返回 (state_dict, config, backbone_name)"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}\n"
            f"训练时加 --save_ckpt 才会生成：python src/train.py --backbone bert-base-chinese --save_ckpt"
        )
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def build_model(ckpt_path: Path, backbone_override: str = None):
    """
    返回 (model, tokenizer, meta)。

    为什么池化方式/隐藏维度要从 ckpt 的 config 里读：
      换 pool（cls/mean/max）不改变 state_dict 的键名，加载不会报错，
      但推理结果会完全不同——这类 bug 不会抛异常，只会让效果莫名变差。
      所以训练时把 config 一起存进 ckpt，推理时严格按它重建结构。
    """
    ckpt = load_ckpt(ckpt_path)
    state, cfg = ckpt["state_dict"], ckpt.get("config", {})
    backbone = backbone_override or ckpt.get("backbone") or "bert-base-chinese"
    spec = backbone_spec(backbone)
    model_dir = ensure_backbone(backbone)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = BackboneClassifier(
        model_dir,
        num_labels=NUM_LABELS,
        pool=cfg.get("pool", "cls"),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state, strict=True)   # 结构必须与训练时完全一致，strict 能立刻暴露不一致
    model.eval()

    meta = {
        "backbone": backbone,
        "display": spec["display"],
        "max_length": cfg.get("max_length", 64),
        "pool": cfg.get("pool", "cls"),
        "metrics": ckpt.get("metrics", {}),
        "best_epoch": ckpt.get("best_epoch"),
        "trainsize": cfg.get("train_size"),
        "seed": cfg.get("seed"),
    }
    return model, tok, meta


@torch.no_grad()
def predict(model, tok, texts, max_length: int = 64, device=None, batch_size: int = 32, topk: int = 3):
    """
    批量推理，返回 list[dict]：
        label_id / label / prob / group / relevant / topk:[{label, prob}]

    为什么要 padding 到 batch 内最长而不是固定 64：
      和训练时保持一致的「动态 padding」口径，短句不会白算 PAD；
      同时 mask 正确，mean/max 池化不会把补位算进去。
    """
    device = device or torch.device("cpu")
    model.to(device)
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = F.softmax(logits.float(), dim=-1).cpu()

        for i, text in enumerate(chunk):
            p = probs[i]
            k = min(topk, NUM_LABELS)
            top_p, top_i = p.topk(k)
            label_id = int(top_i[0])
            group = GROUP_MAP.get(LABELS[label_id], "未知")
            out.append({
                "text": text,
                "label_id": label_id,
                "label": LABELS[label_id],
                "prob": round(float(top_p[0]), 4),
                "group": group,
                "relevant": group in RELEVANT_GROUPS,
                "topk": [
                    {"label": LABELS[int(idx)], "prob": round(float(v), 4)}
                    for v, idx in zip(top_p, top_i)
                ],
            })
    return out


def load_inputs(args) -> list:
    """按 --text / --file 收集待预测文本；json/jsonl 支持 sentence/text/内容 三种字段名"""
    texts = list(args.text or [])
    if args.file:
        p = Path(args.file)
        if not p.exists():
            raise FileNotFoundError(f"找不到输入文件 {p}")
        raw = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            for line in raw.splitlines():
                if line.strip():
                    d = json.loads(line)
                    texts.append(d.get("sentence") or d.get("text") or "")
        elif p.suffix == ".json":
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            for d in data:
                if isinstance(d, str):
                    texts.append(d)
                else:
                    texts.append(d.get("sentence") or d.get("text") or "")
        else:   # 一行一条
            texts.extend([line.strip() for line in raw.splitlines() if line.strip()])
    texts = [t for t in texts if t and t.strip()]
    if args.limit:
        texts = texts[: args.limit]
    return texts


def main():
    ap = argparse.ArgumentParser(description="用训练好的 BERT/RoBERTa 分类模型做推理")
    ap.add_argument("--ckpt", required=True, help="train.py --save_ckpt 产出的 .pt")
    ap.add_argument("--backbone", default=None, help="覆盖 ckpt 里记录的基座（一般不用传）")
    ap.add_argument("--text", action="append", help="待预测文本，可重复传多次")
    ap.add_argument("--file", help="批量输入：.txt(一行一条) / .json / .jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（抽查用）")
    ap.add_argument("--topk", type=int, default=3, help="同时给出前 K 个候选类别")
    ap.add_argument("--out", default=None, help="把结果写成 json")
    ap.add_argument("--device", default="cpu", help="cpu/mps/cuda，默认 cpu（BERT 单句推理 CPU 足够）")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, tok, meta = build_model(Path(args.ckpt), args.backbone)

    texts = load_inputs(args)
    if not texts:
        print("没有输入。用 --text 给一条，或用 --file 指定文件。例：\n"
              "  python src/predict.py --ckpt outputs/ckpt_bert-base-chinese.pt --text \"二手房卖家违约怎么办\"")
        sys.exit(1)

    results = predict(model, tok, texts, meta["max_length"], device, topk=args.topk)

    print(f"\n模型：{meta['display']}  |  池化：{meta['pool']}  |  max_length={meta['max_length']}  |  设备：{device}")
    print("-" * 92)
    print(f"{'文本':<34} {'预测':<6} {'置信':<7} {'主题组':<5} {'投研':<5} Top3")
    print("-" * 92)
    for r in results:
        snippet = (r["text"][:30] + "…") if len(r["text"]) > 30 else r["text"]
        top3 = " / ".join(f"{t['label']}{t['prob']:.2f}" for t in r["topk"])
        print(f"{snippet:<34} {r['label']:<6} {r['prob']:<7.4f} {r['group']:<5} "
              f"{'是' if r['relevant'] else '否':<5} {top3}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()

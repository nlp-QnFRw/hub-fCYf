"""
方法四：预训练模型微调（L3 主力）

同一个脚本训两个模型，回答两个不同的问题：
  macbert（12 层 110M）—— 中文效果能到哪
  rbt3    （ 3 层 16M）—— 砍掉 3/4 的层，效果掉多少、速度快多少

关键工程点（照抄原项目踩过的坑）：
  1. 池化必须 mask 掉 [PAD]；
  2. 分层学习率（编码器 2e-5、分类头 ×5）比换模型更值钱；
  3. 长文本先预 tokenize 一次，不要在 DataLoader 里反复分词（40k×3 epoch 会白等半小时）；
  4. transformers 4.x / 5.x 的 dtype 参数名不同，统一在 model.py 里兼容。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.data import load_splits
from common.labels import CLASSES
from common.metrics import classification_report, expected_calibration_error, to_markdown_table
from common.model_paths import macbert_path, rbt3_path
from model import BertClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


class Tokenized(Dataset):
    """预 tokenize 好的张量集（内存换时间）"""

    def __init__(self, data, tok, max_len):
        self.ids, self.mask, self.y = [], [], []
        for d in data:
            e = tok(d["text"], truncation=True, max_length=max_len, padding="max_length")
            self.ids.append(torch.tensor(e["input_ids"]))
            self.mask.append(torch.tensor(e["attention_mask"]))
            self.y.append(d["label"])
        self.ids = torch.stack(self.ids)
        self.mask = torch.stack(self.mask)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.mask[i], self.y[i]


@torch.no_grad()
def evaluate(model, dl, device, need_probs=False):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    ys, ps, probs = [], [], []
    for ids, mask, y in dl:
        logits = model(ids.to(device), mask.to(device))
        p = torch.softmax(logits.float(), -1)
        ps.extend(p.argmax(-1).cpu().tolist())
        ys.extend(y.tolist())
        if need_probs:
            probs.extend(p.cpu().tolist())
    res = {"acc": round(float(accuracy_score(ys, ps)), 4),
           "macro_f1": round(float(f1_score(ys, ps, average="macro")), 4)}
    return (res, ys, ps, probs) if need_probs else (res, ys, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="macbert", choices=["macbert", "rbt3"])
    ap.add_argument("--tag", default=None, help="结果文件名前缀，默认取 --model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--n_train", type=int, default=0, help=">0 时只用前 N 条（数据量实验）")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="每 epoch 只跑 30 步，验证流程")
    args = ap.parse_args()

    tag = args.tag or args.model
    path = macbert_path() if args.model == "macbert" else rbt3_path()

    from transformers import AutoTokenizer
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(path))
    train, val, meta = load_splits(max_len=args.max_len)
    if args.n_train:
        train = train[: args.n_train]

    t0 = time.time()
    train_ds = Tokenized(train, tok, args.max_len)
    val_ds = Tokenized(val, tok, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    print(f"[data] 训练 {len(train)} / 验证 {len(val)}，预分词 {time.time()-t0:.0f}s", flush=True)

    model = BertClassifier(path, len(CLASSES), pool=args.pool).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    head = list(model.classifier.parameters()) + list(model.dropout.parameters())
    enc = [p for n, p in model.named_parameters() if not n.startswith(("classifier", "dropout"))]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr},
                             {"params": head, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=0.01)

    cw = compute_class_weight("balanced", classes=np.arange(len(CLASSES)),
                              y=[d["label"] for d in train])
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(device))

    steps_per_epoch = min(len(train_dl), 30) if args.smoke else len(train_dl)
    total = steps_per_epoch * args.epochs
    warm = int(total * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = (device == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[cfg] {tag} 参数量 {n_param/1e6:.1f}M | max_len {args.max_len} | pool {args.pool} | "
          f"bs {args.batch_size} | lr {args.lr} | epochs {args.epochs} | amp {use_amp}", flush=True)

    log, best, ckpt_path = [], 0.0, OUT / f"{tag}.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        t1, run, nb = time.time(), 0.0, 0
        for i, (ids, mask, y) in enumerate(train_dl):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            nb += 1
            if i % 200 == 0:
                print(f"  {tag} ep{ep} {i}/{steps_per_epoch} loss={loss.item():.4f} "
                      f"{time.time()-t1:.0f}s", flush=True)
            if args.smoke and nb >= 30:
                break
        res, _, _ = evaluate(model, val_dl, device)
        log.append({"epoch": ep, "train_loss": round(run / max(1, nb), 4),
                    "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                    "epoch_s": round(time.time() - t1, 1)})
        print(f"epoch {ep}: loss={log[-1]['train_loss']} val_acc={res['acc']} "
              f"val_f1={res['macro_f1']} ({log[-1]['epoch_s']}s)", flush=True)
        if res["macro_f1"] > best:
            best = res["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "pool": args.pool, "model": args.model,
                        "val_macro_f1": best, "args": vars(args)}, ckpt_path)

    # 用最佳权重重新评测，顺带测延迟与 ECE
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    t2 = time.time()
    res, ys, ps, probs = evaluate(model, val_dl, device, need_probs=True)
    infer_s = time.time() - t2

    rep = classification_report(ys, ps)
    rep.update({"method": tag, "params_m": round(n_param / 1e6, 2), "pool": args.pool,
                "n_train": len(train), "epochs": args.epochs,
                "train_seconds": round(sum(x["epoch_s"] for x in log), 1),
                "ms_per_sample": round(infer_s * 1000 / len(ys), 1),
                "ece": expected_calibration_error(ys, probs)})

    (OUT / f"{tag}_pred.json").write_text(json.dumps(
        {"meta": {**meta, "n_eval": len(ys), "n_train": len(train)}, "report": rep, "log": log,
         "preds": [{"text": d["text"][:60], "true": y, "pred": p} for d, y, p in zip(val, ys, ps)]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(to_markdown_table({f"{tag}（{n_param/1e6:.1f}M）": rep}))
    print(f"→ ckpt {ckpt_path} | 逐条预测 outputs/{tag}_pred.json")


if __name__ == "__main__":
    main()

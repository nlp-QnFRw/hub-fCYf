"""
export_lm_model.py — 把训练好的字符级语言模型（rnn / lstm / transformer）导出成网页可用的 JS

网页端用纯 JS 复现前向：
    rnn/lstm     : Embedding → N × RNN/LSTM → Linear(hidden → vocab)
    transformer  : Embedding + 位置编码 → N × [LN → 因果多头自注意 → 残差 → LN → FFN → 残差] → Linear

权重按 fp16 + base64 导出（体积减半，对概率的影响在 1e-3 量级），
同时写入几组上下文的「下一个字」log 概率（供浏览器对拍）和验证集 PPL（供页面对比）。

用法:
    python export_lm_model.py                          # 默认三个模型全导
    python export_lm_model.py --model best_model.pt
    python export_lm_model.py --out ..\\web\\lm_model_lstm.js
"""

import argparse
import base64
import glob
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LM_DIR = HERE
WEB_DIR = HERE
sys.path.insert(0, LM_DIR)

from language_model import LM, CharDataset, load_corpus  # noqa: E402  与训练脚本共用模型定义

# 默认导出任务：(checkpoint, 输出文件)
DEFAULT_JOBS = [
    ("best_model_transformer.pt", "lm_model_transformer.js"),
    ("best_model.pt", "lm_model_lstm.js"),
    ("best_model_rnn.pt", "lm_model_rnn.js"),
]

CHECK_CONTEXTS = ["今天", "我们", "中国", "的", "天气", "人工智能", "我喜欢", "这是"]
SEED, VAL_RATIO = 42, 0.05


def b64_fp16(arr):
    return base64.b64encode(np.ascontiguousarray(arr).astype(np.float16).tobytes()).decode("ascii")


def pack_rnn_layer(model, i):
    """PyTorch 的 weight_ih 行序是 i,f,g,o 四组，直接按行平铺即可。"""
    w_ih = getattr(model.rnn, f"weight_ih_l{i}").detach().numpy()
    w_hh = getattr(model.rnn, f"weight_hh_l{i}").detach().numpy()
    b = (getattr(model.rnn, f"bias_ih_l{i}") + getattr(model.rnn, f"bias_hh_l{i}")).detach().numpy()
    return {
        "w_ih": b64_fp16(w_ih),
        "w_hh": b64_fp16(w_hh),
        "bias": b64_fp16(b),
        "in": int(w_ih.shape[1]),
        "hidden": int(w_hh.shape[1]),
        "gates": int(w_ih.shape[0] // w_hh.shape[1]),
    }


def pack_transformer_layer(layer):
    attn = layer.self_attn
    return {
        "qkv_w": b64_fp16(attn.in_proj_weight.detach().numpy()),   # (3E, E) 行序 Q,K,V
        "qkv_b": b64_fp16(attn.in_proj_bias.detach().numpy()),
        "out_w": b64_fp16(attn.out_proj.weight.detach().numpy()),
        "out_b": b64_fp16(attn.out_proj.bias.detach().numpy()),
        "l1_w": b64_fp16(layer.linear1.weight.detach().numpy()),   # (F, E)
        "l1_b": b64_fp16(layer.linear1.bias.detach().numpy()),
        "l2_w": b64_fp16(layer.linear2.weight.detach().numpy()),   # (E, F)
        "l2_b": b64_fp16(layer.linear2.bias.detach().numpy()),
        "n1_w": b64_fp16(layer.norm1.weight.detach().numpy()),
        "n1_b": b64_fp16(layer.norm1.bias.detach().numpy()),
        "n2_w": b64_fp16(layer.norm2.weight.detach().numpy()),
        "n2_b": b64_fp16(layer.norm2.bias.detach().numpy()),
    }


def val_ppl(model, char2idx, cfg, corpus_dir, device="cpu"):
    """与训练时同一套划分（seed 42 / 95-5 / 同 stride）算验证集 PPL。"""
    text = load_corpus(os.path.join(corpus_dir, "*.txt"))
    lines = text.splitlines()
    random.seed(SEED)
    random.shuffle(lines)
    val_text = "\n".join(lines[int(len(lines) * (1 - VAL_RATIO)):])

    ds = CharDataset(val_text, char2idx, cfg["seq_len"], cfg.get("stride", 1))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)
    crit = torch.nn.CrossEntropyLoss()
    total, tokens = 0.0, 0
    model.to(device).eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            total += crit(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)).item() * y.numel()
            tokens += y.numel()
    return math.exp(total / tokens)


def selftest(model, char2idx, idx2char, contexts):
    checks = []
    for ctx in contexts:
        ids = [char2idx[c] for c in ctx if c in char2idx]
        if not ids:
            continue
        with torch.no_grad():
            logp = F.log_softmax(model(torch.tensor([ids], dtype=torch.long))[0, -1], dim=-1).numpy()
        order = np.argsort(-logp)[:8]
        checks.append({
            "context": ctx,
            "top": [[idx2char[int(i)], round(float(logp[i]), 4)] for i in order],
        })
    return checks


def export_one(ckpt, out_path, do_ppl=True, key=None):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    arch = cfg["model"]
    vocab_size = len(char2idx)
    nhead = cfg.get("nhead", 8)
    max_len = cfg.get("max_len", 1024)

    model = LM(
        vocab_size=vocab_size, embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], model_type=arch, dropout=0.0,
        nhead=nhead, max_len=max_len,
    )
    model.load_state_dict(ck["model_state"])
    model.eval()

    fc_in = int(model.fc.weight.shape[1])
    payload = {
        "meta": {
            "arch": arch,
            "file": os.path.basename(ckpt),
            "embed_dim": cfg["embed_dim"],
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "nhead": nhead if arch == "transformer" else 0,
            "max_len": max_len if arch == "transformer" else 0,
            "vocab_size": vocab_size,
            "fc_in": fc_in,
            "params": sum(p.numel() for p in model.parameters()),
            "norm_eps": 1e-5,
            "dtype": "float16",
        },
        "vocab": "".join(idx2char[i] for i in range(vocab_size)),
        "weights": {
            "embed": b64_fp16(model.embed.weight.detach().numpy()),
            "fc_w": b64_fp16(model.fc.weight.detach().numpy()),
            "fc_b": b64_fp16(model.fc.bias.detach().numpy()),
            "layers": ([pack_transformer_layer(l) for l in model.encoder.layers]
                       if arch == "transformer"
                       else [pack_rnn_layer(model, i) for i in range(cfg["num_layers"])]),
        },
        "checks": selftest(model, char2idx, idx2char, CHECK_CONTEXTS),
    }
    if arch == "transformer":
        payload["weights"]["pos"] = b64_fp16(model.pos.weight.detach().numpy())

    if do_ppl:
        ppl = val_ppl(model, char2idx, cfg, os.path.dirname(os.path.abspath(ckpt)))
        payload["meta"]["val_ppl"] = round(ppl, 2)
        model.cpu()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    reg_key = key if key else (arch if arch != "transformer" else "transformer")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write('window.LM_MODELS = window.LM_MODELS || {};\n')
        f.write(f'window.LM_MODELS["{reg_key}"] = ')
        f.write(body)
        f.write(";\n")

    m = payload["meta"]
    print(f"  {os.path.basename(out_path):26s} {arch:12s} 参数 {m['params']:>9,} "
          f"PPL {payload['meta'].get('val_ppl', '-')}"
          f"  {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="单个 checkpoint；省略则导出默认三个")
    ap.add_argument("--out", default=None, help="配合 --model 指定输出文件")
    ap.add_argument("--key", default=None, help="模型在 window.LM_MODELS 中的注册键名（默认用 arch）")
    ap.add_argument("--no-ppl", action="store_true", help="跳过验证集 PPL 计算（更快）")
    args = ap.parse_args()

    if args.model:
        ckpt = args.model if os.path.isabs(args.model) else os.path.join(LM_DIR, args.model)
        out = args.out or os.path.join(WEB_DIR, "lm_model.js")
        jobs = [(ckpt, out, args.key)]
    else:
        jobs = [(os.path.join(LM_DIR, c), os.path.join(WEB_DIR, o), None) for c, o in DEFAULT_JOBS]

    print("导出字符级语言模型 -> web/*.js")
    for ckpt, out, key in jobs:
        export_one(ckpt, out, do_ppl=not args.no_ppl, key=key)
    print("完成。")


if __name__ == "__main__":
    main()

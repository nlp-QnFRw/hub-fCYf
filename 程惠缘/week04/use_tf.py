"""
use_tf.py —— 加载 transformer_lm.py 训好的模型，看它学到了什么

两种玩法:
    1. 预测下一个字：给一句话，看模型认为下一个字最可能是什么
    2. 续写：给个开头，让模型自己往下写

用法:
    python use_tf.py                                  # 内置示例
    python use_tf.py -t "今天天气"                     # 看下一个字的候选
    python use_tf.py -t "今天天气" --gen 30            # 接着往下写 30 个字
    python use_tf.py -t "怎么" --gen 40 --temperature 0.8   # 带点随机性
    python use_tf.py -i                               # 交互模式
    python use_tf.py -m 别的模型.pt
"""
import os
import sys
import argparse

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "best_model_tf.pt")


class TransformerLM(nn.Module):
    """必须和 transformer_lm.py 里的定义完全一致，否则权重对不上。"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn, dropout, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.register_buffer(
            "causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool),
                                 diagonal=1),
            persistent=False)

    def forward(self, x):
        length = x.size(1)
        pos = torch.arange(length, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        h = self.encoder(h, mask=self.causal[:length, :length])
        return self.fc(self.norm(h))


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("arch") != "transformer":
        raise SystemExit(f"{path} 不是 transformer 模型的存档（arch={ckpt.get('arch')}）")
    cfg = ckpt["model_cfg"]
    model = TransformerLM(cfg["vocab_size"], cfg["d_model"], cfg["nhead"],
                          cfg["num_layers"], cfg["ffn"], cfg["dropout"],
                          cfg["max_len"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["char2idx"], ckpt["idx2char"], cfg, ckpt


def show_char(c):
    return {"\n": "⏎", " ": "␣"}.get(c, c)


def predict_next(model, char2idx, idx2char, text, k, device, cfg):
    ids = [char2idx[c] for c in text if c in char2idx][-(cfg["max_len"] - 1):]
    if not ids:
        return []
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[0, -1]
    probs = torch.softmax(logits, dim=-1)
    vals, idx = probs.topk(k)
    return [(idx2char[int(i)], float(p)) for p, i in zip(vals, idx)]


def generate(model, char2idx, idx2char, prompt, n, device, cfg, temperature=0.0):
    ids = [char2idx[c] for c in prompt if c in char2idx][-(cfg["max_len"] - 1):]
    if not ids:
        return ""
    out = []
    for _ in range(n):
        x = torch.tensor([ids[-cfg["max_len"]:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[0, -1]
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        ids.append(nxt)
        out.append(idx2char[nxt])
    return "".join(out)


def demo(model, char2idx, idx2char, device, cfg):
    print("\n（没给 -t，跑内置示例：先看下一个字的候选，再让它续写）")
    for prompt in ("今天天气", "我不知道", "为什么"):
        print(f"\n给定「{prompt}」→ 下一个字最可能是：")
        for ch, p in predict_next(model, char2idx, idx2char, prompt, 5, device, cfg):
            bar = "#" * int(round(p * 30))
            print(f"  {show_char(ch):<3} {p:6.2%}  {bar}")
        print(f"  续写 20 字：{prompt}{generate(model, char2idx, idx2char, prompt, 20, device, cfg)}")


def main():
    ap = argparse.ArgumentParser(description="看 Transformer 语言模型学到了什么")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-t", "--text", default="", help="给一句开头")
    ap.add_argument("-g", "--gen", type=int, default=0, help="续写多少字（0=只预测下一个字）")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0=贪心（确定），越大越随机")
    ap.add_argument("-k", "--topk", type=int, default=5)
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)
    if not os.path.exists(args.model):
        raise SystemExit(f"找不到模型：{args.model}\n  先训一个：python transformer_lm.py")

    model, char2idx, idx2char, cfg, ckpt = load(args.model, device)
    a = ckpt["args"]
    print(f"模型：{os.path.basename(args.model)}  "
          f"d_model={cfg['d_model']} 层={cfg['num_layers']} 头={cfg['nhead']} "
          f"ffn={cfg['ffn']}  词表={cfg['vocab_size']}  最长 {cfg['max_len']}")
    print(f"训练设定：{a['epochs']} epoch / seq_len={a['seq_len']} / stride={a['stride']} "
          f"/ batch={a['batch_size']} / lr={a['lr']}")

    if args.interactive:
        print("\n输入一句开头回车 → 续写；空行或 :q 退出。")
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not line or line in (":q", "quit", "exit"):
                break
            n = args.gen or 20
            print(f"  {line}{generate(model, char2idx, idx2char, line, n, device, cfg, args.temperature)}")
    elif args.text:
        print(f"\n给定「{args.text}」→ 下一个字最可能是：")
        for ch, p in predict_next(model, char2idx, idx2char, args.text, args.topk, device, cfg):
            bar = "#" * int(round(p * 30))
            print(f"  {show_char(ch):<3} {p:6.2%}  {bar}")
        if args.gen:
            print(f"\n续写 {args.gen} 字"
                  f"（temperature={args.temperature}）：")
            print(f"  {args.text}{generate(model, char2idx, idx2char, args.text, args.gen, device, cfg, args.temperature)}")
    else:
        demo(model, char2idx, idx2char, device, cfg)


if __name__ == "__main__":
    main()

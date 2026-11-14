import json
import torch
from sft import load_base_model, BASE_CKPT, TransformerLM


def load_sft_model(path):
    """加载 SFT 训练好的模型"""
    ck = torch.load(path, map_location="cpu")
    cfg = ck["args"]
    model = TransformerLM(len(ck["char2idx"]), cfg["embed_dim"], cfg["num_heads"], cfg["ff_dim"],
                          cfg["num_layers"], cfg["seq_len"], 0.0, tricks=cfg["tricks"])
    model.load_state_dict(ck["model_state"])
    return model, ck["char2idx"], ck["idx2char"]


@torch.no_grad()
def generate(model, instruction, char2idx, idx2char, max_new=60):
    """给一条指令，让模型一个字一个字地生成回答，说出 <EOS> 就停"""
    model.eval()
    prompt = [char2idx[c] for c in instruction if c in char2idx] + [char2idx["<SEP>"]]
    out = []
    for _ in range(max_new):
        logits = model(torch.tensor([prompt + out]))[0, -1]   # 最后一个位置：对"下一个字"的打分
        nxt = logits.argmax().item()                           # 分数最高的那个字的编号
        if nxt == char2idx["<EOS>"]:                           # 说出 <EOS> 就停            
            break
        out.append(nxt)
    return "".join(idx2char[i] for i in out)


if __name__ == "__main__":
    torch.manual_seed(0)
    base, _, char2idx, idx2char = load_base_model(BASE_CKPT)   # SFT 前
    sft, _, _ = load_sft_model("sft_best.pt")                  # SFT 后

    for line in list(open("sft_val.jsonl", encoding="utf-8"))[:5]:
        s = json.loads(line)
        print("=" * 60)
        print("指令  :", s["instruction"])
        print("参考  :", s["output"])
        print("SFT 前:", generate(base, s["instruction"], char2idx, idx2char))
        print("SFT 后:", generate(sft,  s["instruction"], char2idx, idx2char))

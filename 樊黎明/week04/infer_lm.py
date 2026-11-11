import torch
import random
import argparse
from language_model import LM  # 复用训练脚本里的模型类
@torch.no_grad()
def generate_text(model, prompt, char2idx, idx2char, device,
                  max_len=200, temperature=1.0):
    """
    通用字符级生成函数，兼容RNN/LSTM/Transformer三种模型
    temperature > 1.0 生成更随机、更有创意
    temperature < 1.0 生成更保守、更贴近训练语料
    """
    model.eval()
    # 把输入提示转成索引序列，过滤掉不在词表里的字符
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        ids = [random.choice(list(char2idx.values()))]
    
    generated = ids.copy()
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_len):
        # 取最近的seq_len个字符，避免Transformer位置编码越界
        logits = model(x)[:, -1, :]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_id)
        # 拼接新字符，更新输入序列
        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        x = torch.cat([x, next_tensor], dim=1)

    return "".join([idx2char[i] for i in generated])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="week04/best_model.pt")
    parser.add_argument("--prompt", default="床前明月光，")
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 checkpoint，恢复所有训练时的配置
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    model_args = ckpt["args"]

    # 用训练时完全一致的参数重建模型
    model = LM(
        vocab_size=len(char2idx),
        embed_dim=model_args["embed_dim"],
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        model_type=model_args["model"],
        dropout=model_args["dropout"],
        nhead=model_args.get("nhead", 4)
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    print(f"✅ 加载成功: 模型类型={model_args['model'].upper()}, 词表大小={len(char2idx)}")

    # 生成文本
    result = generate_text(
        model, args.prompt, char2idx, idx2char, device,
        max_len=args.max_len, temperature=args.temperature
    )

    print("\n📝 生成结果:")
    print(result)

if __name__ == "__main__":
    main()

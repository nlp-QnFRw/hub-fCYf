"""加载训练好的字符语言模型，交互式预测下一个字符。"""

import argparse
from pathlib import Path

import torch

from language_model import LM as RecurrentLM
from language_model_transformer import LM as TransformerLM


def main():
    parser = argparse.ArgumentParser(description="预测输入文本的下一个字符")
    parser.add_argument("--model_path", default=str(Path(__file__).with_name("transformer_model.pt")))
    parser.add_argument("--topk", type=int, default=10, help="显示概率最高的候选数量")
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("--topk 必须大于 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    config = checkpoint["args"]
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    if config["model"] == "transformer":
        model = TransformerLM(len(char2idx), config["embed_dim"], config["hidden_dim"],
                              config["num_layers"], config["num_heads"], config["seq_len"],
                              config["dropout"]).to(device)
    else:
        model = RecurrentLM(len(char2idx), config["embed_dim"], config["hidden_dim"],
                            config["num_layers"], config["model"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"模型加载成功，使用设备：{device}。输入一个字或一段文字，输入 quit 退出。")
    while True:
        try:
            text = input("\n请输入：")
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            print("请输入至少一个字符。")
            continue
        unknown = list(dict.fromkeys(c for c in text if c not in char2idx))
        if unknown:
            print(f"以下字符不在模型词表中，请修改后重试：{''.join(unknown)}")
            continue
        # 训练时每段最多有 seq_len 个字符作为上下文。
        context = text[-config["seq_len"]:]
        x = torch.tensor([[char2idx[c] for c in context]], dtype=torch.long, device=device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(x)[0, -1], dim=-1)
            values, indices = torch.topk(probabilities, min(args.topk, len(char2idx)))
        print("下一个字符的候选及概率：")
        for value, index in zip(values.tolist(), indices.tolist()):
            print(f"  {idx2char[index]!r}：{value:.2%}")


if __name__ == "__main__":
    main()

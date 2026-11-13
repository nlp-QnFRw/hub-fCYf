import torch
from transformers import EncoderDecoderModel, BertTokenizer

MODEL_PATH = "./bert-sft-qa"
MAX_LENGTH = 128


def load_model():
    print("加载模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = EncoderDecoderModel.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"模型已加载到 {device}")
    return model, tokenizer, device


def generate_answer(model, tokenizer, device, question):
    inputs = tokenizer(
        question,
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            decoder_start_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )

    # BertTokenizerFast decode 会在中文字符 token 之间加空格，需去掉
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(" ", "")
    return answer


def main():
    model, tokenizer, device = load_model()

    print("\n===== BERT SFT 问答推理 =====")
    print("输入问题，模型会生成答案。输入 'exit' 退出。\n")

    while True:
        try:
            question = input("问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            break

        answer = generate_answer(model, tokenizer, device, question)
        print(f"回答: {answer}\n")

    print("已退出。")


if __name__ == "__main__":
    main()

"""
用本地 Qwen2-0.5B-Instruct 做 zero-shot / few-shot 分类，与 BERT fine-tune 对比

教学重点：
  1. LLM zero-shot / few-shot 分类 vs fine-tuning：不需要训练，但需要精心设计 prompt
  2. 结构化输出：通过 prompt 约束模型只输出类别名，简化解析逻辑
  3. 对比视角：LLM 在哪些类别上表现好/差？为什么？

使用方式：
  python classify_llm.py                    # 随机采样 200 条评估
  python classify_llm.py --num_samples 500  # 采样 500 条
  python classify_llm.py --demo             # 只跑 5 条示例，快速演示
  python classify_llm.py --few_shot         # 每类提供 2 条训练集示例
  python classify_llm.py --few_shot --demo  # few-shot 快速演示

依赖：
  pip install torch transformers
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODEL_PATH = ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"

# 15 个类别名，用于 prompt 和解析
LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)

# 固定使用训练集标题，不从验证集取示例，以便两种模式评估同一批样本。
FEW_SHOT_EXAMPLES: dict[str, tuple[str, str]] = {
    "故事": (
        "这座监狱自由度奇高，囚犯们可以互相交换物资，甚至结婚生孩子",
        "它是一艘星际战舰，觉醒了人类意识，即将面临着被拆解",
    ),
    "文化": (
        "上联：千峰入眠松涛静，怎么接下联？",
        "2018年5月全国及各省书法展览征稿启事汇总",
    ),
    "娱乐": (
        "电影《战犬瑞克斯》点映获赞 人犬情深看点十足",
        "如何评价许鞍华导演？她的电影为什么总能给人以触动？",
    ),
    "体育": (
        "再次证明了“无敌是多么寂寞”——逆天的中国乒乓球队！",
        "从哪开始从哪结束！杨毅神预测，火箭勇士西决大战一触即发",
    ),
    "财经": (
        "银行“恼羞成怒”：在微信支付宝上借款两次，不给贷款",
        "如果银行倒闭了，用户会得到赔偿吗？",
    ),
    "房产": (
        "为何全国房价在降，唯独哈尔滨在猛涨？",
        "没资格在成都买房的人，怎样才能买到房？",
    ),
    "汽车": (
        "街头偶遇2018款长安CS35，颜值美炸！或售6万起，还买宝骏510？",
        "比亚迪新能源车4月销售数据公布 比亚迪秦狂销4134辆",
    ),
    "教育": (
        "学校或老师有没有权利没收学生手机？",
        "全国职业教育“奥林匹克”开赛 福州三学子摘金",
    ),
    "科技": (
        "阿里巴巴区块链专利排名全球第一，阿里都有哪些区块链应用？",
        "苹果的A11芯片相当于电脑的哪款CPU？",
    ),
    "军事": (
        "军用匕首在关键时刻能发挥哪些作用？",
        "航母受到多少攻击才会被击沉？中国的东风家族让你刮目相看",
    ),
    "旅游": (
        "去河南的这些地方旅游，大家有什么景点推荐吗？",
        "一个人去海南旅游，该如何规划行程？",
    ),
    "国际": (
        "伊朗为什么宁愿接受制裁也不同意修改伊核协议？",
        "以色列总理将出席俄罗斯阅兵式 5月9日会晤普京",
    ),
    "证券": (
        "建信基金梁洪昀：2018投资三大关键词",
        "如何看待股票止损与死扛？",
    ),
    "农业": (
        "「牛羊催肥」肉羊吃什么饲料催肥增重长得快，一天长7两-1斤",
        "广西贵港适合种植红妃樱桃吗？",
    ),
    "电竞": (
        "《我的世界》问答题：这几个头颅都有什么作用呢？答对的都是大神",
        "王者荣耀：李白和韩信还没加强这个人民币战士却先加强了，悲哀！",
    ),
}

FEW_SHOT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n\n以下每个类别各有两个分类示例。请参考示例判断新标题，"
      "仍然只输出一个类别名，不要复述示例。\n\n"
    + "\n\n".join(
        f"新闻标题：{title}\n类别：{label}"
        for label in LABEL_NAMES
        for title in FEW_SHOT_EXAMPLES[label]
    )
)


def build_prompt(text: str) -> str:
    return f"新闻标题：{text}\n类别："


def load_model(model_path: str, device: torch.device):
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # transformers 5.x 将 torch_dtype 参数重命名为 dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    print("模型加载完成")
    return model, tokenizer


def classify_one(text: str, model, tokenizer, device: torch.device,
                 max_new_tokens: int = 8,
                 system_prompt: str = SYSTEM_PROMPT) -> str:
    """返回模型生成的类别字符串（原始输出，未做解析）。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": build_prompt(text)},
    ]
    # transformers 5.x apply_chat_template 返回 BatchEncoding，需取 ["input_ids"]
    encoding  = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy decoding，保证结果可复现
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的部分
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw_output: str) -> str | None:
    """
    从模型输出中提取类别名。
    模型可能输出 "科技" 或 "科技新闻" 或 "科技类"，做模糊匹配。
    """
    for name in LABEL_NAMES:
        if name in raw_output:
            return name
    return None  # 无法解析


def main():
    parser = argparse.ArgumentParser(description="LLM Zero-Shot / Few-Shot 分类对比")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--num_samples", default=200, type=int,
                        help="从验证集随机采样的样本数")
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--demo",        action="store_true",
                        help="只跑 5 条示例（快速演示）")
    parser.add_argument("--few_shot",    action="store_true",
                        help="每类加入 2 条训练集示例；默认使用 zero-shot")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    system_prompt = FEW_SHOT_SYSTEM_PROMPT if args.few_shot else SYSTEM_PROMPT
    mode = "few_shot" if args.few_shot else "zero_shot"
    mode_title = "Few-Shot" if args.few_shot else "Zero-Shot"
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2name = {int(k): v for k, v in label_map["id2name"].items()}
    name2id = {v: k for k, v in id2name.items()}

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    correct, total, unparseable = 0, 0, 0
    results = []
    t0 = time.time()

    for i, item in enumerate(samples):
        text     = item["sentence"]
        true_id  = item["label"]
        true_name = id2name[true_id]

        raw_output = classify_one(
            text, model, tokenizer, device, system_prompt=system_prompt
        )
        pred_name  = parse_prediction(raw_output)

        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "text":        text,
            "true_label":  true_name,
            "pred_label":  pred_name,
            "raw_output":  raw_output,
            "correct":     is_correct,
        })

        # 打印进度
        status = "✓" if is_correct else ("?" if pred_name is None else "✗")
        print(f"[{i+1:3d}/{len(samples)}] {status} "
              f"真实:{true_name:4s} 预测:{str(pred_name):4s} | {text[:35]}")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"{mode_title} LLM 分类结果（{Path(args.model_path).name}）")
    print(f"{'='*50}")
    print(f"  样本数   : {total}")
    print(f"  准确率   : {correct}/{total} = {acc:.4f}")
    print(f"  无法解析 : {unparseable} 条 ({unparseable/total*100:.1f}%)")
    print(f"  总耗时   : {elapsed:.1f}s, 均值 {elapsed/total:.2f}s/条")

    # ── 与 BERT fine-tune 的对比说明 ─────────────────────────────────────────
    print(f"""
对比参考（典型结果）：
  BERT fine-tune (3 epochs, cls)   val accuracy ≈ 0.57 ~ 0.62
  Qwen2-0.5B {mode_title.lower():9s}     val accuracy ≈ {acc:.2f}

思考题：
  1. LLM 在哪些类别上表现好？为什么？
  2. fine-tuning 数据量越多效果越好吗？试试只用 1000 条训练数据。
  3. 对比 zero-shot 与 few-shot，准确率和耗时分别变化多少？
""")

    # 保存结果
    out_path = ROOT / "outputs" / f"llm_{mode}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode, "examples_per_class": 2 if args.few_shot else 0,
            "accuracy": acc, "total": total, "correct": correct,
            "unparseable": unparseable, "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()

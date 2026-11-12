#region 0. 导入库
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
#endregion

#region 1. 路径常量   【常量:ROOT 项目根目录】
ROOT = Path(__file__).resolve().parent                       # sft_lab/
MODEL_DIR = ROOT / "models" / "Qwen3-0.6B-Base"              # 底座权重
DATA_DIR = ROOT / "data"                                     # 预处理后的数据集
OUT_DIR = ROOT / "outputs" / "lora_sft"                      # LoRA adapter 输出
ART_DIR = ROOT / "artifacts"                                 # 图表、指标、报告
SRC_TRAIN = ROOT.parent / "sft_train" / "sft_train.jsonl"    # 课程给的数据，只读不改
SRC_VAL = ROOT.parent / "sft_train" / "sft_val.jsonl"        # 验证集 200 条，与训练集无重叠
#endregion

#region 2. 对话模板   【常量:IM_START/IM_END Qwen3 的特殊 token】
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
IM_END_ID = 151645                                           # <|im_end|> 的 token id
PAD_ID = 151643                                              # <|endoftext|>
EOS_IDS = [IM_END_ID, PAD_ID]                                # 生成时两个都当停止符

# 回答用 <|endoftext|> 收尾，而不是 <|im_end|>：实测 <|im_end|> 在 Base 词表里几乎是空行
# （embedding 行范数 0.35，普通字约 0.97，<|endoftext|> 1.10），tied embedding 下它的 logit
# 推不动 —— 过拟合 32 条、15 个 epoch 后 loss 仍卡在 7.8。而 <|endoftext|> 是 base 本来就会
# 在句末预测的 token，换过来立刻学会。这就是「SFT 教不了模型一个它没见过的符号」。
ANSWER_END = "<|endoftext|>"

# 为什么手写而不用 tokenizer.chat_template：官方模板会往 assistant 段强塞
# <think>\n\n</think>\n\n，而 Base 模型没有思考行为，学了只是噪声，还造成训练/推理格式不一致。
def build_prompt(instr):
    """渲染提问段。返回：str，形如 <|im_start|>user\\n问题<|im_end|>\\n<|im_start|>assistant\\n。"""
    return f"{IM_START}user\n{instr}{IM_END}\n{IM_START}assistant\n"

def build_answer(out):
    """渲染回答段。返回：str，末尾带 <|endoftext|>\\n，让模型学会「说完就停」。"""
    return f"{out}{ANSWER_END}\n"
#endregion

#region 3. 函数：单条样本 → input_ids + labels（loss mask）   【函数:encode_example,return dict】
def encode_example(tok, instr, out, max_len=512):
    """编码一条样本：提问段 labels 全打 -100，只有回答段参与 loss。返回：dict(input_ids, labels, attention_mask)。"""
    p = tok(build_prompt(instr), add_special_tokens=False)["input_ids"]   # 提问段 token（不加 BOS）
    a = tok(build_answer(out), add_special_tokens=False)["input_ids"]     # 回答段 token，含 im_end
    if len(a) > max_len - 6:                                             # 兜底：极端长答案先保住收尾符
        a = a[: max_len - 6] + ([IM_END_ID] if a[-1] != IM_END_ID else [])
    keep = max_len - len(a)                                              # 留给提问段的额度
    if len(p) > keep:
        p = p[-keep:]                                                    # 砍左边而非右边：保住 "assistant\n" 角色头
    ids = p + a
    labels = [-100] * len(p) + list(a)                                   # -100 = 不算 loss（PyTorch 约定）
    assert sum(x != -100 for x in labels) > 0                            # 回答段被截空就当场报错，不许静默训练
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}
#endregion

#region 4. 模型与分词器加载
def get_tokenizer(padding_side="right"):
    """加载分词器。返回：AutoTokenizer；训练用 right，批量生成必须用 left。"""
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tok.pad_token = IM_END                  # 原 pad 与 eos 同为 151643，换成 im_end 避免生成告警
    tok.padding_side = padding_side
    return tok

def get_lora_config():
    """LoRA 配置：r=8/alpha=16，覆盖 7 个线性层，可训练参数约 4.59M（占 0.77%）。返回：LoraConfig。"""
    return LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

def load_base(adapter=None, trainable=False):
    """加载底座（可选挂 LoRA）。返回：model；trainable=True 时开梯度检查点相关开关。"""
    kw = dict(attn_implementation="sdpa", torch_dtype=torch.bfloat16)   # 5050 无 flash-attn，用 sdpa；4.55 只认 torch_dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)         # 挂上训练好的 adapter
    if trainable:
        model.config.use_cache = False                            # 与梯度检查点互斥
    return model.to("cuda")
#endregion

#region 5. 字符级指标   【说明:不依赖 jieba，全部按字符算】
PUNCT = "，。、！？；：,.!?;:\"'（）()《》【】 \n\t"          # 归一化时要去掉的标点与空白

def norm(s):
    """去掉标点与空白，只留实义字符。返回：str。"""
    return "".join(ch for ch in s if ch not in PUNCT)

def char_f1(pred, ref):
    """字符多重集 F1：不看语序，只看用了哪些字。返回：float，0~1。"""
    a, b = norm(pred), norm(ref)
    inter = sum((Counter(a) & Counter(b)).values())               # 共同字符（按出现次数取小）
    if not inter:
        return 0.0
    p, r = inter / max(1, len(b)), inter / max(1, len(a))         # 精确率用预测长度，召回率用参考长度
    return 2 * p * r / (p + r)

def rouge_l_char(pred, ref):
    """字符级 LCS 的 F1，补上 char_f1 不看语序的缺陷。返回：float，0~1。"""
    a, b = norm(pred), norm(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]          # dp[i][j]=a[:i] 与 b[:j] 的 LCS 长度
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r) if lcs else 0.0

def is_degenerate(text):
    """判断复读退化：任一 8-gram 出现 ≥3 次，或字符二元组多样性 <0.5。返回：bool。"""
    t = norm(text)
    if len(t) < 16:
        return False
    grams = [t[i:i + 8] for i in range(len(t) - 7)]
    if grams and max(Counter(grams).values()) >= 3:
        return True
    bigrams = [t[i:i + 2] for i in range(len(t) - 1)]
    return bool(bigrams) and len(set(bigrams)) / len(bigrams) < 0.5

def count_question_marks(text):
    """统计问号个数，用来抓「继续列问题」的续写行为。返回：int。"""
    return len(re.findall(r"[？?]", text))
#endregion

#region 6. 杂项
def read_jsonl(path):
    """读 jsonl。返回：list[dict]。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_json(obj, path):
    """写 json（自动建目录、UTF-8、不转义中文）。返回：None。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    """读 json。返回：dict。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def ppl_from_loss(loss):
    """交叉熵 → 困惑度。返回：float。"""
    return math.exp(loss)
#endregion

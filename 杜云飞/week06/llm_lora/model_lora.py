#region 0. 模块说明
"""
L4 的核心：LoRA SFT 的数据构造、模型加载与批量推理（train / predict 共用）。

SFT 样本构造四步（顺序不能乱）：
  1. `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` 拿到 prompt **文本**
     （`tokenize=False` 是必须的：transformers 5.x 不传会返回 dict）；
  2. `response = 类名 + eos`（学会"说完就停"，否则会继续编造理由）；
  3. `input_ids = prompt_ids + response_ids`，超长时**从左侧截掉 prompt**，
     保证 response 永远完整（截掉答案等于训练了一个永远不输出的模型）；
  4. `labels = [-100] * len(prompt_ids) + response_ids`，**只在 response 上算 loss**。

显存纪律（本机实测）：
  bs=2 / L=384 → 4.0GB / 12.1 samples/s（推荐）
  bs=4 / L=384 → 7.1GB / 6.1 samples/s（越线，慢一倍）
  bs=8 / L=256 → 8.8GB（已溢到共享内存，不报错，但慢 2 倍）
  所以本模块固定 bs=2 + 梯度累积，把显存压在 6.5GB 以下。

使用方式：
  from llm_lora.model_lora import SFTDataset, load_lora, batch_predict
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from common.prompt import build_messages
from common.paths import model_path

#endregion

#region 2. SFT 数据集
class SFTDataset(Dataset):
    """一条样本 = (prompt + 类名) 的 token 序列 + 只在 response 上有值的 labels。

    参数：
      recs       : 记录列表（含 title / content / label）
      tokenizer  : Qwen tokenizer
      max_len    : 序列上限（prompt+response 共享）
      max_chars  : 正文截断预算
      truncation : 截断策略（与其它方法统一）
    """

    def __init__(self, recs: list[dict], tokenizer, max_len: int = 384,
                 max_chars: int = 300, truncation: str = "head_tail") -> None:
        self.items: list[dict] = []
        eos = tokenizer.eos_token or ""
        for r in recs:
            msgs = build_messages(r, shots=0, max_chars=max_chars, truncation=truncation)
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                        add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            resp_ids = tokenizer(r["label"] + eos, add_special_tokens=False)["input_ids"]

            # 超长时从左侧砍 prompt，保住 response
            budget = max_len - len(resp_ids)
            if budget < 8:  # response 本身就超长（不该发生），强行截断
                resp_ids = resp_ids[: max(1, max_len - 8)]
                budget = max_len - len(resp_ids)
            prompt_ids = prompt_ids[-budget:]

            input_ids = prompt_ids + resp_ids
            labels = [-100] * len(prompt_ids) + resp_ids
            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.items[i]


def collate_sft(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """右侧 padding。三条规则：input_ids→pad_id，attention_mask→0，labels→-100。"""
    max_len = max(x["input_ids"].size(0) for x in batch)
    n = len(batch)
    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    am = torch.zeros((n, max_len), dtype=torch.long)
    lab = torch.full((n, max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        ids[i, :L] = x["input_ids"]
        am[i, :L] = 1
        lab[i, :L] = x["labels"]
    return {"input_ids": ids, "attention_mask": am, "labels": lab}
#endregion

#region 3. 模型加载
def load_base(alias: str = "qwen2-0.5b"):
    """加载基座模型与 tokenizer（fp16，sdpa）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = model_path(alias)
    tk = AutoTokenizer.from_pretrained(path)
    if tk.pad_token_id is None:
        tk.pad_token_id = tk.eos_token_id  # Qwen 系通常没有 pad_token，用 eos 兜底
    tk.padding_side = "left"  # 生成必须左 padding
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                 attn_implementation="sdpa")
    return tk, model


def attach_lora(model, r: int = 8, alpha: int | None = None, dropout: float = 0.05,
                targets: str = "qkvo"):
    """注入 LoRA 并打印可训练参数（**必须打印**：target_modules 写错不会报错，只会"训了等于没训"）。"""
    from peft import LoraConfig, TaskType, get_peft_model

    mapping = {"qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha or 2 * r,
                     lora_dropout=dropout, bias="none", target_modules=mapping[targets])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def load_lora(alias: str = "qwen2-0.5b", adapter_dir: str | None = None):
    """加载基座 + 已训好的 adapter（adapter_dir 为 None 时只返回基座）。"""
    from peft import PeftModel

    tk, model = load_base(alias)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"[LoRA] 已加载 adapter：{adapter_dir}")
    model.eval()
    return tk, model
#endregion

#region 4. 批量推理
@torch.inference_mode()
def batch_predict(tk, model, prompts: list[str], device, max_new_tokens: int = 8):
    """批量生成，返回 (新文本列表, 秒)。只解码 input_len 之后的部分。"""
    import time

    enc = tk(prompts, return_tensors="pt", padding=True, truncation=True,
             max_length=1024).to(device)
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, use_cache=True, pad_token_id=tk.pad_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    texts = tk.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return texts, dt
#endregion

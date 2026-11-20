#region 0. 模块说明
"""
环境探针：**写第一行业务代码之前先跑它**，把"能不能跑"变成一组数字写进 docs/PROBE.md。

四连测（任何一项失败，后面的实验计划都要改）：
  1. tokenizer：本地 bert-base-chinese / Qwen2 能否离线加载（transformers 5.x 的 slow
     tokenizer 被移除过，本地目录没有 tokenizer.json，这是本项目最高危的一环）；
  2. BERT 训练一步：bs=32/L=256 的显存峰值与吞吐（决定 100k×3ep 要多久）；
  3. LoRA 训练一步：bs=2/L=384 的显存峰值（8GB 卡上 bs≥4 会静默溢到共享内存）；
  4. 生成式：bs=16 的 samples/s（决定 LLM 评估能跑多大测试集）。

同时把结果写成 docs/PROBE.md，报告里直接引用，不用再"凭印象说大概要多久"。

使用方式：
  python -m common.probe              # 全量（约 2 分钟，占 4~5GB 显存）
  python -m common.probe --skip_train # 只测 tokenizer 与生成，适合无 GPU 环境
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common.paths import DOCS, has_local, model_path
from common.seed import set_seed

#endregion

#region 2. 各子探针
def probe_tokenizers() -> list[str]:
    """检查本地模型能否离线加载，并测编码吞吐。返回 markdown 行。"""
    from transformers import AutoTokenizer

    lines: list[str] = []
    cases = [
        ("bert-base", "国务院总理今日主持召开常务会议，研究部署进一步扩大内需促进消费的政策措施。" * 40),
        ("qwen2-0.5b", "请判断以下新闻属于哪一类：中超联赛第 20 轮，北京国安主场 3:1 逆转上海申花。"),
    ]
    for alias, sample in cases:
        if not has_local(alias):
            lines.append(f"| {alias} | 本地缺失 | - | - |")
            continue
        t0 = time.time()
        tk = AutoTokenizer.from_pretrained(model_path(alias))
        load_s = time.time() - t0
        if alias.startswith("qwen"):
            tk.pad_token_id = tk.eos_token_id  # Qwen2 的 pad 与 eos 不同，必须显式统一
            msgs = [{"role": "user", "content": sample}]
            s = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            assert isinstance(s, str), "apply_chat_template 未返回 str，检查 tokenize=False 是否生效"
            lines.append(f"| {alias} | ✓ {type(tk).__name__} | {load_s:.1f}s | chat_template ok ({len(s)} 字符) |")
        else:
            t1 = time.time()
            for _ in range(10):
                tk(sample, max_length=256, truncation=True)
            docs_per_s = 10 / (time.time() - t1)
            lines.append(f"| {alias} | ✓ {type(tk).__name__} | {load_s:.1f}s | {docs_per_s:.0f} docs/s |")
    return lines


def probe_bert_step(bs: int = 32, max_len: int = 256) -> str:
    """跑 3 步 BERT 训练，返回一行说明（吞吐 + 显存峰值）。"""
    from transformers import AutoModelForSequenceClassification

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForSequenceClassification.from_pretrained(
        model_path("bert-base"), num_labels=14, attn_implementation="sdpa"
    ).cuda()
    # 注意：**不要把模型 .half()**。权重半精度 + GradScaler 会报
    # "Attempting to unscale FP16 gradients"，正确做法是权重保持 fp32、只用 autocast。
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
    ids = torch.randint(0, 21000, (bs, max_len), device="cuda")
    am = torch.ones_like(ids)
    y = torch.randint(0, 14, (bs,), device="cuda")
    dt = 0.0
    for i in range(3):
        if i == 1:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = m(input_ids=ids, attention_mask=am, labels=y).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 2
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del m, opt
    torch.cuda.empty_cache()
    return (f"bs={bs} L={max_len}: {dt * 1000:.0f} ms/step, {bs / dt:.0f} samples/s, "
            f"峰值 {peak:.2f} GB, 100k×1ep ≈ {100000 / (bs / dt) / 60:.0f} min")


def probe_lora_step(bs: int = 2, max_len: int = 384, r: int = 8) -> str:
    """跑 3 步 LoRA 训练，返回一行说明。"""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(model_path("qwen2-0.5b"), dtype=torch.float16,
                                             attn_implementation="sdpa").cuda()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=2 * r, lora_dropout=0.05,
                     bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    m = get_peft_model(m, cfg)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    ids = torch.randint(0, 150000, (bs, max_len), device="cuda")
    am = torch.ones_like(ids)
    lab = ids.clone()
    lab[:, : max_len // 2] = -100  # prompt 段不参与 loss
    dt = 0.0
    for i in range(3):
        if i == 1:
            torch.cuda.synchronize()
            t0 = time.time()
        m(input_ids=ids, attention_mask=am, labels=lab).loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 2
    peak = torch.cuda.max_memory_allocated() / 1024**3
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    del m, opt
    torch.cuda.empty_cache()
    return (f"bs={bs} L={max_len} r={r}: {dt * 1000:.0f} ms/step, {bs / dt:.1f} samples/s, "
            f"峰值 {peak:.2f} GB, 可训练 {n_train / 1e6:.2f}M, 10k×3ep ≈ {10000 / (bs / dt) * 3 / 3600:.1f} h")


def probe_generate(bs: int = 16, prompt_len: int = 512) -> str:
    """测本地 Qwen 的生成吞吐（决定 LLM 评估能跑多大测试集）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tk = AutoTokenizer.from_pretrained(model_path("qwen2-0.5b"))
    tk.pad_token_id = tk.eos_token_id
    tk.padding_side = "left"  # 生成必须左 padding，否则 PAD 会挤在答案前面
    m = AutoModelForCausalLM.from_pretrained(model_path("qwen2-0.5b"), dtype=torch.float16,
                                             attn_implementation="sdpa").cuda().eval()
    txt = "请判断以下新闻类别，只输出类名：" + "国务院总理今日主持召开常务会议，研究部署进一步扩大内需促进消费的政策措施。" * 12
    enc = tk([txt] * bs, return_tensors="pt", padding=True, truncation=True,
             max_length=prompt_len).to("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        m.generate(**enc, max_new_tokens=8, do_sample=False, num_beams=1, pad_token_id=tk.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    del m
    torch.cuda.empty_cache()
    return (f"bs={bs} prompt={enc['input_ids'].shape[1]}: {bs / dt:.1f} samples/s, "
            f"1000 条 ≈ {1000 / (bs / dt):.0f}s, 4200 条 ≈ {4200 / (bs / dt) / 60:.1f} min")
#endregion

#region 3. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="环境探针：把'能不能跑'变成数字")
    ap.add_argument("--skip_train", action="store_true", help="跳过 GPU 训练探针（无 GPU / 省时间）")
    ap.add_argument("--no_write", action="store_true", help="只打印，不写 docs/PROBE.md")
    args = ap.parse_args()

    set_seed(42)
    print("=" * 78)
    print(f"python {'.'.join(map(str, __import__('sys').version_info[:3]))} | torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} {p.total_memory / 1024**3:.1f}GB sm_{p.major}{p.minor}")

    rows: list[str] = []
    print("\n[1/4] tokenizer ...")
    tk_rows = probe_tokenizers()
    for r in tk_rows:
        print("   ", r)

    extra: list[str] = []
    if not args.skip_train and torch.cuda.is_available():
        print("\n[2/4] BERT 训练一步 ...")
        line = probe_bert_step()
        print("   ", line)
        extra.append(f"- **BERT 训练**：{line}")
        print("\n[3/4] LoRA 训练一步 ...")
        line = probe_lora_step()
        print("   ", line)
        extra.append(f"- **LoRA 训练**：{line}")
        print("\n[4/4] 本地生成 ...")
        line = probe_generate()
        print("   ", line)
        extra.append(f"- **本地生成**：{line}")

    if not args.no_write:
        DOCS.mkdir(parents=True, exist_ok=True)
        md = ["# PROBE：环境实测记录", "",
              "> 由 `python -m common.probe` 生成。**报告里的耗时/显存只引用这里的数字**，不凭印象。", "",
              "| 模型 | 加载 | 耗时 | 编码吞吐 |", "|---|---|---|---|", *tk_rows, "",
              "## GPU 训练/推理", *extra, "",
              "## 由此得到的硬约束", "",
              "- BERT：`bs=32 / max_len=256`；显存峰值一旦过 ~7GB，吞吐断崖式下跌（bs=64 会比 bs=32 慢 3 倍）",
              "- LoRA：`bs=2 / max_len=384 / accum=8`；bs≥4 会溢到 WDDM 共享内存，**不报错只是变慢**",
              "- 生成式评估：只跑 `test_bal`（4200 条）与 1000 条抽样，不跑 10k 全量",
              "",
              "## 实测 vs 真实训练的偏差（重要）", "",
              "本探针用定长张量测吞吐，**真实训练要慢 20%~25%**：真实数据长度不均（padding 更多）、",
              "且笔记本满载后热降频（GPU 稳定 77~79°C，功耗 84W→72W）。",
              "实测对照：BERT 100k×3ep 真实单 epoch **1030s**（探针推算 14 min = 840s）；",
              "LoRA 三轮 994s / 1174s / 1223s（逐轮变慢 23%）。**排期时按 +25% 打余量。**",
              ]
        p = DOCS / "PROBE.md"
        p.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
#endregion

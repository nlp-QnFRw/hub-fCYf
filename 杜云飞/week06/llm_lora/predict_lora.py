#region 0. 模块说明
"""
对已训好的 LoRA adapter 单独出预测（不重新训练）。

为什么单独留一个脚本：
  训练脚本里出预测是为了"训完顺手看一眼"，但实际调参时经常需要
  "拿 3 个 epoch 前的 adapter 再评一遍""换解析口径重评"——重训一遍太贵。
  本脚本只读 `runs/<run_id>/adapter/`，靠目录里有没有 `adapter_config.json` 判断类型，
  **不靠命令行开关**（规范 §3.7）。

使用方式：
  python llm_lora/predict_lora.py --adapter runs/t3__lora__10k-r8-3ep-qkvo/adapter \
      --run_id t3__lora__10k-r8-3ep-qkvo --splits test_bal
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import write_preds, append_index  # noqa: E402
from common.data import load_split  # noqa: E402
from common.prompt import build_messages, parse_label  # noqa: E402
from llm_lora.model_lora import batch_predict, load_lora  # noqa: E402

#endregion

#region 2. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="用已训好的 adapter 出预测")
    ap.add_argument("--adapter", required=True, help="adapter 目录或 run 目录")
    ap.add_argument("--model", default="qwen2-0.5b")
    ap.add_argument("--splits", nargs="+", default=["test_bal", "test_nat_10k"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max_chars", type=int, default=300)
    ap.add_argument("--truncation", default="head_tail")
    ap.add_argument("--run_id", default=None, help="写入哪个 run 目录（默认与 adapter 同级）")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if (adapter / "adapter_config.json").exists():
        run_id = args.run_id or adapter.parent.name  # adapter 目录的上一级就是 run_id
    else:
        # 传的是 run 目录，往里找 adapter/
        if (adapter / "adapter" / "adapter_config.json").exists():
            adapter = adapter / "adapter"
            run_id = args.run_id or adapter.parent.name
        else:
            sys.exit(f"{args.adapter} 里没有 adapter_config.json，不是 LoRA 目录")

    print(f"adapter = {adapter}\nrun_id  = {run_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk, model = load_lora(args.model, str(adapter))
    model = model.to(device)

    for split in args.splits:
        recs = load_split(split, limit=args.n)
        preds: list[dict] = []
        t0 = time.time()
        for i in range(0, len(recs), 16):
            chunk = recs[i: i + 16]
            prompts = [tk.apply_chat_template(
                build_messages(r, shots=0, max_chars=args.max_chars, truncation=args.truncation),
                tokenize=False, add_generation_prompt=True) for r in chunk]
            texts, dt = batch_predict(tk, model, prompts, device)
            per_ms = dt / max(1, len(chunk)) * 1000
            for r, txt in zip(chunk, texts):
                pred, status = parse_label(txt, mode="lenient")
                preds.append({"id": r["id"], "gold": r["label"], "pred": pred,
                              "raw": txt.replace("\n", " "), "status": status,
                              "latency_ms": per_ms})
        write_preds(run_id, split, preds)
        acc = sum(1 for p in preds if p["pred"] == p["gold"]) / max(1, len(preds))
        print(f"  {split}: {len(preds)} 条 acc={acc:.4f} "
              f"{len(recs) / max(1e-6, time.time() - t0):.1f} 条/s")

    append_index(run_id, {"method": "lora-predict"})
    print(f"\n下一步：python eval/evaluate.py --run {run_id}")


if __name__ == "__main__":
    main()
#endregion

#region 0. 导入库
import sys

import torch

from common import MODEL_DIR, SRC_TRAIN, SRC_VAL, build_prompt, get_lora_config, get_tokenizer, read_jsonl
#endregion

#region 1. 常量
EXPECT_SIZE = {"model.safetensors": 1_192_135_096}          # 与下载脚本同一判据
#endregion

#region 2. 函数：逐项体检，全绿才开跑   【函数:main,return 0 可开跑 / 1 有问题】
def main():
    """打印环境红绿灯。返回：0 表示可以开跑，1 表示有问题。"""
    ok = True
    print("【GPU】", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "不可用")
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    print(f"  算力 sm_{cap[0]}{cap[1]}，显存 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)               # 实测 bf16 能不能算
    print(f"  bf16 矩阵乘: {(x @ x).dtype}  torch {torch.__version__}  cuda {torch.version.cuda}")
    ok &= bool(torch.isfinite(x @ x).all())

    print("【模型文件】", MODEL_DIR)
    for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        p = MODEL_DIR / f
        size = p.stat().st_size if p.exists() else 0
        good = size > 0 and size == EXPECT_SIZE.get(f, size)                     # 大文件还要对上字节数
        print(f"  {'OK ' if good else 'BAD'} {f:22s}{size:>14,d}")
        ok &= good

    print("【数据】")
    for name, path in (("train", SRC_TRAIN), ("val", SRC_VAL)):
        n = len(read_jsonl(path)) if path.exists() else 0
        print(f"  {name}: {n} 条")
        ok &= n > 0

    if (MODEL_DIR / "tokenizer_config.json").exists():
        tok = get_tokenizer()
        probe = tok.apply_chat_template([{"role": "user", "content": "什么是人工智能？"},
                                         {"role": "assistant", "content": "人工智能是……"}], tokenize=False)
        print(f"【chat template】存在={tok.chat_template is not None}，"
              f"是否偷偷注入 <think>={('<think>' in probe)}  → 本项目手写 ChatML，不用它")
        print("  手写渲染:", repr(build_prompt("什么是人工智能？")))
        print("【LoRA 目标层】", get_lora_config().target_modules)
    return 0 if ok else 1
#endregion

if __name__ == "__main__":
    sys.exit(main())

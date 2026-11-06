"""任务 5：离线加载已验收模型，展示单个遮盖位置的填空候选。

参考老师 export_bert_viz.py 的“遮盖位置 → 概率 → 前若干候选”流程，
复用 train_bert_mlm.py 的本地加载、单精度推理和完整权重指纹函数。
本脚本没有训练入口；默认只在终端显示，指定 --output 才写入新结果。
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
PY312 = Path("D:/Anaconda3/envs/py312")
DEFAULT_MODEL = (HERE / "outputs" / "task04_train_20261105_141502_537948"
                 / "best_checkpoints" / "update_002500" / "model")
MAX_LENGTH = 64
# 在运行前固定的教学展示，不取用独立测试集，不据此重新选择模型。
DEMO_TEXTS = (
    "中国的首都是北[MASK]。",
    "我喜欢吃[MASK]果。",
    "他把书放在桌子[MASK]，然后就离开了。",
    "看我的新书包好[MASK]吗。",
    "如何打开屏蔽的集成显[MASK]",
)


def prepare_input(tokenizer, text, top_k):
    """输入一条文本，返回 [1, 长度] 的模型输入和唯一遮盖位置。"""
    if not 1 <= top_k <= len(tokenizer):
        raise ValueError(f"候选数必须在 1 至 {len(tokenizer)} 之间。")
    if text.count("[MASK]") != 1:
        raise ValueError("每条文本必须恰好包含一个大写的 [MASK] 遮盖标记。")
    # 不静默截断，防止丢失遮盖标记或改变用户希望保留的上下文。
    batch = tokenizer(text, return_tensors="pt", truncation=False)
    length = batch["input_ids"].shape[1]
    if length > MAX_LENGTH:
        raise ValueError(f"分词后长度为 {length}，超过课程训练长度 {MAX_LENGTH}（含首尾标记）。")
    positions = (batch["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten()
    if positions.numel() != 1:
        raise ValueError("分词后必须恰好存在一个遮盖位置。")
    # 只允许用户提供作为填空的遮盖标记，其他特殊标记会改变输入语义。
    other_special = set(tokenizer.all_special_ids) - {tokenizer.mask_token_id}
    if any(int(token_id) in other_special for token_id in batch["input_ids"][0, 1:-1]):
        raise ValueError("文本包含其他特殊标记或无法识别的词元，请改用可识别的普通文本。")
    return batch, int(positions[0])


def predict_one(model, tokenizer, batch, position, text, top_k, device, helpers):
    """输出 [1, 长度, 词表大小]，只取遮盖位置的整词表概率排名。"""
    torch = helpers["torch"]
    logits = helpers["reference_logits"](model, batch, device)
    if not torch.isfinite(logits).all():
        raise RuntimeError("模型输出存在非有限数值，停止展示。")
    probabilities = logits[0, position].softmax(dim=-1)
    values, ids = probabilities.topk(top_k)
    candidates = []
    for rank, (value, token_id) in enumerate(zip(values.tolist(), ids.tolist()), 1):
        token = tokenizer.convert_ids_to_tokens(token_id)
        candidates.append({
            "rank": rank, "token_id": token_id, "token": token,
            "probability": value,
            "filled_text": text.replace("[MASK]", token, 1),
        })
    return {
        "input": text, "input_ids": batch["input_ids"][0].tolist(),
        "tokens": tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist()),
        "mask_position": position, "input_shape": list(batch["input_ids"].shape),
        "logits_shape": list(logits.shape), "candidates": candidates,
    }


def output_path(value):
    """输出只能是本周 outputs/ 下尚未存在的结果文件，禁止覆盖。"""
    if value is None:
        return None
    path = value.resolve()
    if not path.is_relative_to((HERE / "outputs").resolve()) or path.suffix.lower() != ".json":
        raise ValueError("--output 必须是 week04/outputs/ 下的新 .json 文件。")
    if path.exists():
        raise ValueError(f"结果文件已存在，禁止覆盖：{path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="含一个 [MASK] 的待填空句子")
    mode.add_argument("--demo", action="store_true", help="运行五条固定教学示例")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL, help="本地模型目录")
    parser.add_argument("--top-k", type=int, default=5, help="候选数，默认 5")
    parser.add_argument("--compare-base", action="store_true", help="同时展示原始预训练模型结果")
    parser.add_argument("--output", type=Path, help="可选：本周 outputs/ 下的新结果文件")
    args = parser.parse_args()
    try:
        if Path(sys.prefix).resolve() != PY312.resolve():
            raise ValueError("本项目必须使用 D:/Anaconda3/envs/py312/python.exe。")
        if args.top_k < 1:
            raise ValueError("候选数必须大于或等于 1。")
        destination = output_path(args.output)
        model_path = args.model_dir.resolve()
        if not model_path.is_dir():
            raise ValueError(f"本地模型目录不存在：{model_path}")
        # -I 会移除脚本目录的默认导入路径，因此按明确文件路径复用已有函数。
        # 此名称不会触发训练脚本的 __main__ 入口，不运行训练、不生成文件。
        helpers = runpy.run_path(str(HERE / "train_bert_mlm.py"))
        torch = helpers["torch"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = helpers["load_local_tokenizer"](model_path)
        texts = list(DEMO_TEXTS) if args.demo else [args.text]
        prepared = [prepare_input(tokenizer, text, args.top_k) for text in texts]
        sources = [("selected", "当前指定模型", model_path)]
        if args.compare_base:
            sources.insert(0, ("pretrained", "原始预训练模型", helpers["DEFAULT_MODEL"]))
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable, "device": str(device), "precision": "float32",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_executed": False, "top_k": args.top_k, "max_length": MAX_LENGTH,
            "demo": args.demo, "models": [],
            "note": "单遮盖教学演示，不是验证集或独立测试集评分；概率不代表事实正确性。候选按完整词表排名，未过滤特殊词元。",
        }
        for key, label, path in sources:
            source_tokenizer = helpers["load_local_tokenizer"](path)
            if source_tokenizer.get_vocab() != tokenizer.get_vocab():
                raise ValueError("对照模型词表编号不一致，不能直接复用输入。")
            for text, (batch, _) in zip(texts, prepared):
                other_batch, _ = prepare_input(source_tokenizer, text, args.top_k)
                if any(not torch.equal(value, other_batch[name]) for name, value in batch.items()):
                    raise ValueError("对照模型分词结果不一致，停止直接比较。")
            print(f"\n{label}：{path}", flush=True)
            model, loading = helpers["load_local_model"](path, device)
            model.eval()
            before = helpers["model_state_fingerprint"](model)
            results = [predict_one(model, source_tokenizer, batch, position, text,
                                   args.top_k, device, helpers)
                       for text, (batch, position) in zip(texts, prepared)]
            after = helpers["model_state_fingerprint"](model)
            no_gradients = all(parameter.grad is None for parameter in model.parameters())
            if before != after or not no_gradients:
                raise RuntimeError("推理前后权重或梯度检查失败。")
            for result in results:
                print(f"\n输入：{result['input']}")
                for candidate in result["candidates"]:
                    print(f"  {candidate['rank']}. {candidate['token']}  "
                          f"{candidate['probability']:.4%}  {candidate['filled_text']}")
            report["models"].append({
                "kind": key, "directory": str(path), "loading": loading,
                "state_sha256": before["sha256"], "parameters_unchanged": before == after,
                "no_parameter_gradients": no_gradients, "results": results,
            })
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"\n结果文件：{destination}")
        print("\n推理完成；权重未改变，未生成参数梯度，未执行训练。")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()

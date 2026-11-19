"""
对比实验共用模块：数据集注册表 + 预训练模型解析 + 结果落盘

与原项目的关系（重要）：
  - 本文件夹是 week6-数据接入与分流/code 的**兄弟目录**，原目录零改动；
  - 通过 sys.path 只读复用原项目的 common（data / labels / metrics / model_paths）；
  - 所有结果只写进本文件夹的 outputs/，绝不写回原目录。

「换数据集」怎么换：
  - tnews15  原始 TNEWS 15 类（直接读原项目 data/，只读）
  - tnews4   同一批句子，把 15 类按原 labels.py 的 GROUP_MAP 重映射成 4 组主题
             —— 零下载就能做「换标签口径」的对照实验
  - 其他     在 exp_data/<name>/ 下放 train.json / val.json（格式 {"sentence","label"}，
             可选 labels.json 列类别名），load_dataset("<name>") 即可用

「换模型」怎么换：
  --model_name 既接受本地目录名（先在 pretrain_models 下找，离线优先），
  也接受 HF repo id（本地没有就联网下载，国内网络可设 HF_ENDPOINT=https://hf-mirror.com）
"""

import json
import os
import sys
from pathlib import Path

# 对原目录完全只读：不让 Python 在原项目里写 __pycache__ 字节码缓存
sys.dont_write_bytecode = True

EXP_ROOT = Path(__file__).resolve().parent
EXP_OUT = EXP_ROOT / "outputs"
EXP_DATA = EXP_ROOT / "exp_data"

# 只读复用原项目的 common/（可用环境变量 WEEK6_ORIG_CODE 指定其他位置）
ORIG_CODE = Path(os.environ.get("WEEK6_ORIG_CODE")
                 or (EXP_ROOT.parent / "week6-数据接入与分流" / "code"))
if not ORIG_CODE.is_dir():
    raise FileNotFoundError(
        f"找不到原项目目录 {ORIG_CODE}\n"
        "本对比实验只读复用原项目的 common/（data/labels/metrics/model_paths），\n"
        "请把本文件夹放在 week6-数据接入与分流 的同级目录，或设置环境变量 WEEK6_ORIG_CODE 指向其 code 目录。")
sys.path.insert(0, str(ORIG_CODE))

BUILTIN_DATASETS = ("tnews15", "tnews4")


def load_dataset(name, n=None, seed=42):
    """
    返回 (train, eval_samples, labels, meta)。
    eval_samples 是「去重后的干净验证子集」（沿用原 common/data.py 的口径，seed 固定），
    各方法必须用它评测，数字才可比。
    """
    from common.data import clean_val, load_splits, sample_subset
    from common.labels import LABELS

    if name == "tnews15":
        train, val = load_splits()
        labels = list(LABELS)
    elif name == "tnews4":
        from common.labels import GROUP_MAP, GROUPS
        train, val = load_splits()
        g2i = {g: i for i, g in enumerate(GROUPS)}
        remap = [g2i[GROUP_MAP[lb]] for lb in LABELS]      # 15 类 id → 4 组 id
        train = [dict(d, label=remap[d["label"]]) for d in train]
        val = [dict(d, label=remap[d["label"]]) for d in val]
        labels = list(GROUPS)
    else:
        ddir = EXP_DATA / name
        if not (ddir / "train.json").is_file():
            raise FileNotFoundError(
                f"数据集 {name} 不存在：内置只有 {BUILTIN_DATASETS}，"
                f"或把自备数据放到 {ddir}/（train.json / val.json / 可选 labels.json）")
        train = json.loads((ddir / "train.json").read_text(encoding="utf-8"))
        val = json.loads((ddir / "val.json").read_text(encoding="utf-8"))
        lp = ddir / "labels.json"
        labels = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() \
            else [str(i) for i in range(max(d["label"] for d in train + val) + 1)]

    clean, n_dup = clean_val(train, val)
    samples = sample_subset(clean, n or len(clean), seed)
    meta = {"dataset": name, "labels": labels, "num_labels": len(labels),
            "n_train": len(train), "n_val": len(val), "n_dup": n_dup,
            "n_eval": len(samples), "seed": seed}
    return train, samples, labels, meta


def resolve_model(name):
    """
    本地优先：依次在 <本文件夹>/pretrain_models、环境变量 PRETRAIN_DIR、
    原项目的 pretrain_models 下找（目录名支持带或不带组织前缀）。
    都没有 → 原样返回 HF repo id，由 transformers 联网下载。
    返回 (path_or_id, 来源)
    """
    from common.model_paths import resolve
    candidates = [name] + ([name.rsplit("/", 1)[-1]] if "/" in name else [])
    roots = [EXP_ROOT / "pretrain_models"]
    if os.environ.get("PRETRAIN_DIR"):
        roots.append(Path(os.environ["PRETRAIN_DIR"]))
    for c in candidates:
        try:
            return str(resolve(c, extra_roots=roots)), "local"
        except FileNotFoundError:
            continue
    return name, "hub"


def model_tag(name):
    """HF repo id → 文件名安全片段（hfl/chinese-roberta-wwm-ext → hfl__chinese-roberta-wwm-ext）"""
    return name.replace("/", "__")


def result_path(dataset, tag, method, ext="pred.json"):
    """统一命名：<数据集>_<模型>_<方法>_<ext>，eval_all.py 靠它汇总"""
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    return EXP_OUT / f"{dataset}_{tag}_{method}_{ext}"


def save_result(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    # 自检：python common_exp.py
    print(f"原项目 code 目录：{ORIG_CODE}（存在={ORIG_CODE.is_dir()}）")
    print(f"结果目录：{EXP_OUT}")
    train, samples, labels, meta = load_dataset("tnews15", n=100)
    print("tnews15:", {k: v for k, v in meta.items() if k != "labels"}, f"标签数={len(labels)}")
    _, samples4, labels4, meta4 = load_dataset("tnews4", n=100)
    print("tnews4 :", {k: v for k, v in meta4.items() if k != "labels"}, f"标签={labels4}")
    print("抽样一致性（同 seed 两批 sentence 应完全相同）：",
          [d["sentence"] for d in samples] == [d["sentence"] for d in samples4])

"""
统一汇总：把本文件夹 outputs/ 里所有 *_pred.json 拉成对比表 —— 改编自原 eval/eval_all.py

改动点：
  1. 不再写死七份文件名，自动扫描 outputs/ 下所有结果；
     每份结果自带的 meta.dataset / report.model / report.method 决定它在表里的位置；
  2. 出两张视角：①同一数据集上「换方法」横向对比 ②同一方法上「换模型/换数据集」对照；
  3. 沿用原版口径检查：样本数 / seed 不一致就报警——这是数字不可比最常见的原因。

用法：
  python eval_all.py                        # 汇总已有结果
  python eval_all.py --run-light            # 先把秒级的规则 + TF-IDF 各数据集跑一遍再汇总
  python eval_all.py --dataset tnews4       # 只看某个数据集
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_exp import EXP_OUT   # noqa: E402

METHOD_NAMES = {
    "rule_keyword": "规则/关键词",
    "tfidf_svm": "TF-IDF+SVM",
    "bert_finetune": "BERT 微调",
    "llm_zero_shot": "LLM zero-shot",
    "llm_lora": "LLM LoRA SFT",
    "llm_full_ft": "LLM 全量微调",
}


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] {path.name} 解析失败：{e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-light", action="store_true", help="先跑规则 + TF-IDF（秒级）")
    ap.add_argument("--dataset", default=None, help="只汇总某个数据集")
    args = ap.parse_args()

    if args.run_light:
        for script in ("run_rule.py", "run_tfidf.py"):
            for ds in (args.dataset,) if args.dataset else ("tnews15", "tnews4"):
                print(f"→ 运行 {script} --dataset {ds}")
                subprocess.run([sys.executable, str(Path(__file__).parent / script),
                                "--dataset", ds], check=False)

    rows, metas = [], {}
    for path in sorted(EXP_OUT.glob("*_pred.json")):
        data = load(path)
        if not data:
            continue
        r, m = data.get("report", {}), data.get("meta", {})
        ds, model, method = m.get("dataset", "?"), r.get("model", "?"), r.get("method", path.stem)
        if args.dataset and ds != args.dataset:
            continue
        metas[f"{ds}|{method}|{model}"] = m
        dist = r.get("pred_distribution") or {}
        rows.append({
            "dataset": ds, "method": METHOD_NAMES.get(method, method), "model": model,
            "acc": r.get("acc_incl_unparsed", r.get("acc", "-")),
            "macro_f1": r.get("macro_f1", "-"),
            "n": m.get("n_eval", r.get("n")),
            "used_classes": dist.get("used_classes", "-"),
            "unparsed": r.get("unparsed", 0),
            "fit_or_speed": r.get("fit_seconds") or r.get("sec_per_item") or r.get("ms_per_sample") or "-",
        })

    if not rows:
        print(f"outputs/ 里还没有结果文件，先跑：python eval_all.py --run-light")
        return

    # 口径检查：同一数据集上各方法的样本数 / seed 必须一致，否则不可比
    for ds in {r["dataset"] for r in rows}:
        group = [m for k, m in metas.items() if k.startswith(ds + "|")]
        sizes = {m.get("n_eval") for m in group}
        seeds = {m.get("seed") for m in group}
        if len(sizes) > 1:
            print(f"[warn] {ds} 各方法样本量不一致：{sizes}，结果不可直接比较")
        if len(seeds) > 1:
            print(f"[warn] {ds} 各方法 seed 不一致：{seeds}，抽样集合可能不同")

    def render(title, items):
        lines = [f"## {title}", "",
                 "| 数据集 | 方法 | 模型 | 准确率(含失败) | Macro-F1 | 样本数 | 覆盖类别 | 解析失败 | 耗时/速度 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in items:
            lines.append(f"| {r['dataset']} | {r['method']} | {r['model']} | {r['acc']} | "
                         f"{r['macro_f1']} | {r['n']} | {r['used_classes']} | {r['unparsed']} | {r['fit_or_speed']} |")
        return "\n".join(lines) + "\n"

    by_ds = sorted(rows, key=lambda r: (r["dataset"], r["method"], str(r["model"])))
    out_md = [render("方法 × 模型 × 数据集 总表", by_ds)]

    # 视角二：固定 tnews15 的 BERT，只换预训练模型 —— 「换模型值几个点」一眼看出
    same_ds = [r for r in by_ds if r["dataset"] == "tnews15"]
    bert_rows = [r for r in same_ds if "微调" in r["method"]]
    if len({str(r["model"]) for r in bert_rows}) > 1:
        out_md.append(render("换预训练模型对照（tnews15 + BERT 微调）", bert_rows))

    print("\n" + out_md[0])
    summary = EXP_OUT / "summary.md"
    summary.write_text("\n".join(out_md) + "\n口径：所有方法在同一批「去重后的干净验证样本」上评测（seed=42）。\n",
                       encoding="utf-8")
    print(f"→ 汇总写入 {summary}")


if __name__ == "__main__":
    main()

"""
可选工具：把 HuggingFace 上的分类数据集转成本地 json，作为「换数据集」实验素材。
转换结果写到 exp_data/<name>/（train.json / val.json / labels.json），
之后所有 run_*.py 直接 --dataset <name> 即可，无需再改代码。

需要：pip install datasets   （国内网络建议先 export HF_ENDPOINT=https://hf-mirror.com）

用法：
  python make_custom_dataset.py                          # 默认拉 CLUE 的 iflytek（119 类 APP 描述）
  python make_custom_dataset.py --hf clue --config iflytek --name iflytek
  python make_custom_dataset.py --hf my_own_repo --train_split train --val_split validation \
         --text_field sentence --label_field label --name mydata --max_val 10000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_exp import EXP_DATA   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="clue", help="HF 数据集 repo id")
    ap.add_argument("--config", default="iflytek", help="数据集子配置（没有就传空字符串）")
    ap.add_argument("--name", default="iflytek", help="保存目录名 = 之后 --dataset 用的名字")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="validation")
    ap.add_argument("--text_field", default="sentence")
    ap.add_argument("--label_field", default="label")
    ap.add_argument("--max_val", type=int, default=10000, help="验证集最多保留条数")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.hf, args.config or None)
    out = EXP_DATA / args.name
    out.mkdir(parents=True, exist_ok=True)

    def dump(split, fname):
        rows = [{"sentence": r[args.text_field], "label": int(r[args.label_field])}
                for r in ds[split]]
        if fname == "val.json":
            rows = rows[: args.max_val]
        (out / fname).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return len(rows)

    n_tr = dump(args.train_split, "train.json")
    n_va = dump(args.val_split, "val.json")

    # 类别名：数据集自带 ClassLabel 就存下来，否则退化为数字标签
    names = None
    feats = ds[args.val_split].features
    label_feat = feats.get(args.label_field)
    if hasattr(label_feat, "names"):
        names = list(label_feat.names)
    if names:
        (out / "labels.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")

    print(f"[{args.name}] train={n_tr} val={n_va} 类别数={len(names) if names else '（未提供，用数字）'}")
    print(f"→ 已写入 {out}，现在可以：python run_tfidf.py --dataset {args.name}")


if __name__ == "__main__":
    main()

# 分类任务对比实验（换预训练模型 / 换数据集）

> 在 [week6-数据接入与分流](../week6-数据接入与分流/code) 的基础上改编：**原目录零改动**（只读引用它的
> `common/` 与 `data/`），本文件夹所有脚本的结果只写进自己的 `outputs/`。
>
> ⚠️ 运行前提：本文件夹必须与 `week6-数据接入与分流` 在同一父目录下（或设环境变量
> `WEEK6_ORIG_CODE` 指向原项目 code 目录），否则脚本报「找不到原项目目录」。
>
> 对比实验 = 三条轴各拧一下，看哪个指标动：
> ① 方法轴（规则 / TF-IDF / BERT / LLM zero-shot / LoRA / 全量微调）
> ② 模型轴（`--model_name` / `--base_model` 换预训练模型）
> ③ 数据集轴（`--dataset` 换数据 / 换标签口径）

## 文件一览

```
对比实验/
├── common_exp.py          # 共用：数据集注册表 + 模型解析（本地优先）+ 统一命名落盘
├── run_rule.py            # 方法一：规则/关键词（改编自 rule/classify_rule.py）
├── run_tfidf.py           # 方法二：TF-IDF + LinearSVC（改编自 tfidf/classify_tfidf.py）
├── run_bert.py            # 方法三：BERT 系微调（改编自 bert/train_bert.py + predict_bert.py）★换模型主入口
├── run_llm_zero_shot.py   # 方法四：LLM 零样本（改编自 llm_zero_shot/）
├── run_llm_lora.py        # 方法五：LLM LoRA / 全量微调（改编自 llm_lora/）
├── eval_all.py            # 汇总：自动扫描 outputs/ 拉对比表
├── make_custom_dataset.py # 可选：从 HF 拉第二个数据集转成本地 json
├── exp_data/              # 自备/下载数据集放这里（每个子目录 train.json + val.json）
└── outputs/               # 所有结果只写这里：<数据集>_<模型>_<方法>_pred.json
```

## 依赖与模型

```bash
pip install torch transformers peft scikit-learn datasets tqdm   # datasets 仅换数据集时需要
```

- 模型解析**本地优先**：先找 `对比实验/pretrain_models/`、环境变量 `PRETRAIN_DIR`、原项目的 `pretrain_models/`；
  都没有就按 HF repo id 联网下载（国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`）。
- 两个内置数据集零下载：`tnews15`（原项目 15 类）和 `tnews4`（同一批句子按原 labels.py 的
  GROUP_MAP 重映射成金融/产业/宏观/噪音 4 组）。

## 五分钟跑通（先跑秒级方法）

```bash
python common_exp.py                     # 自检：数据能读、tnews4 重映射正确、抽样一致

# ① 方法轴 + 数据集轴：秒级方法在两个数据集上各跑一遍
python eval_all.py --run-light           # 自动跑 rule + tfidf（tnews15 与 tnews4 各一次）

# ② 模型轴：BERT 换预训练模型（每个约 14 分钟 GPU，CPU 慢但能跑）
python run_bert.py --model_name bert-base-chinese
python run_bert.py --model_name hfl/chinese-roberta-wwm-ext
python run_bert.py --model_name hfl/chinese-macbert-base
python run_bert.py --eval_only           # 重启后只评测，不重训

# ③ 换数据集跑 BERT：任务从 15 类变 4 组，看各方法差距怎么变
python run_bert.py --dataset tnews4 --model_name bert-base-chinese

# ④ LLM 轴：换基座模型（需要 GPU，0.5B 级模型约 1.2 GB）
python run_llm_zero_shot.py --n 200
python run_llm_zero_shot.py --base_model Qwen/Qwen2.5-0.5B --n 200
python run_llm_lora.py --num_train 5000 --epochs 2            # LoRA
python run_llm_lora.py --mode full_ft --num_train 3000 --epochs 2 --gradient_checkpointing
python run_llm_lora.py --eval_only                            # 只评测已有 adapter

# ⑤ 真·换数据集（可选，需联网）：CLUE iflytek（119 类 APP 描述）→ 之后所有脚本 --dataset iflytek
python make_custom_dataset.py
python run_tfidf.py --dataset iflytek

# ⑥ 汇总两张表（总表 + 换模型对照表）
python eval_all.py
```

## 结果怎么看

- `outputs/summary.md`：**总表**（数据集 × 方法 × 模型），以及固定 tnews15 + BERT 的
  **换模型对照表**——「换个预训练模型值几个点」一眼看出。
- 单次结果：`outputs/<数据集>_<模型>_<方法>_pred.json`，带逐条预测（错误分析用）。
- 横向可比的前提（沿用原项目口径）：所有方法都在**同一批去重后的干净验证样本**上评测
  （seed=42）；`eval_all.py` 会自动检查样本数/seed 是否一致，不一致会报警。

## 可以预期的大致结论（以实际跑出为准）

1. **换预训练模型**（BERT 系）：RoBERTa-wwm / MacBERT 通常比 bert-base-chinese 高 1~3 个点，
   但差距远小于「规则 vs 微调」的方法差距——选型时方法优先于模型。
2. **换标签口径**（tnews15 → tnews4）：15 类变 4 组，所有方法都大涨；原本塌缩的 LLM zero-shot
   在粗粒度标签下可用的概率也变高——「任务变简单」对弱方法的救场作用最明显。
3. **换数据集**（iflytek 119 类长文本）：类别数一多，规则法彻底失效、TF-IDF 相对排名上升，
   BERT 优势收窄——短文本封闭标签是判别式模型的主场，换了场景结论就可能翻。

## 与原项目的边界

- 本文件夹**不写入** `week6-数据接入与分流/` 的任何位置；原项目的训练产物（outputs/ckpt 等）也不读。
- 训练技巧消融（Label Smoothing / Focal / FGM）、泄漏对照（--leak）、API zero-shot 等实验仍在原项目
  脚本里，需要时回原目录跑。

"""
方法三：BERT 系判别式微调（L2 主力）—— 改编自原 bert/train_bert.py + predict_bert.py

改动点（相对原版）：
  1. 加 --model_name：这就是「换预训练模型」的入口。既接受本地目录名
     （先在 pretrain_models 下找，离线优先），也接受 HF repo id（本地没有就联网下载）：
       python run_bert.py --model_name bert-base-chinese                    # 原版对照
       python run_bert.py --model_name hfl/chinese-roberta-wwm-ext          # RoBERTa-wwm
       python run_bert.py --model_name hfl/chinese-macbert-base             # MacBERT
       python run_bert.py --model_name hfl/chinese-electra-180g-base-discriminator
  2. 加 --dataset：tnews15 / tnews4 / 自备数据集，类别数自动跟随数据集；
  3. 训练完直接在「同一批干净验证子集」上评测并落盘，ckpt / 日志 / 预测
     全部按 <数据集>_<模型> 命名写进本文件夹 outputs/，不碰原目录；
  4. 砍掉原版的 Label Smoothing / Focal / FGM 三个技巧开关（原任务上实测都不涨，
     换模型对比实验用不到，需要时回原脚本跑）。

用法：
  python run_bert.py                                              # 默认 bert-base-chinese + tnews15
  python run_bert.py --model_name hfl/chinese-roberta-wwm-ext     # 换模型对照
  python run_bert.py --dataset tnews4 --epochs 3                  # 换数据集对照
  python run_bert.py --eval_only                                  # 只用已有 ckpt 重新评测
  python run_bert.py --max_steps 20 --max_eval 300 --no_save      # 冒烟：验流程不落盘
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_exp import EXP_OUT, load_dataset, model_tag, resolve_model, result_path, save_result
from common.metrics import classification_report, expected_calibration_error, to_markdown_table


class TextDataset(Dataset):
    """与原版 TnewsDataset 相同，只是不再叫 Tnews（数据集可换）"""

    def __init__(self, data, tok, max_length=64):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d["sentence"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(d["label"])}


class BertClassifier(nn.Module):
    """BERT + 池化 + 线性头（cls / mean / max 三选一，mask 掉 [PAD] 再池化）"""

    def __init__(self, model_dir, num_labels, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


@torch.no_grad()
def predict(model, tok, texts, device, max_length=64, batch_size=64):
    """返回 (preds, probs, 每样本毫秒数)——评测口径与原版 predict_bert.py 一致"""
    preds, probs = [], []
    t0 = time.time()
    model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1)
        preds.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
    ms = (time.time() - t0) * 1000 / max(1, len(texts))
    return preds, probs, round(ms, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-chinese",
                    help="本地 pretrain_models 目录名或 HF repo id")
    ap.add_argument("--dataset", default="tnews15")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="最终评测抽样条数，0=全部干净样本")
    ap.add_argument("--max_eval", type=int, default=0, help="训练中每轮验证条数上限，0=全部")
    ap.add_argument("--max_steps", type=int, default=0, help="每 epoch 最多步数，0=全跑（冒烟用）")
    ap.add_argument("--no_save", action="store_true", help="不落盘 ckpt（冒烟用）")
    ap.add_argument("--eval_only", action="store_true", help="跳过训练，直接加载已有 ckpt 评测")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path, src = resolve_model(args.model_name)
    tag = model_tag(args.model_name)
    print(f"[model] {args.model_name} → {model_path}（{src}）| device={device}", flush=True)

    train, samples, labels, meta = load_dataset(args.dataset, args.n or None, args.seed)
    num_labels = len(labels)
    tok = AutoTokenizer.from_pretrained(model_path)
    ckpt_path = result_path(args.dataset, tag, "bert", ext="bert_ckpt.pt")

    model = None
    if not args.eval_only:
        model = BertClassifier(model_path, num_labels, pool=args.pool).to(device)
        head_params = list(model.classifier.parameters())
        bert_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
        opt = torch.optim.AdamW([
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ], weight_decay=0.01)

        train_dl = DataLoader(TextDataset(train, tok, args.max_length),
                              batch_size=args.batch_size, shuffle=True)

        if args.no_class_weight:
            weight = None
        else:
            import numpy as np
            from sklearn.utils.class_weight import compute_class_weight
            cw = compute_class_weight("balanced", classes=np.arange(num_labels),
                                      y=[d["label"] for d in train])
            weight = torch.tensor(cw, dtype=torch.float).to(device)
        crit = nn.CrossEntropyLoss(weight=weight)

        steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
        total_steps = steps_per_epoch * args.epochs
        warmup = int(total_steps * 0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        # 训练监控集：小成本看每轮趋势（最终指标以训练后的干净子集评测为准）
        monitor_n = args.max_eval or 2000
        _, monitor, _, _ = load_dataset(args.dataset, monitor_n, args.seed)
        monitor_dl = DataLoader(TextDataset(monitor, tok, args.max_length), batch_size=args.batch_size)

        print(f"[cfg] dataset={args.dataset} num_labels={num_labels} pool={args.pool} "
              f"class_weight={'off' if weight is None else 'on'} epochs={args.epochs} "
              f"steps/epoch={steps_per_epoch} save={'off' if args.no_save else 'on'}", flush=True)

        log, best = [], -1.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, running, n_step = time.time(), 0.0, 0
            for i, batch in enumerate(train_dl):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss = crit(model(ids, mask), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                n_step += 1
                if i % 200 == 0:
                    print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
                if args.max_steps and n_step >= args.max_steps:
                    break

            acc = evaluate_acc(model, monitor_dl, device)
            log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                        "monitor_acc": acc, "elapsed_s": round(time.time() - t0, 1)})
            print(f"epoch {epoch}: monitor_acc={acc}", flush=True)
            if acc > best and not args.no_save:
                best = acc
                torch.save({"state_dict": model.state_dict(), "pool": args.pool,
                            "model_name": args.model_name, "dataset": args.dataset,
                            "num_labels": num_labels, "args": vars(args)}, ckpt_path)

        save_result(result_path(args.dataset, tag, "bert", ext="bert_train_log.json"), log)
        if args.no_save:
            print("--no_save：冒烟模式，ckpt 与日志都没写")
        else:
            print(f"最佳监控 acc={best} → {ckpt_path}")

    if args.no_save and not args.eval_only:
        return   # 冒烟模式不做最终评测

    # ── 最终评测：加载最优 ckpt（冒烟时用当前模型），在干净子集上出正式指标 ──
    if model is None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = BertClassifier(model_path, ckpt.get("num_labels", num_labels),
                               pool=ckpt.get("pool", args.pool))
        model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    texts = [d["sentence"] for d in samples]
    y_true = [d["label"] for d in samples]
    preds, probs, ms = predict(model, tok, texts, device, args.max_length)

    report = classification_report(y_true, preds, num_classes=num_labels)
    report.update({"method": "bert_finetune", "model": args.model_name, "pool": args.pool,
                   "ms_per_sample": ms,
                   "ece": expected_calibration_error(y_true, probs)})

    out = result_path(args.dataset, tag, "bert")
    save_result(out, {"meta": meta, "report": report,
                      "preds": [{"text": t, "true": y, "pred": p}
                                for t, y, p in zip(texts, y_true, preds)]})
    print(f"[{args.dataset}] 评测 {meta['n_eval']} 条（干净集）| {ms} ms/条 | ECE={report['ece']}")
    print(to_markdown_table({f"BERT 微调 · {args.model_name}": report}))
    print(f"→ 结果写入 {out}")


@torch.no_grad()
def evaluate_acc(model, loader, device):
    from sklearn.metrics import accuracy_score
    model.eval()
    ys, ps = [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
    return round(float(accuracy_score(ys, ps)), 4)


if __name__ == "__main__":
    main()

"""
方法五：LLM 指令微调 LoRA / 全量微调（L4）—— 改编自原 llm_lora/train_lora.py + eval_lora.py

改动点（相对原版）：
  1. 加 --base_model：「换 LLM」的入口（本地目录名或 HF repo id）：
       python run_llm_lora.py                                        # Qwen/Qwen3-0.6B + LoRA
       python run_llm_lora.py --base_model Qwen/Qwen2.5-0.5B         # 换模型对照
       python run_llm_lora.py --full_ft                              # 全量微调对照
  2. 加 --dataset：SFT 答案与 prompt 类别清单自动跟随数据集；
  3. 训练完直接在同一批干净验证子集上生成评测（原版拆成 train_lora / eval_lora 两个脚本），
     adapter 与日志按 <数据集>_<模型> 命名写进本文件夹 outputs/adapters/，不碰原目录。

用法：
  python run_llm_lora.py --num_train 5000 --epochs 2
  python run_llm_lora.py --dataset tnews4 --num_train 5000 --epochs 2
  python run_llm_lora.py --max_steps 20                     # 冒烟测流程
  python run_llm_lora.py --eval_only                        # 只用已有 adapter 评测
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_exp import EXP_OUT, load_dataset, model_tag, resolve_model, result_path, save_result
from common.metrics import classification_report, pred_distribution, to_markdown_table


class SFTDataset(Dataset):
    """
    分类样本 → chat 格式 SFT 样本。
    核心仍是 Loss Masking：prompt 部分 labels=-100（不计 loss），只有答案 token 产生梯度。
    与原版唯一区别：答案来自数据集自己的 labels 表（不再是写死的 TNEWS LABELS）。
    """

    def __init__(self, data, tok, system, labels, max_length=128):
        self.data, self.tok, self.max_length = data, tok, max_length
        self.system, self.labels = system, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        answer = self.labels[d["label"]]
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": d["sentence"]}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        input_ids = (p_ids + a_ids)[: self.max_length]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_length]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    """input_ids 用 pad 补齐、labels 用 -100 补齐（PAD 不产生 loss）"""
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def parse_label(text, labels):
    hits = [n for n in labels if n in text]
    return labels.index(max(hits, key=len)) if hits else -1


@torch.no_grad()
def generate_eval(model, tok, samples, labels, max_new_tokens=8):
    """与训练同一个 chat 模板 + enable_thinking=False，逐条贪心生成后解析标签"""
    system = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
              "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(labels))
    preds, raws = [], []
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": d["sentence"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws.append(raw)
        preds.append(parse_label(raw, labels))
        if i % 50 == 0:
            print(f"  eval {i}/{len(samples)}", flush=True)
    return preds, raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dataset", default="tnews15")
    ap.add_argument("--mode", default="lora", choices=["lora", "full_ft"])
    ap.add_argument("--num_train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4, help="full_ft 建议 1~2")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="不填则 lora=2e-4、full_ft=2e-5")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--gradient_checkpointing", action="store_true", help="full_ft 在 8G 卡上必开")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 只跑这么多步（冒烟）")
    ap.add_argument("--n_eval", type=int, default=500, help="训练后评测的干净样本条数")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path, src = resolve_model(args.base_model)
    tag = model_tag(args.base_model)
    print(f"[model] {args.base_model} → {model_path}（{src}）| device={device} | mode={args.mode}", flush=True)

    train, samples, labels, meta = load_dataset(args.dataset, args.n_eval, args.seed)
    if not args.eval_only:
        train = train[: args.num_train]
    SYSTEM = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
              "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(labels))

    dtype = torch.bfloat16 if (args.mode == "full_ft" and device == "cuda") else (
        torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    adapter_dir = EXP_OUT / "adapters" / f"{args.dataset}_{tag}_{args.mode}"
    model = None
    if not args.eval_only:
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        if args.mode == "full_ft":
            model = base.to(device)
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            lr = args.lr or 2e-5
            print(f"[full_ft] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M（全部可训练）lr={lr}")
        else:
            from peft import LoraConfig, TaskType, get_peft_model
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                             lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            model = get_peft_model(base, cfg).to(device)
            lr = args.lr or 2e-4
            model.print_trainable_parameters()

        train_dl = DataLoader(SFTDataset(train, tok, SYSTEM, labels, args.max_length),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, tok.pad_token_id))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))

        log, step, t_start = [], 0, time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            running, n_batch = 0.0, 0
            for i, batch in enumerate(train_dl):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=(device == "cuda" and dtype == torch.float16)):
                    loss = model(**batch).loss / args.grad_accum
                scaler.scale(loss).backward()
                if (i + 1) % args.grad_accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                running += loss.item() * args.grad_accum
                n_batch += 1
                if i % 100 == 0:
                    print(f"  epoch {epoch} {i}/{len(train_dl)} loss={loss.item()*args.grad_accum:.4f}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
            log.append({"epoch": epoch, "train_loss": round(running / max(1, n_batch), 4),
                        "steps": step, "elapsed_s": round(time.time() - t_start, 1),
                        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None})
            print(f"epoch {epoch}: train_loss={log[-1]['train_loss']}", flush=True)
            if args.max_steps:
                break

        save_result(result_path(args.dataset, tag, args.mode, ext="llm_train_log.json"), log)
        if not args.max_steps:
            adapter_dir.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            print(f"权重 → {adapter_dir}")

    if args.max_steps and not args.eval_only:
        print("--max_steps：冒烟模式，不做最终评测")
        return

    # ── 最终评测：挂 adapter（或直接加载完整权重）后在干净子集上生成 ──
    if model is None:
        if (adapter_dir / "adapter_config.json").exists():
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
            model = PeftModel.from_pretrained(base, str(adapter_dir))
            print(f"[eval] 挂载 LoRA adapter：{adapter_dir}")
        else:
            model = AutoModelForCausalLM.from_pretrained(str(adapter_dir), dtype=dtype)
            print(f"[eval] 加载完整权重：{adapter_dir}")
    model.to(device).eval()

    preds, raws = generate_eval(model, tok, samples, labels)
    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds, num_classes=len(labels))
    report["method"] = f"llm_{args.mode}"
    report["model"] = args.base_model
    report["pred_distribution"] = pred_distribution(preds)

    out = result_path(args.dataset, tag, args.mode)
    save_result(out, {"meta": meta, "report": report,
                      "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                for d, p, r in zip(samples, preds, raws)]})
    print(f"[{args.dataset}] 评测 {meta['n_eval']} 条（干净集）")
    print(to_markdown_table({f"{args.base_model} + {args.mode}": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(labels)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

"""
方法四：大模型零样本（L3，不训练）—— 改编自原 llm_zero_shot/classify_zero_shot.py

改动点（相对原版）：
  1. 加 --base_model：「换 LLM」的入口。本地 pretrain_models 里找得到就用本地权重，
     否则按 HF repo id 联网下载：
       python run_llm_zero_shot.py                                   # 默认 Qwen/Qwen3-0.6B
       python run_llm_zero_shot.py --base_model Qwen/Qwen2.5-0.5B    # 换模型对照
       python run_llm_zero_shot.py --base_model Qwen/Qwen3-1.7B
  2. 加 --dataset：prompt 里的类别清单自动跟随数据集（tnews15 15 类 / tnews4 4 组 / 自备）；
  3. 结果写进本文件夹 outputs/，不碰原目录。

用法：
  python run_llm_zero_shot.py --n 200
  python run_llm_zero_shot.py --dataset tnews4 --n 200
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_exp import load_dataset, model_tag, resolve_model, result_path, save_result
from common.metrics import classification_report, pred_distribution, to_markdown_table


def parse_label(text, labels):
    """从生成文本里抽类别名（包含匹配 + 取最长命中），抽不到返回 -1"""
    hits = [n for n in labels if n in text]
    return labels.index(max(hits, key=len)) if hits else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B",
                    help="本地 pretrain_models 目录名或 HF repo id")
    ap.add_argument("--dataset", default="tnews15")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path, src = resolve_model(args.base_model)
    print(f"[model] {args.base_model} → {model_path}（{src}）| device={device}", flush=True)

    _, samples, labels, meta = load_dataset(args.dataset, args.n, args.seed)
    tag = model_tag(args.base_model)

    # Prompt 三件套：角色 + 类别定义 + 输出约束（类别名必须列全，否则模型自己发明标签）
    SYSTEM = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
              "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(labels))

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    preds, raws, t0 = [], [], time.time()
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": d["sentence"]}]
        # Qwen3 必须 enable_thinking=False，否则 <think> 推理链会打乱「只输出类别名」的格式
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws.append(raw)
        preds.append(parse_label(raw, labels))
        if i % 50 == 0:
            print(f"  {i}/{len(samples)} {time.time()-t0:.0f}s", flush=True)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds, num_classes=len(labels))
    report["method"] = "llm_zero_shot"
    report["model"] = args.base_model
    report["sec_per_item"] = round((time.time() - t0) / max(1, len(samples)), 2)
    report["pred_distribution"] = pred_distribution(preds)

    out = result_path(args.dataset, tag, "zs")
    save_result(out, {"meta": meta, "report": report,
                      "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                for d, p, r in zip(samples, preds, raws)]})

    print(f"[{args.dataset}] 评测 {meta['n_eval']} 条 | {report['sec_per_item']}s/条")
    print(to_markdown_table({f"{args.base_model} zero-shot": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(labels)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

"""
方法一：规则 / 关键词（L0 基线）—— 改编自原 rule/classify_rule.py

改动点（相对原版）：
  1. 加 --dataset：在 tnews15 / tnews4 / 自备数据集上都能跑（规则逻辑不变：
     标题里出现哪个类别名就判哪类，全没有就猜训练集最多类）；
  2. 结果文件按 <数据集>_<模型>_<方法> 命名，写入本文件夹 outputs/，不碰原目录。

用法：
  python run_rule.py                      # tnews15 全量干净验证集
  python run_rule.py --dataset tnews4     # 4 组口径下规则法的天花板
  python run_rule.py --n 500
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_exp import load_dataset, model_tag, result_path, save_result
from common.metrics import classification_report, pred_distribution, to_markdown_table


def classify(text: str, labels, fallback: int) -> int:
    """与原版一致：按 labels 顺序取第一个命中的类别名，不做优先级——暴露规则法的真实上限"""
    for i, name in enumerate(labels):
        if name in text:
            return i
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tnews15")
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部干净样本")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, samples, labels, meta = load_dataset(args.dataset, args.n or None, args.seed)
    fallback = Counter(d["label"] for d in train).most_common(1)[0][0]   # majority baseline

    y_true = [d["label"] for d in samples]
    y_pred = [classify(d["sentence"], labels, fallback) for d in samples]

    report = classification_report(y_true, y_pred, num_classes=len(labels))
    report["method"] = "rule_keyword"
    report["model"] = "-"            # 规则法没有模型
    report["pred_distribution"] = pred_distribution(y_pred)

    tag = model_tag("rule")
    out = result_path(args.dataset, tag, "rule")
    save_result(out, {"meta": meta, "report": report,
                      "preds": [{"text": d["sentence"], "true": t, "pred": p}
                                for d, t, p in zip(samples, y_true, y_pred)]})

    hit = sum(1 for d in samples if any(n in d["sentence"] for n in labels)) / max(1, len(samples))
    print(f"[{args.dataset}] 样本 {meta['n_eval']} 条（干净集，剔除重复 {meta['n_dup']} 条）| 类别名命中率 {hit:.2%}")
    print(to_markdown_table({"规则/关键词": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

"""
方法二：TF-IDF + LinearSVC（L1 基线）—— 改编自原 tfidf/classify_tfidf.py

改动点（相对原版）：
  1. 加 --dataset（tnews15 / tnews4 / 自备数据集）；
  2. 结果文件按 <数据集>_<模型>_<方法> 命名，写入本文件夹 outputs/；
  3. 保留 --ngram / --C 两个调参旋钮（原项目的调参对比实验还能复现）。

用法：
  python run_tfidf.py                          # tnews15，char(1,2) + C=1.0
  python run_tfidf.py --dataset tnews4         # 换成 4 组口径：任务变简单，看各方法差距怎么变
  python run_tfidf.py --n 500 --ngram 1,3 --C 0.5
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_exp import load_dataset, result_path, save_result
from common.metrics import classification_report, to_markdown_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tnews15")
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部干净样本")
    ap.add_argument("--ngram", default="1,2", help="字符 n-gram 范围，如 1,2 或 1,3")
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC 正则强度")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    train, samples, labels, meta = load_dataset(args.dataset, args.n or None, args.seed)

    lo, hi = (int(x) for x in args.ngram.split(","))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), max_features=300000)

    t0 = time.time()
    Xtr = vec.fit_transform([d["sentence"] for d in train])
    Xte = vec.transform([d["sentence"] for d in samples])
    clf = LinearSVC(C=args.C).fit(Xtr, [d["label"] for d in train])
    y_pred = clf.predict(Xte).tolist()
    fit_seconds = round(time.time() - t0, 1)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, y_pred, num_classes=len(labels))
    report.update({"method": "tfidf_svm", "model": f"char({lo},{hi})+LinearSVC(C={args.C})",
                   "fit_seconds": fit_seconds, "ngram": args.ngram, "C": args.C})

    out = result_path(args.dataset, "tfidf", "tfidf")
    save_result(out, {"meta": meta, "report": report,
                      "preds": [{"text": d["sentence"], "true": t, "pred": p}
                                for d, t, p in zip(samples, y_true, y_pred)]})

    print(f"[{args.dataset}] 评测 {meta['n_eval']} 条 | 训练+推理 {fit_seconds}s | ngram={args.ngram} C={args.C}")
    print(to_markdown_table({"TF-IDF+SVM": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

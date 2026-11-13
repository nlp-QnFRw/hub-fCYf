"""准备数据：解压语料 → 构建字符级词表 → 打印统计。

运行：
    python scripts/prepare_data.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sftlab.data import read_jsonl, summarize  # noqa: E402
from sftlab.template import encode_example, render_template  # noqa: E402
from sftlab.tokenizer import CharTokenizer  # noqa: E402
from sftlab.utils import PROJECT_ROOT, save_json  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


def unzip_corpus() -> None:
    """把仓库根目录的 sft_train.zip 解到 data/raw/，已存在就跳过。"""
    target = RAW_DIR / "sft_train.jsonl"
    if target.exists():
        print(f"[1/3] 语料已存在，跳过解压: {target.relative_to(PROJECT_ROOT)}")
        return

    # 压缩包在课程目录（本项目的上级），不在 work/ 里，所以两个位置都找一下
    candidates = [PROJECT_ROOT / "sft_train.zip", PROJECT_ROOT.parent / "sft_train.zip"]
    zpath = next((p for p in candidates if p.exists()), None)
    if zpath is None:
        raise FileNotFoundError(
            "找不到 sft_train.zip，请把它放到以下任一位置：\n  "
            + "\n  ".join(str(p) for p in candidates)
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW_DIR)
        names = z.namelist()
    print(f"[1/3] 已解压 {len(names)} 个文件 → {RAW_DIR.relative_to(PROJECT_ROOT)}: {', '.join(names)}")


def main() -> None:
    unzip_corpus()

    train_rows = read_jsonl(RAW_DIR / "sft_train.jsonl")
    val_rows = read_jsonl(RAW_DIR / "sft_val.jsonl")
    print(f"[2/3] 训练集 {summarize(train_rows)}")
    print(f"      验证集 {summarize(val_rows)}")

    # 词表必须只用训练集统计：用上验证集等于提前「看过考卷」，虽然影响很小，但习惯要正。
    tok = CharTokenizer.build(r["instruction"] + r["output"] for r in train_rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(PROC_DIR / "vocab.json")
    print(f"[3/3] 词表 {len(tok)} 个 token → data/processed/vocab.json")

    # 顺手存一条模板样例，方便对照 docs/ 里的说明
    demo = encode_example(tok, train_rows[0]["instruction"], train_rows[0]["output"])
    save_json(
        {
            "template": render_template(tok, train_rows[0]["instruction"], train_rows[0]["output"]),
            "prompt_len": demo.prompt_len,
            "answer_len": demo.answer_len,
            "supervised_ratio": round(demo.supervised_ratio, 3),
        },
        PROC_DIR / "sample.json",
    )

    print("\n样例（模型真正看到的字符串）：")
    print("  " + render_template(tok, train_rows[0]["instruction"], train_rows[0]["output"]))
    print(f"  prompt {demo.prompt_len} 个 token | 回答 {demo.answer_len} 个 token "
          f"| 参与 loss 的比例 {demo.supervised_ratio:.1%}")
    print("\n下一步: python scripts/train.py --stage pretrain")


if __name__ == "__main__":
    main()

"""保留原始 JSONL，另存只含 text 字段的 tagged 数据。"""

import argparse
import json
from pathlib import Path

from sft_common import SCRIPT_DIR, Sample, load_tagged, make_tagged_text, read_jsonl


def convert_file(source, destination):
    """转换并逐条还原校验；原文件不变，成功返回记录数。"""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("输出路径不能覆盖原始数据")
    rows = read_jsonl(source)
    originals, tagged = [], []
    for row_no, row in enumerate(rows, 1):
        try:
            text = make_tagged_text(row.get("instruction"), row.get("output"))
        except ValueError as exc:
            raise ValueError(f"{source}: 第 {row_no} 条记录: {exc}") from exc
        originals.append(Sample(row["instruction"], row["output"]))
        tagged.append({"text": text})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in tagged:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if load_tagged(temporary) != originals:
        raise ValueError("转换后的数据未通过逐条还原校验")
    temporary.replace(destination)
    return len(tagged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()
    for split in ("train", "val"):
        source = args.data_dir / f"sft_{split}.jsonl"
        destination = args.data_dir / f"sft_{split}_tagged.jsonl"
        count = convert_file(source, destination)
        print(f"{source.name} -> {destination.name}: {count} 条，逐条还原校验通过")


if __name__ == "__main__":
    main()

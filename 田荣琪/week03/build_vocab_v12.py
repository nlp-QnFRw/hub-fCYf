# -*- coding: utf-8 -*-
"""
构建12类语种专用字符表
"""

from pathlib import Path
from collections import Counter

from train_langid_v12 import (
    load_split,
    build_char_vocab,
    normalize,
    load_vocab_file
)


HERE = Path(__file__).resolve().parent

DATA_DIR = HERE / "data" / "langid" / "v12"

VOCAB_PATH = HERE / "data" / "langid" / "vocab" / "vocab_v12.txt"


def main():

    train_rows = load_split(str(DATA_DIR), "train")

    sentences = [text for text, label in train_rows]

    vocab = build_char_vocab(sentences)

    freq = Counter()

    for text in sentences:
        freq.update(normalize(text))


    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:

        # 写入表头
        f.write("# 字符\tid\t出现次数\n")

        # 写入字符、ID和频率
        for char, idx in vocab.items():
            count = freq.get(char, 0)

            f.write(f"{char}\t{idx}\t{count}\n")

    print("字符表已保存：", VOCAB_PATH)

    loaded_vocab = load_vocab_file(str(VOCAB_PATH))

    assert vocab == loaded_vocab, "保存和读取后的字符表不一致！"



if __name__ == "__main__":
    main()

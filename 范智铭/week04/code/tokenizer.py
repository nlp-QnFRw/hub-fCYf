"""字符级分词器（char-level tokenizer）。

词表文件 ``vocab.txt`` 的格式：每行一个 token，行号即 token id。
前 4 行固定为特殊 token：``<pad> <unk> <bos> <eos>``，
其后的行按字符在训练集中的频次从高到低排列。

用法::
    from tokenizer import CharTokenizer
    tok = CharTokenizer.from_file(cfg)
    ids = tok.encode("今天天气怎么样？", add_bos=True, add_eos=True)
    text = tok.decode(ids)          # 默认跳过特殊 token
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

#: 特殊 token 的顺序即它们在词表中的 id：pad=0, unk=1, bos=2, eos=3
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)

_SPECIAL_PATTERN = re.compile(
    "(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")"
)

class CharTokenizer:
    """把一个字符串映射成字符 id 序列，并支持反向解码。"""

    def __init__(self,tokens: Sequence[str]) -> None:
        tokens = list(tokens)
        if any(token == "" for token in tokens):
            raise ValueError("词表中存在空行，请检查或重新生成 vocab.txt")
        counts = Counter(tokens)
        duplicates = {token: count for token, count in counts.items() if count > 1}
        if duplicates:
            shown = "、".join(
                f"{token!r}（出现 {count} 次）"
                for token, count in list(duplicates.items())[:5]
            )
            if len(duplicates) > 5:
                shown += f" 等 {len(duplicates)} 个 token"
            raise ValueError(
                f"词表中存在重复 token：{shown}，请检查或重新生成 vocab.txt"
            )
        if not all(token in tokens for token in SPECIAL_TOKENS):
            raise ValueError(f"词表必须包含特殊 token：{SPECIAL_TOKENS}")
        self.itos: list[str] = tokens
        self.stoi: dict[str, int] = {token: i for i, token in enumerate(tokens)}
        self._special_set = frozenset(SPECIAL_TOKENS)

    # ------------------------------------------------------------------ 构建
    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "CharTokenizer":
        """统计 ``texts`` 中的字符频次并构建词表。

        Args:
            texts: 训练语料文本，逐条统计。
            min_freq: 出现次数低于该值的字符会被丢弃，编码为 ``<unk>``。
            special_tokens: 特殊 token 列表，按给定顺序放在词表最前面。
        """
        if min_freq < 1:
            raise ValueError("min_freq 必须 >= 1")

        counter: Counter[str] = Counter()
        for text in texts:
            for ch in text:
                if ch in "\r\n":
                    continue
                counter[ch] += 1

        chars = [ch for ch, freq in counter.items() if freq >= min_freq]
        # 频次降序；频次相同时按字符编码升序，保证结果可复现
        chars.sort(key=lambda ch: (-counter[ch], ch))
        return cls(list(special_tokens) + chars)

    # ------------------------------------------------------------------ 读写
    @classmethod
    def from_file(cls, cfg, path: str | Path | None = None) -> "CharTokenizer":
        if path is None:
            path = Path(__file__).resolve().parent.parent / cfg.hparams.vocab_path
        path = Path(path)
        with path.open(encoding="utf-8-sig") as f:
            tokens = [line.rstrip("\r\n") for line in f]
        while tokens and tokens[-1] == "":
            tokens.pop()
        return cls(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.itos) + "\n")

    # ------------------------------------------------------------------ 属性
    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    # ------------------------------------------------------------------ 编解码
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """把文本编码成 id 列表。

        文本中可直接书写 ``<eos>`` 等特殊 token（例如 train.txt 每行末尾），
        它们会被当作一个整体映射到对应的特殊 id，而不是逐字符切分。
        """
        ids: list[int] = [self.bos_id] if add_bos else []
        for piece in _SPECIAL_PATTERN.split(text):
            if not piece:
                continue
            if piece in self._special_set:
                ids.append(self.stoi[piece])
            else:
                ids.extend(self.stoi.get(ch, self.unk_id) for ch in piece)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """把 id 列表还原成文本；``skip_special`` 为真时丢弃特殊 token。"""
        if skip_special:
            tokens = (
                self.itos[i]
                for i in ids
                if 0 <= i < len(self.itos) and self.itos[i] not in self._special_set
            )
        else:
            tokens = (self.itos[i] for i in ids if 0 <= i < len(self.itos))
        return "".join(tokens)

    def encode_lines(self, lines: Iterable[str]) -> list[list[int]]:
        """按行编码：每行内部的 ``<eos>`` 保持为单个特殊 id。"""
        return [self.encode(line) for line in lines]

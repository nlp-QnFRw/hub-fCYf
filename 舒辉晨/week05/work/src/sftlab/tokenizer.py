"""字符级分词器。

为什么不直接用 HuggingFace 的 tokenizer？
    本项目要能在**完全离线**、不下载任何模型权重的机器上跑通，
    所以词表直接从语料里统计：出现过的每个汉字/标点/字母都是一个 token。
    词表只有几千，embedding 也小，CPU 都能训。

代价是序列比 BPE 长（一个汉字一个 token），但对「理解 SFT 到底改了什么」这件事
完全没有影响 —— chat template 和 loss mask 的逻辑与真实大模型一模一样。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

# 四个特殊 token。顺序不能变：pad 必须是 0，因为 padding 的 label 要填成 -100，
# 而 pad_id=0 时后面很多地方（比如 mask 判断）都省事。
PAD_TOKEN = "<pad>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

SPECIAL_TOKENS = [PAD_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN]


class CharTokenizer:
    """把一个字符串切成字符 id 序列，特殊 token 作为整体识别。"""

    def __init__(self, itos: Sequence[str]) -> None:
        self.itos = list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        # 特殊 token 按长度倒序，保证扫描时优先匹配更长的那个
        self._specials = sorted(
            (t for t in self.itos[: len(SPECIAL_TOKENS)]), key=len, reverse=True
        )

    # ── 构建 ──────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "CharTokenizer":
        """统计语料里出现过的字符，按频次从高到低排成词表（前面留给特殊 token）。"""
        counter: Counter[str] = Counter()
        for t in texts:
            # 特殊 token 由模板插入，不参与字频统计
            cleaned = t
            for sp in SPECIAL_TOKENS:
                cleaned = cleaned.replace(sp, "")
            counter.update(cleaned)

        chars = [c for c, n in counter.most_common() if n >= min_freq and not c.isspace()]
        # 换行和空格也保留，否则句子之间会粘在一起
        extra = [c for c in ("\n", " ") if c not in chars]
        return cls(SPECIAL_TOKENS + chars + extra)

    # ── 属性 ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def user_id(self) -> int:
        return self.stoi[USER_TOKEN]

    @property
    def assistant_id(self) -> int:
        return self.stoi[ASSISTANT_TOKEN]

    @property
    def end_id(self) -> int:
        return self.stoi[END_TOKEN]

    # ── 编解码 ────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """逐字符扫描；遇到特殊 token 就整体吃掉，否则按单字符处理。"""
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            for sp in self._specials:
                if text.startswith(sp, i):
                    ids.append(self.stoi[sp])
                    i += len(sp)
                    break
            else:
                ch = text[i]
                # 词表里没有的字符（极少数生僻字）直接丢弃，而不是塞一个 unk：
                # 少了 unk 就少一个「所有生僻字共享梯度」的坑，数据量小的时候更稳。
                idx = self.stoi.get(ch)
                if idx is not None:
                    ids.append(idx)
                i += 1
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """id → 文本。默认丢掉特殊 token，调试时可关掉看完整结构。"""
        out: list[str] = []
        specials = set(SPECIAL_TOKENS)
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if skip_special and tok in specials:
                continue
            out.append(tok)
        return "".join(out)

    # ── 存取 ──────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"])

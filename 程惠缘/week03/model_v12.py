# -*- coding: utf-8 -*-
"""model_v12.py —— 12 语种识别要用的模型与预处理

自成一体：不 import 课程里的任何脚本（train_langid_cls.py / eval_langid.py 都不需要）。
只需要 torch。

里面所有东西都和课堂那套等价，只是把「语种列表 / 字符表」从写死改成从数据算：
    · 标签 = 训练集里出现过的 label，排序后编号
    · 字符表 = 从训练集统计（build_char_vocab），不读 vocab_v8.txt
"""
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn

MAX_LEN = 40          # 数据集的长度窗口就是 8~40 字符，超出的截断
PAD_ID, UNK_ID = 0, 1

# 支持的结构：课堂的 nn.RNN 只是其中一种
ARCHS = ("rnn", "lstm", "gru", "bilstm")


# ─── 预处理（必须和训练时一致，否则准确率会悄悄掉）─────────────────────
def normalize(text):
    """NFKC + 小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么要保留 M：印地语、泰语的元音是组合标记，只留字母会把它们砍成裸辅音。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def build_char_vocab(sentences, min_freq=1):
    """从训练集统计字符表：<PAD>=0、<UNK>=1，其余按出现次数从高到低编号。"""
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", PAD_ID), ("<UNK>", UNK_ID)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    """写成和 vocab_v8.txt 一样的格式：字符 \\t id \\t 出现次数"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def coverage(vocab, freq, ratio=0.95):
    """前多少个字符覆盖 ratio 的字符出现次数。"""
    total = sum(freq.values())
    acc, n = 0, 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq.get(ch, 0)
        if acc >= ratio * total:
            return n
    return n


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, UNK_ID) for ch in normalize(text)][:maxlen]
    ids += [PAD_ID] * (maxlen - len(ids))
    return ids


# ─── 模型 ──────────────────────────────────────────────────────────────
class RecurrentClassifier(nn.Module):
    """Embedding → RNN/LSTM/GRU → 掩码 max-pool → LayerNorm → Dropout → Linear

    和课堂的 LangIDClassifier 结构完全一致，只是 cell 可以换、可以双向。
    """

    def __init__(self, vocab_size, num_classes, cell=nn.RNN, embed_dim=64,
                 hidden_dim=64, dropout=0.3, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.rnn = cell(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        mask = (x != PAD_ID).unsqueeze(-1)          # 补齐位不参与池化
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        return self.fc(self.dropout(self.norm(pooled)))


_CELLS = {"rnn": (nn.RNN, False), "lstm": (nn.LSTM, False),
          "gru": (nn.GRU, False), "bilstm": (nn.LSTM, True)}


def build_model(arch, vocab_size, num_classes, embed_dim=64, hidden_dim=64, dropout=0.3):
    """hidden_dim 是**单个方向**的隐藏维数。

    bilstm 会把两个方向拼起来，所以输出维数是 hidden_dim * 2。
    想让 BiLSTM 的总维度和单向 LSTM 一样（64），传 hidden_dim=32 进来。
    """
    if arch not in _CELLS:
        raise ValueError(f"不认识的 arch：{arch}（可选 {' / '.join(_CELLS)}）")
    cell, bidir = _CELLS[arch]
    return RecurrentClassifier(vocab_size, num_classes, cell=cell, embed_dim=embed_dim,
                               hidden_dim=hidden_dim, dropout=dropout, bidirectional=bidir)


def load_for_inference(path):
    """读 checkpoint，重建模型。返回 (model, vocab, labels, maxlen, ckpt)。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(ckpt["arch"], len(ckpt["vocab"]), len(ckpt["labels"]),
                        embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
                        dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["vocab"], ckpt["labels"], ckpt.get("maxlen", MAX_LEN), ckpt


def predict_proba(model, vocab, text, maxlen=MAX_LEN):
    ids = torch.tensor([encode(text, vocab, maxlen)])
    with torch.no_grad():
        return torch.softmax(model(ids), dim=1)[0]

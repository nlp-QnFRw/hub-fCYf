from collections import Counter, OrderedDict
import unicodedata
import os 
import pandas


# ─── 1. 文本归一化（必须和建数据时一致）───────────────────────────u─────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))

# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    多语种的关键点：这张表不需要按语种分开建。汉字、拉丁字母、西里尔字母、
    阿拉伯字母、天城文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    课堂主流程没有直接调用它（字符表是从 vocab_v8.txt 读的）；
    一旦换了训练集或加了语种，就要用它重新生成一张表，
    否则新语种的字符会全部落到 1 号 <UNK>，模型等于睁眼瞎。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    apperance_counts = [0,0]
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, count in chars:
        apperance_counts.append(count)
        vocab[ch] = len(vocab)
    return vocab, apperance_counts


def build_text():
    train_csv_path = os.path.join("data", 'langid', 'v12','train.csv')
    # with open(train_csv_path, 'r', encoding="utf-8")as f:
    csv_data = pandas.read_csv(train_csv_path)
    with open(os.path.join('data', 'langid','vocab','vocab_v12.txt'), 'w', encoding='utf-8')as f:
        f.write("#\t字符\tid\t出现次数\n")
        sentences = []
        for i in range(len(csv_data)):
            text = csv_data.iloc[i]['text']
            sentences.append(text)
        vocab,apperance_counts = build_char_vocab(sentences)
        for idx, item in enumerate(vocab.items()):
            f.write(f"{item[0]}\t{item[1]}\t{apperance_counts[idx]}\n")

if __name__ == "__main__":
    main()
"""
任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（作业 12 语种：课堂 8 个 + ko / th / vi / tr）
    python train_langid_12cls.py                          # 全流程：训练 + 评测 + 存档
    python train_langid_12cls.py --epochs 12              # 多训几轮
    python train_langid_12cls.py --save-vocab             # 顺手把 12 类字符表写成文件
    python train_langid_12cls.py --no-extra               # 跳过难度切片 / 开放集，只看主指标
    python train_langid_12cls.py --epochs 1 --save ""     # 试跑一次、不覆盖已有存档（--save 空串 = 不存）
"""

#region 0. 导入库
import os                    
import csv                   
import time                  
import random                
import argparse              
import unicodedata           # NFKC 归一化 + 判断字符类别（字母 L / 组合符号 M）
import torch                 
import torch.nn as nn        
from torch.utils.data import Dataset, DataLoader   # 数据集封装 + 批量送数据
from collections import Counter, OrderedDict       # 数频率 / 有序字典（词表要按 id 顺序保存）
#endregion

#region 1. 定义超参数与路径
SEED = 42             # 随机种子
MAX_LEN = 40          # 一句话最多看 40 个字符：数据本身是按 8~40 字符窗口采样的，正好全覆盖，不用再放宽
EMBED_DIM = 64       
HIDDEN_DIM = 160      # 隐藏维度,64维度太小了
DROPOUT = 0.5         # 训练时随机丢掉 50% 的神经元，防过拟合（评测时自动关掉）；维度变大后要相应加大
BATCH_SIZE = 64       # B: 一次喂给模型 64 条句子
LR = 1e-3             
EPOCHS = 20           
NUM_THREADS = 1       # 小模型单线程反而更快（线程调度开销 > 并行收益），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))                    # 本文件所在目录
DATA_ROOT = os.path.join(HERE, "data", "langid")                     # 数据根目录
DATA_DIR = os.path.join(DATA_ROOT, "v12")                            # 作业用的 12 语种数据
DEFAULT_SAVE = os.path.join(HERE, "langid_rnn_v12.pt")               # 默认存档路径
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"] #12语种标签
NUM_CLASSES = len(LABELS)   # 12：模型最后一层的输出维度
#endregion

#region 2. 函数：文本归一化                            【函数:normalize(原始文本),return 归一化后的字符串】
def normalize(text):
    """NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」"""
    t = unicodedata.normalize("NFKC", text).lower()          # NFKC：全角转半角、兼容字符统一；lower：大小写不敏感
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))   # 逐字符过滤：L=字母、M=组合符号（涉及印度语和泰语），其余（数字/标点/emoji）全扔
#endregion

#region 3. 字符表操作：统计字符表 / 存储字符表 / 读取字符表
#region 3.1 函数：从句子列表统计字符表   【函数:build_char_vocab(句子列表,最低频次),return 有序字典 字符→id】
def build_char_vocab(sentences, min_freq=1):
    """字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。
    建立词表，<PAD> 0 补齐；<UNK> 1 未见过的字符,其余按出现次数从高到低编号，次数相同按字符编码排序"""
    freq = Counter()                                    # 计数器：字符 → 出现次数
    for s in sentences:                                 # 遍历每一条训练句子
        freq.update(normalize(s))                       # Counter.update(字符串) 会按字符拆开累加
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))      # 排序键：先按次数降序（负号变降序），次数相同再按字符编码升序
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])   # 初始化词表：PAD 与 UNK 固定 ID
    for ch, _ in chars:                                 # 剩下的字符依次领编号
        vocab[ch] = len(vocab)                          # 用当前长度当新 id，天然从 2 开始连续递增
    return vocab, freq                                  # 返回：(词表, 频率表)，频率表只用来打印统计
#endregion

#region 3.2 函数：把字符表写成文本文件   【函数:save_vocab(词表,频率表,路径)】
def save_vocab(vocab, freq, path):
    """字符表写成文本文件：每行「字符 \\t id \\t 出现次数」，首行是 # 注释。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)   # 目录不存在就建（比如 data/langid/vocab）
    with open(path, "w", encoding="utf-8") as f:        
        f.write("# 字符\tid\t出现次数\n")                # 表头用 # 开头，读的时候跳过
        for ch, idx in vocab.items():                   # 按 id 顺序遍历（OrderedDict 保证顺序）
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")  # PAD/UNK 在频率表里没有，默认 0
#endregion

#region 3.3 函数：读现成的字符表   【函数:load_vocab(路径),return 有序字典 字符→id】
def load_vocab(path):
    """读取字符表，创建字典，key:字符,value:id"""
    vocab = OrderedDict()                               # 有序字典：保持文件里的 id 顺序
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():  # 跳过注释行与空行(line.strip去掉首尾空白，\n,\t等视为空字符串)
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")  # 三列：字符 / id / 次数，第三列用不上
            vocab[ch] = int(idx)                        # id 在文件里是字符串，要转成 int
    return vocab
#endregion
#endregion

#region 4. 函数：句子 → 定长 ID 列表                   【函数:encode(句子,词表,截断长度),return 长度恒为 40 的 id 列表】
def encode(text, vocab, maxlen=MAX_LEN):
    """文本 → 固定长度整数序列：归一化 → 逐字符查表 → 截断 → 补 0"""
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]   # 先归一化再查表；查不到就是 1 号 <UNK>；[:40] 截断
    ids += [0] * (maxlen - len(ids))                              # 不够 40 个就用 0 补齐
    return ids                                                    # 长度恒为 40 的列表
#endregion

#region 5. 读数据/标签自检              
#region 5.1 函数：读取划分                             【函数:load_split(数据目录,划分名),return [(text,label),...]】
def load_split(data_dir, name):
    """读 train.csv / dev.csv / test.csv，返回 [(文本, 标签), ...]。"""
    path = os.path.join(data_dir, f"{name}.csv")        # 比如 data/langid/v12/train.csv
    rows = []                                           # 空列表，装所有样本
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):                     # DictReader：按表头把每行变成字典
            rows.append((r["text"], r["label"]))        # 只取两列，拼成元组
    return rows
#endregion

#region 5.2 函数：标签自检（数据里的标签必须和 LABELS 对得上）   【函数:check_labels(样本,划分名)】
def check_labels(rows, split_name):
    """
    检查三件事，任何一件不满足就直接退出，避免训到一半才发现输出格子不够：
      1. 数据里有 LABELS 之外的标签 → 模型没有对应输出格
      2. LABELS 里有数据中没出现的标签 → 那个格子永远学不到，评测时除以 0 会很难看
      3. 12 类是否真的每类都有
    """
    seen = sorted({l for _, l in rows})                 # 集合去重再排序：这一划分里到底出现了哪些标签
    extra = [l for l in seen if l not in LABELS]        # 数据有、12类LABELS里面没有 → 多出来的
    missing = [l for l in LABELS if l not in seen]      # 12类LABELS里面有、数据没有 → 缺的
    if extra:
        raise SystemExit(
            f"[{split_name}] 出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{NUM_CLASSES} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    if missing:
        raise SystemExit(
            f"[{split_name}] 缺少这些标签的样本：{' '.join(missing)}\n"
            f"  模型会白留一个学不到东西的输出格，请检查数据或 LABELS。"
        )
#endregion
#endregion

#region 6. 类：创建数据集                                      【集合:self.X 全部句子ID, self.y 全部标签ID】
class TextDataset(Dataset):
    """
    一次性把全部句子编码成 id 列表（数据才 9600 条，内存放得下，不用每次 __getitem__ 再算）。
    self.X: List[List[int]]，长度 = 样本数，每条 40 个 id
    self.y: List[int]，长度 = 样本数，取值 0~11
    """

    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]        # 文本 → 定长 id 列表
        self.y = [label2id[l] for _, l in rows]             # 标签字符串 → 整数 id（0~11）

    def __len__(self):
        return len(self.y)                                  # DataLoader 靠它知道一共多少条

    def __getitem__(self, i):
        # 返回 (张量, 整数)：X 转 long（Embedding 只认整型索引），y 保持 int 交给 CrossEntropyLoss
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]
#endregion

#region 7. 模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
#region 7.1 类：12 类语种分类器   【类别数:NUM_CLASSES=12】
class LangIDClassifier(nn.Module):
    """
    结构：ids(B,L) → Embedding(B,L,E) → RNN(B,L,H) → 掩码MaxPool(B,H)
          → LayerNorm(B,H) → Dropout(B,H) → Linear(B,C)

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -inf 后取最大值就干净了。

    结构刻意和课堂 8 类版保持一致（同名的层、同样的超参名），
    这样存出来的 .pt 能被 eval_langid.py 直接读 —— 作业评测口径统一。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        # 字符嵌入表：vocab_size 行 × embed_dim 列；padding_idx=0 → 0 号向量恒为 0 且不更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 单层 RNN：吃 (B,L,E)，吐 (B,L,H)；batch_first=True 表示第一维是 batch 而不是时间步
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # LayerNorm：对最后一维 H 做归一化，把池化后的句子向量拉回同一量纲，训练更稳
        self.norm = nn.LayerNorm(hidden_dim)
        # Dropout：训练时随机置 0，抑制"靠背句子"过拟合
        self.dropout = nn.Dropout(dropout)
        # 分类头：H → C，输出未过 softmax 的 logits
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """x: (B, L) 的整数张量，返回 (B, C) 的 logits（没做 softmax，CrossEntropyLoss 内部会做）。"""
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)，补齐位为 False；unsqueeze(-1) 是为了能和 (B,L,H) 对齐广播
        emb = self.embedding(x)                    # 查字符嵌入：(B, L, E)，E=64
        h, _ = self.rnn(emb)                       # 走完时间步：(B, L, H)；第二个返回值是最后一步的隐状态，这里用不上
        h = h.masked_fill(~mask, -1e9)             # 补齐位填成极小值：~(mask) 取反，让它们永远当不了最大值
        pooled = h.max(dim=1)[0]                   # (B,L,H) → (B,H) ← L 维被 max 消掉；[0] 取最大值、[1] 取索引（不要）
        pooled = self.dropout(self.norm(pooled))   # 先归一化再丢包：LayerNorm(B,H) → Dropout(B,H)
        return self.fc(pooled)                     # (B,C) 未过 softmax 的 logits，C=12
#endregion
#endregion

#region 8. 评测指标
#region 8.1 函数：整批预测                     【函数:predict(模型,加载器),return (预测id列表, 真实id列表)】
def predict(model, loader):
    """把加载器里所有样本过一遍，返回 (预测标签 id 列表, 真实标签 id 列表)。"""
    model.eval()                                   # 切评测模式：关掉 Dropout，LayerNorm 用统计值
    preds, golds = [], []                          # 两个空列表，分别装预测和真实
    with torch.no_grad():                          # 评测不需要梯度，省显存也更快
        for X, y in loader:                        # X:(B,L)  y:(B,)
            preds += model(X).argmax(dim=1).tolist()   # logits(B,C) → 取每行最大值下标(B,) → 转成 Python 列表
            golds += y.tolist()                    # 真实标签同样是 (B,) → 列表
    return preds, golds
#endregion

#region 8.2 函数：macro-F1                    【函数:macro_f1(预测,真实,类别数),return macro-F1】
def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）"""
    f1s = []                                       # 每一类的 F1 都往这里塞
    for c in range(num_classes):                   # 逐类算
        tp = sum(p == c and g == c for p, g in zip(preds, golds))   # 真阳：该是 c 也判成 c
        fp = sum(p == c and g != c for p, g in zip(preds, golds))   # 假阳：不是 c 却判成 c
        fn = sum(p != c and g == c for p, g in zip(preds, golds))   # 假阴：是 c 却判成别人
        if tp + fp + fn == 0:                      # 这一类在测试集里根本没出现，跳过
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0  # 精确率：判成 c 的里面有多少真的是 c
        rec = tp / (tp + fn) if tp + fn else 0.0   # 召回率：真的 c 里面有多少被找出来了
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)   # 调和平均
    return sum(f1s) / len(f1s)                     # 类别间算术平均
#endregion

#region 8.3 函数：混淆矩阵                     【函数:confusion(预测,真实,标签表),return 12x12 矩阵】
def confusion(preds, golds, labels):
    """行 = 真实标签，列 = 预测标签，mat[g][p] 表示"真实 g 被判成 p"的条数。"""
    n = len(labels)                                # 12
    mat = [[0] * n for _ in range(n)]              # 12×12 全 0 矩阵
    for p, g in zip(preds, golds):
        mat[g][p] += 1                             # 注意顺序：先真实后预测,比方说[1][11]=250
    return mat
#endregion

#region 8.4 函数：只打印最容易混淆的几格        【函数:print_confusion(矩阵,标签表,前几名)】
def print_confusion(mat, labels, top=5):
    """只打印错的格子，按错误数排序"""
    pairs = [(mat[g][p], labels[g], labels[p])     # (错误条数, 真实, 预测)
             for g in range(len(labels)) for p in range(len(labels)) if g != p]   # 只取对角线以外的格子
    pairs.sort(reverse=True)                       # 从错得最多的往下排
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:                  # 只看前 top 个
        if cnt == 0:                               # 一个都没错，后面不用看了
            break
        print(f"    {g} → {p}: {cnt} 条")

#region 8.5 函数：逐类打印 P / R / F1             【函数:per_class_report(预测,真实,标签表)】
def per_class_report(preds, golds, labels):
    """逐类打印 precision / recall / f1"""
    print("\n  逐类指标：")
    print(f"    {'标签':<6}{'样本':>6}{'precision':>12}{'recall':>10}{'f1':>10}")
    for c, lab in enumerate(labels):                # 类别 id 与名字一起遍历
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"    {lab:<6}{tp+fn:>6}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}")   # 样本数 = 该类真实条数 = tp+fn

#region 8.6 函数：按句子长度分档看准确率              【函数:accuracy_by_length(文本列表,预测,真实)】
def accuracy_by_length(texts, preds, golds):
    """
    语种识别最真实的难点在短文本（搜索词、商品标题）。
    数据是"四个长度桶等量采样"的，所以整体准确率会偏乐观；
    按长度分档一看，短句那一档通常掉得最明显 —— 实测 8~12 字符只有 0.938，20 字符以上都在 0.98 左右，
    """
    buckets = [(2, 7), (8, 12), (12, 20), (20, 30), (30, 999)]   # (下限, 上限)，左闭右开
    print("\n  按长度分档的准确率：")
    for lo, hi in buckets:
        idx = [i for i, t in enumerate(texts) if lo <= len(normalize(t)) < hi]   # 这一档所有样本的下标
        if not idx:                                # 这一档没样本就跳过
            continue
        hit = sum(preds[i] == golds[i] for i in idx)   # 这一档里预测对的条数
        print(f"    {lo:>3}~{hi if hi < 999 else '+':>3} 字符："
              f"{hit:4d}/{len(idx):4d} = {hit/len(idx):.3f}")
#endregion
#endregion

#region 9. 难度切片 + 开放集拒识
#region 9.1 函数：难度切片评测   【函数:run_hard_slices(模型,词表),data source:test_hard.csv】
def run_hard_slices(model, vocab):
    """
    难度集按 slice 分组：short（2~7 字符）/ nodeaccent（去重音）/ nokana（日语去假名）。
    只评测模型认识的 12 个标签（难度集里还有别的语种时自动跳过）。
    期望：nokana 那一档惨不忍睹，因为去掉假名后日语剩下的就是一堆汉字，和中文没区别 ——
    这不是"模型没学会"，是输入里已经没有区分信息了。
    """
    path = os.path.join(DATA_ROOT, "test_hard.csv")
    if not os.path.exists(path):                   # 没这个文件就安静跳过
        print("\n[跳过] 找不到 test_hard.csv")
        return
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))   # 读三列：text / label / slice
            if r["label"] in LABELS]               # 只保留 12 类之内的（v12 的难度集本来就只有这 12 类）
    label2id = {l: i for i, l in enumerate(LABELS)}

    print(f"\n=== 难度诊断集 {len(rows)} 条 ===")
    by_slice = {}                                  # slice 名 → 该组样本列表
    for r in rows:
        by_slice.setdefault(r["slice"], []).append(r)

    model.eval()
    for name in ("short", "nodeaccent", "nokana"): # 固定顺序打印，方便横竖对比
        sub = by_slice.get(name, [])
        if not sub:
            print(f"  [{name}] 无可用样本")
            continue
        ids = torch.tensor([encode(r["text"], vocab) for r in sub])   # (N, 40)
        with torch.no_grad():
            preds = model(ids).argmax(dim=1).tolist()                 # (N,) 预测 id
        golds = [label2id[r["label"]] for r in sub]                   # (N,) 真实 id
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)  # 这一档的准确率
        print(f"  [{name}] {len(sub):4d} 条  accuracy = {acc:.3f}")
        hit, per = Counter(), Counter()            # 每类命中数 / 每类样本数
        for p, g in zip(preds, golds):
            per[g] += 1
            if p == g:
                hit[g] += 1
        print("      " + "  ".join(f"{LABELS[g]}:{hit[g]}/{per[g]}" for g in sorted(per)))

#endregion
#region 9.2 函数：开放集拒识   【函数:run_openset(模型,词表),data source:test_openset.csv + dev.csv】
def run_openset(model, vocab, dev_rows):
    """
    训练里没出现过的语种（el/bg/sw/ur）喂给模型，看它敢不敢说"不认识"。
    做法：先用验证集标定阈值 —— 取验证集上 max(softmax) 的 5% 分位数，
    意味着"正常输入里有 95% 的置信度高于它"。测试时低于阈值就判 unknown。
    """
    path = os.path.join(DATA_ROOT, "test_openset.csv")
    if not os.path.exists(path):
        print("\n[跳过] 找不到 test_openset.csv")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    model.eval()

    def max_prob(texts):
        """批量算置信度：返回每个样本 softmax 最大值及其下标。"""
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):          # 分批前向，避免一次塞太多
                batch = torch.tensor([encode(t, vocab) for t in texts[i:i + 256]])
                out.append(torch.softmax(model(batch), dim=1))
        prob = torch.cat(out, dim=0)                     # (N, 12)
        return prob.max(dim=1)[0], prob.argmax(dim=1)    # (最大值, 对应类别)

    dev_conf, _ = max_prob([t for t, _ in dev_rows])     # 验证集置信度，用来定阈值
    thr = torch.quantile(dev_conf, 0.05).item()          # 5% 分位数当阈值
    os_conf, os_pred = max_prob([r["text"] for r in rows])   # 开放集置信度与预测

    keep = (dev_conf >= thr).float().mean().item()       # 正常输入保留率（应接近 95%）
    rej = (os_conf < thr).float().mean().item()          # 开放集拒识率（越高越好）
    print(f"\n=== 开放集拒识（{len(rows)} 条未见语种）===")
    print(f"  阈值（验证集 max-prob 的 5% 分位）= {thr:.3f}")
    print(f"  正常输入保留率 = {keep:.1%}   开放集拒识率 = {rej:.1%}")
    print(f"  开放集平均置信度 = {os_conf.mean():.3f} （正常输入 = {dev_conf.mean():.3f}）")

    print("\n  没见过的语种被认成了谁：")
    pred_list = os_pred.tolist()
    for lab in sorted({r["label"] for r in rows}):       # 逐个开放集语种看
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        c = Counter(LABELS[pred_list[i]] for i in idx)   # 统计被判成了哪些已知语种
        top = ", ".join(f"{k}:{v}" for k, v in c.most_common(3))
        print(f"    {lab}（{len(idx)} 条，平均置信度 {os_conf[idx].mean().item():.2f}）→ {top}")
#endregion
#endregion

#region 10. 训练主流程
#region 10.1 函数：入口 main()   【函数:main(),return None，副作用：打印日志 + 存 .pt】
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR, help="数据目录，默认 data/langid/v12")
    ap.add_argument("--vocab", default="", help="现成字符表文件；不填就从训练集现场统计")
    ap.add_argument("--save-vocab", action="store_true",
                    help="把现场统计的字符表写成 data/langid/vocab/vocab_v12.txt")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default=DEFAULT_SAVE, help="模型存档路径")
    ap.add_argument("--no-extra", action="store_true", help="跳过难度切片与开放集评测")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)             # 限制线程数：小模型单线程更快
    random.seed(SEED)                               # Python 侧随机种子
    torch.manual_seed(SEED)                         # torch 侧随机种子（参数初始化 + shuffle）

    #region 10.1.1 读数据 + 标签自检
    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")
    check_labels(train_rows, "train")               # 三个划分都查一遍，标签必须正好是 LABELS 这 12 个
    check_labels(dev_rows, "dev")
    check_labels(test_rows, "test")

    label2id = {l: i for i, l in enumerate(LABELS)} # 标签名 → 0~11，顺序即输出格子的顺序
    num_classes = NUM_CLASSES
    #endregion

    #region 10.1.2 字符表：优先读文件，否则现场统计
    if args.vocab:                                  # 指定了就按文件来（前后两次实验表一致）
        vocab = load_vocab(args.vocab)
        vocab_src = os.path.basename(args.vocab)
    else:
        # 只从**训练集**统计：dev/test 是"没见过的数据"，它们的字符不该提前泄露进词表
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        vocab_src = "现场统计自 train.csv"
        if args.save_vocab:                         # 字符表存档
            out = os.path.join(DATA_ROOT, "vocab", "vocab_v12.txt")
            save_vocab(vocab, freq, out)
            print(f"字符表已写出：{out}")
    #endregion

    #region 10.1.3 可选：每类只取 N 条（演示数据少 → 过拟合）
    if args.per_class:                              # 0 表示不裁剪
        by_label = {}                               # 标签 → 该类的全部句子
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)     # 采样后按标签聚在一起了，重新打散
    #endregion

    #region 10.1.4 打印数据概况
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（{vocab_src}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")   # 12 类 → 8.3%，这就是"什么都没学会"的分数线
    #endregion

    #region 10.1.5 三个 DataLoader
    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)   # 训练集打乱：避免模型记住样本顺序
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)  # 验证集不用打乱
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)
    #endregion

    #region 10.1.6 建模型 / 损失 / 优化器
    model = LangIDClassifier(len(vocab), num_classes)                    # 词表大小 → 嵌入表行数
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")  # 参数量：可训练标量总数

    criterion = nn.CrossEntropyLoss()                                    # 多分类标准损失：内部做 log_softmax + NLL
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)              # Adam：自适应学习率，小模型上比 SGD 省心
    #endregion

    #region 10.1.7 训练循环
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()                                # 切训练模式：打开 Dropout
        total = 0.0                                  # 本轮所有 batch 的 loss 累加
        for X, y in train_loader:                    # X:(B,L)  y:(B,)
            logits = model(X)                        # 前向：(B,L) → (B,12)
            loss = criterion(logits, y)              # 一个 batch 的平均交叉熵
            optimizer.zero_grad()                    # 清梯度（PyTorch 默认会累加）
            loss.backward()                          # 反向传播，算每个参数的梯度
            optimizer.step()                         # 按梯度更新参数
            total += loss.item()                     # .item() 把 0 维张量取成 Python 浮点数
        preds, golds = predict(model, dev_loader)    # 每个 epoch 在验证集上看一眼
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)   # 验证集准确率
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")
    #endregion

    #region 10.1.8 测试集总评
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)       # 整体准确率
    f1 = macro_f1(preds, golds, num_classes)                           # 12 类平均 F1
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=6)     # 12 类，多看两格
    per_class_report(preds, golds, LABELS)
    accuracy_by_length([t for t, _ in test_rows], preds, golds)
    #endregion

    #region 10.1.9 加餐评测：难度切片 + 开放集
    if not args.no_extra:
        run_hard_slices(model, vocab)               # short / nodeaccent / nokana
        run_openset(model, vocab, dev_rows)         # 4 个训练里没出现过的语种
    #endregion

    #region 10.1.10 推理示例：拿测试集前 5 条看概率
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:5]]           # 注意：这几条是测试集原文，模型训练时没见过
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))   # 单条也要包成 (1, 40)
            prob = torch.softmax(logits, dim=1)[0]                # (1,12) → (12,) 概率分布
            pid = int(prob.argmax())                              # 概率最大的那一格
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")
    #endregion

    #region 10.1.11 存档
    if args.save:
        torch.save({"state_dict": model.state_dict(),   # 只存参数，不存整个类（跨版本更稳）
                    "vocab": dict(vocab),               # 字符 → id，推理时新句子要查它
                    "labels": LABELS,                   # 12 个标签的顺序，输出格子与它对号
                    "maxlen": MAX_LEN,                  # 截断长度，推理必须和训练一致
                    "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")
    #endregion


if __name__ == "__main__":                          # 当作脚本运行时才跑 main（被 import 时不跑）
    main()
#endregion
#endregion

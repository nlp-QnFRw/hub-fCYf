# -*- coding: utf-8 -*-
"""train_sft.py —— 字符级 SFT（监督微调）：把「续写器」调成「答题器」

【这个文件想让你看懂的一件事】
    和 week4 的 transformer_lm.py 是同一个 Transformer，只改了两件事：
        ① 数据的组织方式：每条样本是一句 (指令, 回答)，用特殊标记包起来
        ② loss 的落点：指令段打 -100 不算 loss，只学「怎么答」
    SFT 的全部改动就是这两条 —— 结构不动、目标不动（还是 next-token），只改数据和 loss 位置。

【数据】同目录下的 sft_train.jsonl / sft_val.jsonl
    每行 {"instruction": "...", "output": "..."}
    由 prepare_sft.py 从 sft_train.zip 清洗而来。

【训练序列长这样】
    <|user|>给定两个数字，计算它们的乘积。 12 和 34<|assistant|>这两个数的乘积是 408。<|end|>
    ╰────────────── labels = -100，不学 ──────────────╯╰──── 只在这段算 loss ────╯

【用法】
    python train_sft.py                                    # 训练（默认 10 epoch）
    python train_sft.py --epochs 3                         # 快跑
    python train_sft.py --chat "12 和 34 的乘积是多少？"     # 训完直接用一句话问
    python train_sft.py --interactive                      # 交互模式，一行一句
    python train_sft.py --init-from "..\\week04\\best_model_tf.pt" --save best_sft_from_pretrain.pt
【阅读顺序】建议从上往下：常量 → 数据（最关键的 SFTDataset）→ 模型 → 训练 → 推理 → 主流程
产出 best_sft_model.pt（权重 + 字表 + 超参 + 模板，可独立加载）
"""
import os                                          # 拼路径、判断文件是否存在
import json                                        # 读 JSONL，一行一个 JSON
import math                                        # math.exp：loss 换算成 PPL
import time                                        # 计时：看看一个 epoch 要多久
import argparse                                    # 命令行参数
import random                                      # 数据打散、随机种子

import torch                                       # 张量、自动求导、GPU
import torch.nn as nn                              # 各种网络层（Embedding / Linear / LayerNorm…）
from torch.utils.data import Dataset, DataLoader    # 标准数据接口 + 批量加载

# 本文件所在目录。数据和模型默认都放在这里，所以从任何目录运行都能找到文件
HERE = os.path.dirname(os.path.abspath(__file__))

# ── 对话模板 ───────────────────────────────────────────────────────
# 注意：这是**字符级**模型，所以 "<|user|>" 不是「一个特殊 token」，
#       而是 8 个普通字符（< | u s e r | >），会被逐个编码。
#       换成 BPE 分词器时，这些标记才会变成单个 token —— 但作用完全一样。
USER = "<|user|>"           # 用户说话的起点
ASSISTANT = "<|assistant|>"  # 助手开始回答的起点（模型看到它就该出答案了）
END = "<|end|>"             # 这一轮说完的标记：模型学会在这里停下
PAD = "<pad>"               # 补齐用的占位符，它自己占用编号 0

MAX_LEN = 160               # 序列最长 160 字符：数据里最长 111 字 + 模板 28 字，留余量
PAD_ID = 0                  # <pad> 的编号固定为 0，后面 embedding 也靠这个 id 识别补齐位

# ══════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════
def read_jsonl(path):
    rows = []                                            # 结果容器，元素是 (指令, 回答) 二元组
    with open(path, encoding="utf-8") as f:              # 显式指定 utf-8：Windows 默认 gbk，中文会炸
        for line in f:                                   # 逐行读，几千条不需要一次全读进内存
            line = line.strip()                          # 去掉行尾换行和首尾空白
            if line:                                     # 跳过空行（文件末尾常有一个）
                obj = json.loads(line)                   # 一行 = 一个 JSON 对象
                rows.append((obj["instruction"], obj["output"]))   # 只要这两个字段
    return rows


def build_vocab(rows):
    """字表：从「模板 + 指令 + 回答」全量统计，按出现次数从高到低编号，<pad> 固定 0。"""
    from collections import Counter                      # 只在函数里 import，属于"用到才引"
    freq = Counter()                                     # 字符 → 出现次数
    for q, a in rows:                                    # 遍历每条样本
        freq.update(USER + q + ASSISTANT + a + END)      # 注意：模板字符也要统计进去
    # 排序规则：出现次数多的在前；次数一样时按字符编码排，保证每次跑结果完全一致
    chars = [c for c, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    char2idx = {PAD: PAD_ID}                             # 先占住 0 号给 <pad>
    for c in chars:                                      # 剩下的字符依次编号
        if c not in char2idx:                            # 防御：理论上不会重复，加了更安全
            char2idx[c] = len(char2idx)                  # 新编号 = 当前表长
    idx2char = {i: c for c, i in char2idx.items()}       # 反查表：生成时需要 id → 字符
    return char2idx, idx2char


class SFTDataset(Dataset):
    """每条样本 = 完整的一问一答，并在构造时就把 labels 算好。

    ⚠ 两个最容易写错、而且错了会「静默变好」的地方（下面代码里都有注释）：

    ① labels 必须往后错一位
       第 t 个位置的输出，预测的是第 t+1 个字符。如果写成 labels[t] = ids[t]，
       模型只要把当前位置的输入「抄」过来就能把 loss 压到 0 ——
       于是 Train PPL 掉到 1.00、Val PPL 也接近 1.0，看着完美，
       但一生成就露馅（会吐出一串重复字符）。
       判断口诀：**语言模型的 loss 逼近 0，几乎一定是泄漏，不是学得好。**

    ② 指令段打 -100
       prompt 的字符不作为预测目标。交叉熵的 ignore_index=-100 会跳过这些位置，
       模型只学「看到指令后该怎么答」，不去背「指令本身长什么样」。
    """

    def __init__(self, rows, char2idx):
        self.samples = []                                        # 元素是 (ids, labels) 两条等长列表
        for q, a in rows:                                        # 逐条样本处理
            prompt = USER + q + ASSISTANT                        # 前半段：<|user|>问题<|assistant|>
            answer = a + END                                     # 后半段：回答<|end|>
            ids = [char2idx.get(c, 1) for c in prompt + answer][:MAX_LEN]   # 逐字符查表；表外字符给 1；超长截断
            n_prompt = min(len(prompt), len(ids))                # 指令段实际长度（考虑截断）

            labels = [-100] * len(ids)                           # 先全部设为「不算 loss」
            # 位置 t 的预测目标是 ids[t+1]；只保留「目标落在回答段」的那些位置。
            # 从 n_prompt-1 开始：那一位的输出，预测的正是回答的第一个字。
            # 到 len(ids)-1 结束：最后一位没有「下一个字」，所以不能算。
            for t in range(max(0, n_prompt - 1), len(ids) - 1):
                labels[t] = ids[t + 1]                           # 对齐到下一个字符（这就是「错一位」）
            self.samples.append((ids, labels))                   # 存起来

    def __len__(self):
        return len(self.samples)                                 # DataLoader 靠它知道有多少条

    def __getitem__(self, i):
        ids, labels = self.samples[i]                            # 取出第 i 条
        return torch.tensor(ids), torch.tensor(labels)           # 转成张量（Python 列表不能直接进模型）


def collate(batch):
    """DataLoader 的组批函数：一个 batch 里句子长短不一，要补齐成矩形。

    补齐位在 ids 里填 0（<pad>），在 labels 里填 -100（不参与 loss）。
    """
    max_len = max(x.size(0) for x, _ in batch)                   # 这个 batch 里最长的那条有多长
    xs = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)   # 输入：先整块填 <pad>
    ys = torch.full((len(batch), max_len), -100, dtype=torch.long)     # 标签：先整块填 -100
    for i, (x, y) in enumerate(batch):                           # 逐条贴回去
        xs[i, :x.size(0)] = x                                    # 真实长度部分覆盖掉 <pad>
        ys[i, :y.size(0)] = y                                    # 真实标签覆盖掉 -100
    return xs, ys                                                # 形状都是 (batch, max_len)

# ══════════════════════════════════════════════════════════════════
# 模型（和 week4 transformer_lm.py 完全一致的结构，一个层都没加）
# ══════════════════════════════════════════════════════════════════
class TransformerLM(nn.Module):
    """字符级因果 Transformer：Embedding + 位置编码 + N×EncoderLayer + Linear。"""

    def __init__(self, vocab_size, d_model=192, nhead=4, num_layers=4,
                 ffn=512, dropout=0.1, max_len=MAX_LEN):
        super().__init__()                                       # 初始化 nn.Module 的底层结构
        self.tok = nn.Embedding(vocab_size, d_model)             # 字表大小 × 隐层维度：每个字一个向量
        self.pos = nn.Embedding(max_len, d_model)                # 位置编码：让模型知道「这是第几个字」
        layer = nn.TransformerEncoderLayer(                      # 一层标准 Transformer 编码块
            d_model=d_model,                                     # 隐层维度
            nhead=nhead,                                         # 多头注意力的头数（160/4=40 维每头）
            dim_feedforward=ffn,                                 # 前馈层中间维度
            dropout=dropout,                                     # 两个子层后的 dropout
            batch_first=True,                                    # 输入形状是 (batch, 序列, 维度)
            norm_first=True,                                     # pre-LN：先归一化再进子层，训练更稳
            activation="gelu")                                   # 激活函数用 GELU（BERT/GPT 的常规选择）
        self.encoder = nn.TransformerEncoder(layer, num_layers,  # 堆 N 层
                                             enable_nested_tensor=False)   # 关掉优化，避免和自定义 mask 打架
        self.norm = nn.LayerNorm(d_model)                        # 输出前的最后一次归一化
        self.fc = nn.Linear(d_model, vocab_size)                 # 投影回词表：每个位置输出「下一个字」的打分
        # 因果掩码：上三角为 True 表示「不许看」。它是下三角=能看自己及以前，上三角=看不到未来。
        # 漏了它模型就能偷看答案，PPL 会假性地变得极低 —— 这是语言模型最经典的错误之一。
        self.register_buffer(                                    # register_buffer：跟着模型走，但不参与训练
            "causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool),
                                 diagonal=1),                        # triu(diagonal=1) = 严格上三角
            persistent=False)                                    # 不存进 state_dict（省空间，加载时重建即可）

    def forward(self, x):
        """x: (batch, 序列长度) 的整数 id → 输出 (batch, 序列长度, 词表大小) 的打分。"""
        length = x.size(1)                                       # 当前输入有多长（可能小于 MAX_LEN）
        pos = torch.arange(length, device=x.device).unsqueeze(0)  # [0,1,2,...,length-1]，形状 (1, length)
        h = self.tok(x) + self.pos(pos)                          # 字向量 + 位置向量（直接相加）
        h = self.encoder(h, mask=self.causal[:length, :length])   # 只取用得到的那一块掩码
        return self.fc(self.norm(h))                             # 每个位置都吐一个「下一个字」的分布

# ══════════════════════════════════════════════════════════════════
# 训练 / 评估
# ══════════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """跑完一遍数据。train=True 时更新参数；False 时只算指标。"""
    model.train(train)                                           # 切模式：训练态（dropout 生效）/ 推理态（关闭）
    total_loss, total_tokens = 0.0, 0                            # 按 token 加权累计，避免最后一批偏小造成偏差
    for x, y in loader:                                          # 一批一批取
        x, y = x.to(device), y.to(device)                        # 数据搬到 GPU（CPU 上是原地不动）
        logits = model(x)                                        # 前向：得到每个位置的词表打分
        # 把 (batch, seq, vocab) 拉平成 (batch*seq, vocab)，
        # 标签也拉平成 (batch*seq,)，交叉熵会自动跳过 -100 的位置
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:                                                # 只有训练时才反传
            optimizer.zero_grad()                                # 清掉上一步的梯度（不清会累加）
            loss.backward()                                      # 反向传播算梯度
            optimizer.step()                                     # 更新参数
        n = (y != -100).sum().item()                             # 本批真正参与 loss 的位置数
        total_loss += loss.item() * n                            # loss 是这些位置的平均，乘回去才是总和
        total_tokens += n                                        # 累计 token 数
    avg = total_loss / max(1, total_tokens)                      # 再除回来，得到整个 epoch 的平均 loss
    return avg, math.exp(avg)                                    # PPL = e^loss，可理解为「每步在几个字里犹豫」


def answer_question(model, char2idx, idx2char, question, device, max_new=60,
                    rep_penalty=1.3, early_stop_after=45, stats=None):
    """贪心解码：喂 <|user|>问题<|assistant|>，一个字一个字往外蹦，直到 <|end|> 或到上限。

    两个后处理，都是为了治「车轱辘话」——实测把严重重复从 39.3% 压到 1.6%：

    · rep_penalty：惩罚「刚生成过的字」。纯贪心解码时，一旦某个字被打分最高，
      它就会一直吐同一个字（"是星期是星期是星期…"）。把最近出现过的字打分除以 1.3，
      就能逼它换字。注意只惩罚**生成出来**的部分，不动 prompt ——
      否则答案里想引用题目中的数字都会被压。

    · early_stop_after：兜底刹车。只有生成超过 45 个字还没停时才生效（数据里答案最长 64、
      中位 22），用来救那一小撮始终吐不出 <|end|> 的情况。
      不能一遇句号就停 —— 11.3% 的正常答案中间本来就有句号，那样会截断。
    """
    prompt = USER + question + ASSISTANT                         # 拼成训练时见过的格式
    ids = [char2idx[c] for c in prompt if c in char2idx]         # 逐字查表；字表外的字符直接跳过
    if not ids:                                                  # 一个可用字符都没有（比如纯 emoji）
        return "（这句里没有字表内的字符）"
    n_prompt = len(ids)                                          # 记住 prompt 有多长：之后只看新生成的部分
    out = ""                                                     # 累积已生成的文本，用来判断是否出现 <|end|>
    model.eval()                                                 # 推理模式：关掉 dropout，保证结果可复现
    for _ in range(max_new):                                     # 最多生成 max_new 个字符
        x = torch.tensor([ids[-MAX_LEN:]], device=device)        # 只保留最近 MAX_LEN 个字符（不超过位置编码上限）
        with torch.no_grad():                                    # 推理不需要梯度，省显存也更快
            logits = model(x)[0, -1].float()                     # 最后一个位置对「下一个字」的打分
        if rep_penalty != 1.0:                                   # 重复惩罚开关（1.0 = 关闭）
            for t in set(ids[n_prompt:][-40:]):                  # 只看已生成部分的最近 40 个字
                # 标准做法：正打分除、负打分乘，效果都是「往下压」
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
        nxt = int(logits.argmax())                               # 贪心：取打分最高的那个字
        if nxt == PAD_ID:                                        # 万一生成出 <pad>，说明模型已经乱掉了
            if stats is not None:
                stats["stop"] = "pad"
            break                                                # 直接停，别输出占位符
        ch = idx2char[nxt]                                       # id 换回字符
        ids.append(nxt)                                          # 拼进序列，下一步继续往后生成
        out += ch                                                # 同时拼进可读文本
        if out.endswith(END):                                    # 模型自己说「我说完了」
            if stats is not None:                                # eval_sft.py 用它统计「停住率」
                stats["stop"] = "end"
            return out[:-len(END)]                               # 去掉 <|end|> 再返回
        if len(out) >= early_stop_after and ch in "。？！\n":     # 兜底：写太长了还没停，遇到句末标点就收
            if stats is not None:
                stats["stop"] = "early"
            return out
    if stats is not None:
        stats["stop"] = "max"                                    # 到上限还没说完
    return out                                                   # 到上限还没说完，就返回已生成的部分

# ══════════════════════════════════════════════════════════════════
# 存档 / 读档
# ══════════════════════════════════════════════════════════════════
def save_ckpt(path, model, char2idx, idx2char, args, extra=None):
    """把「重建模型需要的一切」打包存下来，这样 .pt 可以脱离本脚本被读取。

    ⚠ 结构超参必须**从模型本身反读**，不能抄命令行变量：
      用 --init-from 时，网络结构是预训练给定的（比如 192/512/4 层），
      而命令行里的 --d_model 等仍是默认值（160/384/3），抄错就会存出一个
      「记录的形状和权重对不上」的废档 —— 这个坑在加 --init-from 时踩过一次。
    """
    first = model.encoder.layers[0]                              # 拿第一层来反读结构
    cfg = {"d_model": model.tok.embedding_dim,                   # Embedding 的维度 = d_model
           "nhead": first.self_attn.num_heads,                   # 多头注意力的头数
           "num_layers": len(model.encoder.layers),              # 有多少层
           "ffn": first.linear1.out_features,                    # 前馈层中间维度
           "dropout": first.dropout.p,                           # dropout 比例（不影响权重，仅记录）
           "max_len": model.pos.num_embeddings,                  # 位置编码有多少个位置
           "vocab_size": len(char2idx)}                          # 字表大小
    saved_args = dict(vars(args))                                # 把命令行参数整份存下（方便复现）
    saved_args["model"] = "transformer-sft"                      # 标记模型类型，和课堂脚本的存档约定保持一致
    payload = {"model_state": model.state_dict(),                # 权重
               "char2idx": char2idx,                             # 字 → id
               "idx2char": idx2char,                             # id → 字
               "args": saved_args,                               # 命令行参数
               "arch": "transformer-sft",                        # 结构名
               "model_cfg": cfg,                                 # 结构超参（从模型反读）
               "template": {"user": USER, "assistant": ASSISTANT, "end": END}}   # 对话模板
    if extra:                                                    # 额外信息（最佳 PPL、第几轮）
        payload.update(extra)
    torch.save(payload, path)                                    # 落盘


def load_ckpt(path, device):
    """按存档里记录的形状把模型重建出来，并加载权重。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)   # weights_only=False：允许存 dict 等对象
    cfg = ckpt["model_cfg"]                                      # 取出结构超参
    model = TransformerLM(cfg["vocab_size"], cfg["d_model"], cfg["nhead"],
                          cfg["num_layers"], cfg["ffn"], cfg["dropout"],
                          cfg["max_len"]).to(device)             # 按原形状建网，搬到设备上
    model.load_state_dict(ckpt["model_state"])                   # 灌权重（形状对不上会在这里报错）
    model.eval()                                                 # 推理模式
    return model, ckpt["char2idx"], ckpt["idx2char"], ckpt        # 连字表和原始 ckpt 一起返回


# ──────────────────────────────────────────────────────────────────
# 从预训练模型出发做 SFT（--init-from）
# 这两件事比看起来麻烦：预训练模型的字表和我们的不一样，必须先对齐再拷权重
# ──────────────────────────────────────────────────────────────────
def merge_pretrained_vocab(base_ckpt, rows):
    """把预训练字表扩成 SFT 能用的字表，返回 (新字表, 反查表, 预训练字数, 新增字符列表)。

    编号规则：0=<pad>，1..N=预训练原字表（顺序不动），N+1..=新增字符。
    预训练的 0 号是 "\\n"，我们的 0 号要留给 <pad>，所以所有预训练字整体往后挪一位。
    """
    base_chars = [c for c, _ in sorted(base_ckpt["char2idx"].items(), key=lambda kv: kv[1])]  # 按原编号排好
    need = set()                                                 # SFT 数据里出现过的全部字符
    for q, a in rows:
        need |= set(USER + q + ASSISTANT + a + END)              # 连模板一起算
    extra = sorted(need - set(base_chars))                       # 预训练没见过的（模板里的 < | > 都在这里）
    char2idx = {PAD: PAD_ID}                                     # 0 号给 <pad>
    for c in base_chars:                                         # 1..N：预训练字表原封不动
        char2idx[c] = len(char2idx)
    for c in extra:                                              # 新增字符排在最后
        char2idx[c] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(base_chars), extra


def init_from_pretrained(base_ckpt, char2idx, n_base, dropout):
    """按新字表建网，然后把预训练权重「平移 + 补齐」地灌进去。"""
    cfg = base_ckpt["model_cfg"]                                 # 预训练模型的结构超参
    sd = base_ckpt["model_state"]                                # 预训练权重
    model = TransformerLM(len(char2idx),                         # 新字表更大
                          d_model=cfg["d_model"],                # 结构必须和预训练一致，否则权重对不上
                          nhead=cfg["nhead"],
                          num_layers=cfg["num_layers"],
                          ffn=cfg["ffn"],
                          dropout=dropout,                       # dropout 只是训练超参，可以重设
                          max_len=MAX_LEN)                       # 位置编码扩容到 MAX_LEN
    merged = {k: v.clone() for k, v in model.state_dict().items()}   # 以「新建网」的形状为底

    # ① 字嵌入：预训练第 i 行 → 新表第 i+1 行（因为 0 号让给了 <pad>）
    merged["tok.weight"].zero_()                                 # 0 号 <pad> 保持全零（Embedding 的 padding_idx 也不会更新它）
    merged["tok.weight"][1:1 + n_base] = sd["tok.weight"]        # 预训练 4000 行整体后移一位
    # ② 输出层同理：每一行对应一个字，也要后移
    merged["fc.weight"][1:1 + n_base] = sd["fc.weight"]
    merged["fc.bias"][1:1 + n_base] = sd["fc.bias"]
    # ③ 位置编码：预训练只有 64 个位置，这里只拷前 64 行，多出来的位置保持随机初始化
    keep = min(cfg["max_len"], MAX_LEN)
    merged["pos.weight"][:keep] = sd["pos.weight"][:keep]
    # ④ 其余权重（多层 Transformer、LayerNorm）形状完全一致，直接整块拷
    for k in merged:
        if k.startswith(("tok.", "fc.", "pos.")):                 # 前面已单独处理过的三类跳过
            continue
        merged[k] = sd[k]
    model.load_state_dict(merged)                                # 灌进新建的网络
    return model

# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="字符级 SFT：把续写器调成答题器")
    ap.add_argument("--data-dir", default=HERE)                  # 数据目录，默认脚本自己所在的目录
    ap.add_argument("--save", default=os.path.join(HERE, "best_sft_model.pt"))   # 模型存哪
    ap.add_argument("--epochs", type=int, default=10)             # 训练轮数（小数据 5~10 轮就到顶了）
    ap.add_argument("--batch_size", type=int, default=32)         # 每批多少条
    ap.add_argument("--lr", type=float, default=1e-3)             # 学习率
    ap.add_argument("--d_model", type=int, default=160)           # 隐层维度
    ap.add_argument("--nhead", type=int, default=4)               # 注意力头数
    ap.add_argument("--num_layers", type=int, default=3)          # Transformer 层数
    ap.add_argument("--ffn", type=int, default=384)               # 前馈层中间维度
    ap.add_argument("--dropout", type=float, default=0.3)         # dropout 比例（小数据要压一压过拟合）
    ap.add_argument("--seed", type=int, default=42)               # 随机种子，保证可复现
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--chat", default=None, help="不训练，直接用现有模型回答一句话")
    ap.add_argument("--interactive", action="store_true", help="不训练，进交互模式")
    ap.add_argument("--init-from", default=None,
                    help="从一个预训练模型出发做 SFT（比如 week4 的 best_model_tf.pt）；不给就从零训")
    args = ap.parse_args()                                       # 解析命令行

    device = torch.device(                                       # 选设备
        "cuda" if args.device == "auto" and torch.cuda.is_available()   # auto 且有 GPU → 用 GPU
        else "cpu" if args.device == "auto" else args.device)           # auto 但没 GPU → CPU；否则听用户的

    # ── 分支一：只做推理（不训练）────────────────────────────────
    if args.chat is not None or args.interactive:                # 用户要么给了一句话，要么要交互
        if not os.path.exists(args.save):                        # 还没训过
            raise SystemExit(f"还没训练过，找不到 {args.save}\n  先跑：python train_sft.py")
        model, char2idx, idx2char, ckpt = load_ckpt(args.save, device)   # 读档
        print(f"模型：{os.path.basename(args.save)}  "
              f"词表 {len(char2idx)}  （训练时最佳验证 PPL "
              f"{ckpt.get('best_val_ppl', float('nan')):.2f}）")   # 打印一份模型身份信息
        if args.chat is not None:                                # 单句模式
            print(f"\n问：{args.chat}")
            print(f"答：{answer_question(model, char2idx, idx2char, args.chat, device)}")
        else:                                                    # 交互模式
            print("\n输入问题回车，空行或 :q 退出。")
            while True:                                          # 一直读，直到用户退出
                try:
                    q = input("\n问> ").strip()                   # 读一行
                except (EOFError, KeyboardInterrupt):             # Ctrl-D / Ctrl-C
                    print("\n再见。")
                    break
                if not q or q in (":q", "quit", "exit"):          # 空行或退出命令
                    break
                print(f"答：{answer_question(model, char2idx, idx2char, q, device)}")
        return                                                   # 推理分支做完就结束，不进训练

    # ── 分支二：训练 ──────────────────────────────────────────────
    torch.manual_seed(args.seed)                                 # 固定 torch 的随机种子（初始化、shuffle 都受影响）
    random.seed(args.seed)                                       # 固定 python 的随机种子

    train_rows = read_jsonl(os.path.join(args.data_dir, "sft_train.jsonl"))   # 读训练集
    val_rows = read_jsonl(os.path.join(args.data_dir, "sft_val.jsonl"))       # 读验证集

    base_ckpt = None                                             # 要接预训练模型时，放这里
    if args.init_from:                                           # ① 从 week4 的预训练模型出发
        if not os.path.exists(args.init_from):
            raise SystemExit(f"找不到预训练模型：{args.init_from}")
        base_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        char2idx, idx2char, n_base, extra = merge_pretrained_vocab(base_ckpt, train_rows + val_rows)
        print(f"从预训练模型初始化：{os.path.basename(args.init_from)}")
        print(f"  字表 {n_base} → {len(char2idx)}（新增 {len(extra)} 个字符，"
              f"含模板 < | > 等）")
    else:                                                        # ② 从零开始
        char2idx, idx2char = build_vocab(train_rows + val_rows)  # 字表用两边一起统计（略乐观，但简单）

    print(f"device: {device}  model: TRANSFORMER(SFT)"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))   # 有 GPU 就报型号
    print(f"训练 {len(train_rows)} 条 / 验证 {len(val_rows)} 条   字表 {len(char2idx)}")
    print("模板：" + USER + "{指令}" + ASSISTANT + "{回答}" + END)   # 把模板打出来，一眼看清训练序列长什么样

    train_loader = DataLoader(SFTDataset(train_rows, char2idx), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)   # 训练集：每轮打乱
    val_loader = DataLoader(SFTDataset(val_rows, char2idx), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)    # 验证集：不用打乱
    print(f"batch {args.batch_size} → 每 epoch {len(train_loader)} 步\n")

    if base_ckpt is not None:                                    # 从预训练权重出发（结构跟预训练一致）
        model = init_from_pretrained(base_ckpt, char2idx, n_base,
                                     args.dropout).to(device)    # 平移字表、补位置编码
    else:                                                        # 从零开始（结构听命令行）
        model = TransformerLM(len(char2idx), d_model=args.d_model, nhead=args.nhead,
                              num_layers=args.num_layers, ffn=args.ffn,
                              dropout=args.dropout).to(device)   # 建网并搬到设备上
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")   # 参数量：衡量模型大小

    criterion = nn.CrossEntropyLoss(ignore_index=-100)           # ← SFT 的关键：-100 的位置直接跳过
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam 是最常用的默认优化器

    best_ppl = float("inf")                                      # 记录历史最好的验证 PPL
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}")                   # 表头
    print("-" * 56)
    t0 = time.time()                                             # 开始计时
    for epoch in range(1, args.epochs + 1):                      # 逐轮训练
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, True)   # 训练一遍
        with torch.no_grad():                                    # 验证不需要梯度
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, False)   # 验证一遍
        mark = "  *" if va_ppl < best_ppl else ""                # 比历史最好还好就标记一下
        if va_ppl < best_ppl:                                    # 刷新记录
            best_ppl = va_ppl                                    # 更新最好成绩
            save_ckpt(args.save, model, char2idx, idx2char, args,
                      extra={"best_val_ppl": best_ppl, "epoch": epoch})   # 只存最好的那一版
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{mark}")        # 打印这一轮的成绩
    print(f"\n训练耗时 {time.time() - t0:.0f} 秒。最佳验证 PPL {best_ppl:.2f}"
          f"  已保存 {args.save}")

    # ── 训完立刻验货：拿验证集里的问题问它 ──────────────────────
    model, char2idx, idx2char, _ = load_ckpt(args.save, device)  # 重新读回「最好那一版」再测
    print("\n" + "=" * 60)
    print("效果抽查（问句取自验证集）")
    print("=" * 60)
    for q, gold in val_rows[:6]:                                 # 抽前 6 条看效果
        pred = answer_question(model, char2idx, idx2char, q, device)   # 让模型答
        print(f"\n问：{q}")
        print(f"答：{pred}")
        print(f"标准答案：{gold}")                                # 对比标准答案，肉眼看看差在哪


if __name__ == "__main__":
    main()                                                       # 直接运行本文件时才执行

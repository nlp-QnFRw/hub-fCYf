"""
model.py —— 一个最小的 Decoder-only 大模型（GPT），从零手写，不依赖 transformers。
================================================================================
对应《第五周 大模型语言初探》的知识点：
    · 01/02  Decoder-only + Causal Mask（下三角）→ "只能看自己和过去"
    · 03     Next-Token Prediction（预训练目标，100% token 参与 loss）
    · 04     logits → 采样（temperature / top-k / top-p，本文件提供算子，CLI 在 sample.py）
    · 05     现代 LLM 的标配组件：RMSNorm、RoPE、SwiGLU、权重共享、KV Cache

模型整体结构（与 Llama / Qwen 同源，只是层数极少）：

    token ids (B,T)
      → Embedding
      → N × Block( RMSNorm → Attention(RoPE, Causal) → 残差 ； RMSNorm → SwiGLU FFN → 残差 )
      → RMSNorm → Linear(V)  ← 与 Embedding 共享权重（weight tying）
      → logits (B,T,V)

参数命名里的 B=batch，T=序列长度，V=词表大小，D=d_model，H=头数，hd=每头维度。
================================================================================
"""
#region 0. 导入库   【模块:标准库 math/os/collections/dataclasses + torch】
import math
import os
import collections
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
#endregion

PAD, UNK, EOS = 0, 1, 2  #特殊 token 的固定 id   【常量:PAD=0,UNK=1,EOS=2】

#region 1. 分词器：字符级（char-level）   【类:CharTokenizer】
class CharTokenizer:
    #region 1.1 构造函数：把字符列表变成双向查表                   【函数:__init__(chars),return None】
    def __init__(self, chars):
        self.itos = ["<pad>", "<unk>", "<eos>"] + list(chars)  # 列表，id → 字符   char:真实字符的列表
        self.stoi = {c: i for i, c in enumerate(self.itos)}    # 字典，字符:id
		#stoi 是从 itos 反推的，不是独立构建。这样只要改 itos，stoi 自动跟着对——不会出现"两张表不一致"的 bug
    #endregion

    #region 1.2 类方法：从语料建词表                      【函数:from_text(文本或文档列表,最小词频,最大词表),return CharTokenizer 实例】
    @classmethod                                         #类方法/工厂方法，不需要新建类再调用，直接可以CharTokenizer.from_text(..)
    def from_text(cls, text_or_docs, min_freq=1, max_chars=None):
        if isinstance(text_or_docs, str):                # 情况一：一整段字符串，Counter 直接统计每个字符出现次数
            cnt = collections.Counter(text_or_docs)      # 字典，字符 → 出现次数
        else:                                            # 情况二：文档列表，逐篇累加（避免把 1 亿字拼成一个大字符串，省一次内存拷贝）
            cnt = collections.Counter()                  # 统计元素出现次数,创建空字典
            for d in text_or_docs:                       # 遍历文档
                cnt.update(d)                            # update增加计数，更新字典
        items = [(c, n) for c, n in cnt.items() if c not in "<>"]    # 列表，元素是 (字符, 次数)，去掉"<"和">"，避免特殊token污染
        # items =[('床', 2), ('前', 2), ('光', 2), ('明', 1)……]，元组解包：把 ('床', 2) 拆成 c='床', n=2
        if min_freq > 1:                                             # 过滤掉低频字（不过此处min_freq=1，没过滤）
            items = [it for it in items if it[1] >= min_freq]        # 长尾字的训练信号几乎没有，来自初始化，只会让采样时更容易抽到垃圾字符
        if max_chars and len(items) > max_chars:                     # 词表还是太大，就按词频从高到低截断，只留最常见的 max_chars 个
            items.sort(key=lambda x: -x[1])
            items = items[:max_chars]
        return cls(sorted(c for c, _ in items)) # 最后按字符编码排序，而不是按词频：这样词表与语料顺序无关，两次构建结果一致
    #endregion

    #region 1.3 编码：文本 → id 列表                 【函数:encode(字符串,是否补EOS),return id 列表】
    def encode(self, s, add_eos=False):
        """字符串 → id 列表。add_eos=True 时在末尾补一个停止符（训练文档要用）。"""
        ids = [self.stoi.get(c, UNK) for c in s]     # 列表，长度 = 字符数；查不到的字符用 UNK
        return ids + ([EOS] if add_eos else [])      # 训练时补 EOS，推理时通常不补
    #endregion

    #region 1.4 解码：id 列表 → 文本   【函数:decode(id 列表),return 字符串】
    def decode(self, ids):
        """id 列表 → 字符串。遇到 EOS 就停，跳过 PAD/UNK。"""
        out = []              # 列表，装还原出来的字符
        for i in ids:
            if i == EOS:      # 遇到停止符就停，这才是模型"学会停下"的意义
                break
            if i in (PAD, UNK):
                continue      # 填充位和未知字符跳过，否则会打印出 <pad>
            out.append(self.itos[i])
        return "".join(out)
    #endregion

    #region 1.5 属性：词表大小         【函数:vocab_size,return int】
    @property
    def vocab_size(self):
        """词表大小 V，等于 embedding 的行数，也等于 lm_head 的输出维度。"""
        return len(self.itos)
    #endregion
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 2. 基础组件
# ══════════════════════════════════════════════════════════════════════════════
#region 2.1 RMSNorm：只按均方根归一化   【类:RMSNorm】
class RMSNorm(nn.Module):
    """RMSNorm：只按均方根归一化，不减均值、没有 bias，比 LayerNorm 便宜。
    现代 LLM（Llama/Qwen/GLM）几乎都用它。

    公式：y = x / sqrt(mean(x²) + eps) * g
    对比 LayerNorm：(x - mean) / std * g + b —— 少一次减均值、少一个 bias，
    实测在 LLM 里效果几乎不变，但省下大量逐元素运算。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # 防止除以 0，1e-6 是通行取值
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放 g，(D,)，初始化为全 1

    def forward(self, x):
        """
        输入 x: (..., D) → 输出同形状。
        注意这里没有 bias，也没有减均值，所以只有一行。
        """
        # x: (..., D)  →  x / rms(x) * g
        # rms 形状 (..., 1)：先平方 → 在最后一维求均值 → 加 eps → 开方再取倒数
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms) * self.weight  # 广播机制让 weight(D,) 自动对齐到最后一维
#endregion

#region 2.2 RoPE 旋转位置编码：建 cos/sin 表   【函数:build_rope_cache(每头维度,最大长度,基数),return (cos (T,hd), sin (T,hd))】
def build_rope_cache(head_dim, max_len, base=10000.0, device="cpu"):
    """预计算 RoPE 的 cos/sin 表。

    RoPE（旋转位置编码）= 把 q/k 向量按位置"旋转"一个角度：
    两个 token 的内积只和它们的相对距离有关 → 模型天然知道"谁离我多远"。
    它比可学习的绝对位置编码更能外推，是现在所有主流 LLM 的默认选择。

    为什么用预计算表：每步都现算 cos/sin 会拖慢训练，而表只和位置有关、
    和数据无关，训练前算好一次就行（本文件把它注册成 buffer，跟着模型走）。
    """
    # 每个维度对一个旋转频率：低频维度转得慢（管长距离），高频维度转得快（管局部）
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(max_len, dtype=torch.float32, device=device)  # (T,) 位置 0,1,2,...
    freqs = torch.outer(pos, inv_freq)              # (T, hd/2) 外积：第 i 个位置 × 第 j 个频率
    emb = torch.cat([freqs, freqs], dim=-1)         # (T, hd) 前后半段拼成一样，方便后面"转一半"
    return emb.cos(), emb.sin()  # 各 (T, hd)，查表时按位置切片
#endregion

#region 2.3 RoPE 辅助：把向量前半段和后半段交换并取负   【函数:rotate_half(张量),return 同形状张量】
def rotate_half(x):
    """把最后一维切成两半，交换位置并把前半段取负：[-x2, x1]。

    这一步是旋转的代数实现 —— 二维旋转矩阵 [[cos,-sin],[sin,cos]] 作用在 (x1,x2) 上，
    展开就是 x1*cos - x2*sin 和 x1*sin + x2*cos，正好等于下面 apply_rope 的写法。
    """
    x1, x2 = x.chunk(2, dim=-1)  # 各 (..., hd/2)
    return torch.cat([-x2, x1], dim=-1)  # (..., hd)
#endregion

#region 2.4 RoPE 主算子：x*cos + rotate_half(x)*sin   【函数:apply_rope(x,cos,sin),return 旋转后的 x】
def apply_rope(x, cos, sin):
    """x: (B,H,T,hd)；cos/sin: (T,hd) 或 (1,1,T,hd)

    广播规则：cos/sin 是 (1,1,T,hd)，会自动扩到 (B,H,T,hd)，
    也就是同一个位置上所有 batch、所有头共用一套角度。
    """
    return x * cos + rotate_half(x) * sin
#endregion

#region 2.5 多头自注意力 + Causal Mask   【类:CausalSelfAttention】
class CausalSelfAttention(nn.Module):
    """多头自注意力 + Causal Mask。

    课程核心公式：Attention(Q,K,V) = softmax(QKᵀ/√d + mask) · V
    mask 就在 softmax 之前加：不允许看的位置 score = -inf → 权重变 0。
    这里 mask 就是那个 L×L 的下三角矩阵，一行一个 token 的"可见范围"。

    为什么要除以 √d：q·k 是 hd 个数相乘再求和，维度越高数值越大，
    softmax 会被推到大梯度区（几乎变成 one-hot），除以 √hd 把方差拉回 1。
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0  # 必须能整除，否则"切头"切不整齐
        self.n_head = cfg.n_head  # H：头数，比如 6
        self.head_dim = cfg.d_model // cfg.n_head  # hd：每头维度，384/6=64
        # 一次线性层同时算出 q、k、v，比三个独立的 Linear 少两次 kernel 启动
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)  # (D → 3D)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)  # 多头拼回来后再混一次
        self.dropout = cfg.dropout
        # 手动实现 mask 时用的开关：True 时用 F.scaled_dot_product_attention（快），
        # False 时用显式 masked_fill(-inf)（慢，但能把 mask 打印出来讲原理）
        self.fast_attn = cfg.fast_attn

    def forward(self, x, cos, sin, kv_cache=None):
        """
        输入 x: (B,T,D)；cos/sin: (T,hd) 位置表；kv_cache: (past_k, past_v) 或 (None,None)
        返回：(输出 (B,T,D), 本层新的 (k, v))
        """
        B, T, D = x.shape  # B=batch，T=当前要处理的 token 数（训练时 512，decode 时 1）
        q, k, v = self.qkv(x).split(D, dim=-1)  # 三个 (B,T,D)，各占 1/3
        # (B,T,D) → (B,H,T,hd)
        # 这里 view 把 D 拆成 (H,hd)，transpose 把 H 提到 T 前面，后面才能按头并行算注意力
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 位置编码：decode 阶段新 token 的绝对位置 = 已有缓存长度
        # 比如已经生成 100 个 token，新来的这个就是第 100 号位置，不是第 0 号
        offset = 0 if kv_cache is None or kv_cache[0] is None else kv_cache[0].shape[2]
        cos_t = cos[offset: offset + T].view(1, 1, T, self.head_dim)  # (1,1,T,hd)
        sin_t = sin[offset: offset + T].view(1, 1, T, self.head_dim)  # (1,1,T,hd)
        q, k = apply_rope(q, cos_t, sin_t), apply_rope(k, cos_t, sin_t)  # v 不旋转，只有 q/k 带位置

        # KV Cache：把历史的 K/V 存下来，生成第 t 个 token 时不用重算前 t-1 个
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                # 沿时间维拼接：(B,H,T,hd) + (B,H,past,hd) → (B,H,past+T,hd)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        Tk = k.shape[2]  # k/v 的总长度 = 历史长度 + 本次长度
        if self.fast_attn:
            # PyTorch 内置核（内部就是 flash/mem-efficient attention + causal mask）
            if T == Tk:
                # 训练阶段：q 和 k 一样长，直接用官方的 is_causal 开关，最省事也最快
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
                )  # (B,H,T,hd)
            else:
                # KV Cache 的 decode 阶段：q 只有 1 个（新 token），k 有 Tk 个（全部历史），
                # 此时 is_causal 靠"右上对齐"会算错，必须自己给出 mask。
                rows = torch.arange(T, device=x.device).view(-1, 1) + (Tk - T)  # query 的绝对位置 (T,1)
                cols = torch.arange(Tk, device=x.device).view(1, -1)  # key 的绝对位置 (1,Tk)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=(cols <= rows),  # (T,Tk) 的布尔矩阵，True=允许看
                    dropout_p=self.dropout if self.training else 0.0
                )
        else:
            # —— 课程里手写的那三行：算相似度 → masked_fill → softmax ——
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)      # (B,H,T,Tk)
            # 第 i 行（query 位置 offset+i）只允许看 j <= offset+i
            rows = torch.arange(T, device=x.device).view(-1, 1) + (Tk - T)
            cols = torch.arange(Tk, device=x.device).view(1, -1)
            mask = (cols <= rows)                                            # True=可见
            # mask 形状类似（T=4, Tk=4）：
            #   [T T F F]
            #   [T T T F]
            #   [T T T T]  ← 每行比上一行多看见一个，这就是"下三角"
            scores = scores.masked_fill(~mask, float("-inf"))                # 屏蔽 → -inf
            w = torch.softmax(scores, dim=-1)                                # -inf 位置权重=0
            # 为什么用 -inf 而不是 0：softmax 里 exp(0)=1 会分走权重，
            # 只有 exp(-inf)=0 才是真正的"完全看不见"
            w = F.dropout(w, self.dropout, self.training)
            y = w @ v  # (B,H,T,Tk) @ (B,H,Tk,hd) → (B,H,T,hd)，Tk 这一维被消掉

        # (B,H,T,hd) → (B,T,H,hd) → (B,T,D)：把头拼回通道维，才能接后面的 Linear
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y), (k, v)  # 返回新的 k/v，供下一步复用（这就是 KV Cache）
#endregion

#region 2.6 SwiGLU 前馈层   【类:SwiGLU】
class SwiGLU(nn.Module):
    """SwiGLU 前馈层：FFN(x) = W2( silu(W1x) ⊙ W3x )
    比 ReLU/GELU 的两层 MLP 多一个门控分支，同样算力下效果更好（Llama 起成为标配）。

    结构：D → hidden（两条支路并行）→ 门控相乘 → hidden → D
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg.d_model * cfg.ffn_mult * 2 / 3)   # 参数量与 4x 的普通 MLP 对齐
        # 为什么乘 2/3：SwiGLU 有 w1/w3 两条输入支路，参数比普通 MLP 多 50%，
        # 把中间维压到 4D 的 2/3，总参数量才和经典 FFN 的 4D→D 持平
        hidden = (hidden + 63) // 64 * 64                   # 取 64 的倍数
        # 取 64 的倍数是为了让矩阵乘落在对齐的显存块上，GPU 的 tensor core 更喜欢这种尺寸
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)   # 门控分支
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)   # 内容分支
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)   # 降回原维度
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """(B,T,D) → (B,T,D)，中间在 hidden 维上膨胀再压回。"""
        # silu(w1(x)) 当"闸门"（0~1 之间连续取值），乘到 w3(x) 上控制信息流量
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))
#endregion

#region 2.7 残差块（Pre-Norm）   【类:Block】
class Block(nn.Module):
    """Pre-Norm 残差块：x = x + Attn(Norm(x))；x = x + FFN(Norm(x))
    Pre-Norm（先归一化再进子层）比原始 GPT 的 Post-Norm 更容易训练深层网络。

    结构：Norm → Attention → 加回 x → Norm → FFN → 再加回 x
    """

    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)  # 进注意力前的归一化
        self.attn = CausalSelfAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)  # 进 FFN 前的归一化
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        """(B,T,D) 进，(B,T,D) 出；同时把本层的 k/v 缓存传出去。"""
        a, new_cache = self.attn(self.n1(x), cos, sin, kv_cache)  # 先 Norm 再进注意力（Pre-Norm）
        # 残差连接：梯度可以沿这条"高速公路"直接回流到浅层，深层网络才训得动
        x = x + a
        x = x + self.ffn(self.n2(x))  # FFN 同样走残差
        return x, new_cache
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 3. 模型主体
# ══════════════════════════════════════════════════════════════════════════════
#region 3.1 超参配置   【类:GPTConfig，字段即模型规模】
@dataclass
class GPTConfig:
    """模型超参。改这里就等于换模型规模，train / sample / chat 都从这个类读配置。"""
    vocab_size: int = 2579  # V：词表大小（真实训练时会被语料覆盖，如 5153）
    block_size: int = 256      # 上下文长度（课程里的 T）
    d_model: int = 384         # 隐层维度
    n_head: int = 6  # H：注意力头数，必须能整除 d_model
    n_layer: int = 8  # N：堆多少个 Block，对参数量影响最大
    ffn_mult: float = 4.0  # FFN 中间维相对 d_model 的倍数
    dropout: float = 0.1  # 训练时的随机丢弃比例，防过拟合
    fast_attn: bool = True     # True=用 PyTorch 融合核；False=手写 masked_fill 版本
#endregion

#region 3.2 GPT 主体   【类:GPT】
class GPT(nn.Module):
    """Decoder-only Transformer。

    结构：Embedding → N × Block → RMSNorm → Linear(V) → logits
    参数规模近似 N ∝ n_layer · d_model²（attention 占大头，FFN 再补一部分）。
    """

    #region 3.2.1 构造函数：搭骨架 + 初始化   【函数:__init__(cfg),return None】
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg  # 存一份，generate / save_ckpt 都要读
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)  # 查表：(B,T) → (B,T,D)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])  # N 个残差块
        self.norm_f = RMSNorm(cfg.d_model)  # 最后一层归一化，进 lm_head 之前
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # (B,T,D) → (B,T,V)
        self.apply(self._init)  # 递归给所有子模块做初始化
        self.lm_head.weight = self.embed.weight      # weight tying：省参数 + 表征更一致（放在 init 之后，保证共享的是同一块 tensor）
        # 为什么能共享：输入是"字符→向量"，输出是"向量→字符"，本来就是同一张表的两面；
        # 共享后参数量少掉 V*D 个（这里约 5153*384 ≈ 198 万），而且两边梯度会一起更新
        # 残差分支按层数缩放初始化（GPT-2 的做法），深层网络更稳
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                # 这两层在残差主路上，输出会被反复累加，标准差要按 1/√(2N) 缩小，
                # 否则 8 层累加下来激活值会指数级放大
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # RoPE 表注册成 buffer：跟着模型 .to(device) 走，但不算参数、不进优化器
        # persistent=False 表示不写进 state_dict，加载旧权重时不会因为形状不同报错
        self.register_buffer("rope_cos", torch.zeros(cfg.block_size, cfg.d_model // cfg.n_head),
                             persistent=False)
        self.register_buffer("rope_sin", torch.zeros(cfg.block_size, cfg.d_model // cfg.n_head),
                             persistent=False)
        self._refresh_rope(cfg.block_size)  # 用真实的角度值填满上面两张表
    #endregion

    #region 3.2.2 权重初始化   【函数:_init(模块),return None】
    def _init(self, m):
        """给 Linear 和 Embedding 做正态初始化，bias 清零。"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)  # std=0.02 是 GPT-2 的经验值
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 偏置初始为 0，让初始输出不带额外偏移
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    #endregion

    #region 3.2.3 重建 RoPE 表   【函数:_refresh_rope(长度),return None】
    def _refresh_rope(self, length):
        """按需要的长度（重新）生成 cos/sin 表。生成长文本超长时会自动扩展。"""
        cos, sin = build_rope_cache(self.cfg.d_model // self.cfg.n_head, length,
                                    device=self.rope_cos.device)  # 各 (length, hd)
        self.rope_cos, self.rope_sin = cos, sin
    #endregion

    #region 3.2.4 统计参数量   【函数:num_params(),return int】
    def num_params(self):
        """返回可训练参数总量（因为权重共享，V*D 那份只算一次）。"""
        return sum(p.numel() for p in self.parameters())
    #endregion

    #region 3.2.5 前向：ids → logits（可选 loss）   【函数:forward(idx(B,T),targets,缓存,是否返回缓存),return (logits (B,T,V), loss, 新缓存)】
    def forward(self, idx, targets=None, kv_caches=None, use_cache=False):
        """idx:(B,T) → logits:(B,T,V)；给了 targets 就顺便算 loss。
        注意：这就是课程里"训练信号密度 100%"的预训练 loss —— 每个位置都要预测下一个 token。
        """
        B, T = idx.shape  # B=batch，T=本次输入的 token 数
        x = self.drop(self.embed(idx))  # (B,T) → (B,T,D)，查表得到词向量
        # 生成时新 token 的绝对位置 = 已有的缓存长度，所以 RoPE 表要按 (缓存长度 + T) 检查
        offset = 0 if not kv_caches else (kv_caches[0][0].shape[2] if kv_caches[0][0] is not None else 0)
        if offset + T > self.rope_cos.shape[0]:      # 序列超长就动态扩展 RoPE 表
            self._refresh_rope(offset + T)
        cos, sin = self.rope_cos, self.rope_sin      # 传完整表，由 attention 按 offset 切片

        new_caches = [] if use_cache else None  # 列表，长度 = n_layer，每项是 (k,v)
        for i, blk in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None  # 本层上次留下的 k/v
            x, nc = blk(x, cos, sin, cache)  # (B,T,D) → (B,T,D)，形状不变
            if use_cache:
                new_caches.append(nc)
                # 为什么形状不变：残差结构保证每个 Block 的输入输出维度一致，
                # 这也是能无脑堆 8 层、80 层的原因

        logits = self.lm_head(self.norm_f(x))  # (B,T,D) → (B,T,V)，每个位置一个词表分布
        loss = None
        if targets is not None:
            # 交叉熵：把 (B,T,V) 拉平成 (B*T, V)，标签 (B*T,)
            # 注意这里 reshape 没有做 shift —— 位移是在 train.py 造数据时完成的（x=前 T 个，y=后移一位）
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_caches
    #endregion

    # ──────────────────────────────────────────────────────────────────────
    # 4. 采样（课程 04 节的全部策略都落在这里）
    # ──────────────────────────────────────────────────────────────────────
    #region 4.1 把 logits 洗成概率分布   【z函数:filter_logits(logits,温度,K,P,重复惩罚,历史id,封禁id),return probs (B,V)】
    @staticmethod
    def filter_logits(logits, temperature=1.0, top_k=None, top_p=None, rep_penalty=1.0, prev_ids=None,
                      ban_ids=None):
        """把原始 logits 变成"清洗过的概率分布"。顺序很重要：先温度，再惩罚，再 top-k，再 top-p。"""
        logits = logits.float()  # 统一转 fp32，避免 bf16 下 softmax 数值精度不够
        # ① Temperature：p(i)=softmax(logits/T)。T<1 分布更尖锐→更确定；T>1 更均匀→更发散
        if temperature is not None and temperature != 1.0:
            logits = logits / max(temperature, 1e-5)  # 除以 0 会炸，兜个底

        # ①.5 封禁某些 token（聊天界面用：压掉 EOS，否则模型十几个字就收尾）
        if ban_ids:
            logits = logits.clone()  # 先复制，别把调用方的张量改了
            logits[..., list(ban_ids)] = float("-inf")  # exp(-inf)=0，概率直接归零

        # ② 重复惩罚：把已经出现过的 token 的分数压低（教学用，Llama 系常用）
        if rep_penalty and rep_penalty != 1.0 and prev_ids is not None:
            # 先去重：同一个字出现 10 次也只惩罚一次，否则惩罚会累积到爆炸
            uniq = sorted(set(prev_ids.tolist() if torch.is_tensor(prev_ids) else list(prev_ids)))
            sel = logits[..., uniq]  # (B, len(uniq)) 只取出这些位置
            # 正分数除以 rp（压低），负分数乘以 rp（继续压低）—— 两个方向都是"更不可能"
            logits[..., uniq] = torch.where(sel > 0, sel / rep_penalty, sel * rep_penalty)

        # ③ Top-K：只保留分数最高的 K 个，其余置 -inf（softmax 后≈0）
        if top_k:
            k = min(top_k, logits.size(-1))  # K 不能超过词表大小
            kth = torch.topk(logits, k).values[..., -1, None]  # 第 K 大的那个值当阈值，(B,1)
            logits = logits.masked_fill(logits < kth, float("-inf"))  # 比阈值小的一律砍掉

        probs = F.softmax(logits, dim=-1)  # (B,V)，每行和为 1

        # ④ Top-P（核采样）：按概率降序累加，保留"累积概率刚达到 P"的最小集合
        if top_p is not None and 0 < top_p < 1.0:
            sorted_p, sorted_idx = probs.sort(descending=True)  # 概率从大到小排 (B,V)
            cum = sorted_p.cumsum(-1)                 # 包含自己在内的累积概率
            prev = cum - sorted_p                     # 不含自己的累积概率（即"前面那些"）
            # prev < P 才保留：等价于课程里 cum <= P 的写法，但不会"多留一个"，
            # 且排名第 1 的 token（prev=0）永远保留 —— 天然带兜底，P 再小也至少留 1 个
            mask = prev < top_p  # (B,V) 布尔矩阵，True=留在核里
            sorted_p = sorted_p * mask  # 核外的概率清零
            # scatter 把排序后的概率放回原来的词表位置：scatter(维度, 索引, 值)
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_p)
            probs = probs / probs.sum(-1, keepdim=True)   # 重新归一化，保证和为 1
            # 为什么要重新归一化：砍掉核外的 token 后，剩下的概率和 < 1，
            # 不归一化 multinomial 会报错或行为异常
        return probs
    #endregion

    #region 4.2 自回归生成   【函数:generate(idx(B,T0),最多生成数,温度,K,P,是否贪心,重复惩罚,用缓存,遇EOS停,回调,封禁id),return 生成出的 id 列表】
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.9, top_k=50, top_p=0.9,
                 greedy=False, rep_penalty=1.0, use_cache=True, stop_at_eos=True,
                 on_token=None, ban_ids=None):
        """自回归生成：每一步把新 token 拼回输入再跑一次前向（课程 02 节的 t1/t2/t3）。

        两处工程加速：
          · use_cache=True → KV Cache，第 t 步只算 1 个 token，而不是重算整段
          · on_token 回调 → 每生成一个 token 就吐给调用方，这就是"流式输出"

        @torch.no_grad()：推理不需要梯度，关掉后显存占用大幅下降（不用存中间激活）。
        """
        self.eval()  # 关掉 dropout，否则生成结果会随机抖动
        caches = None
        if use_cache:
            caches = [(None, None)] * len(self.blocks)  # 每层一个空槽，第一轮没有历史
            cur = idx  # 有缓存时 cur 只放"最新的 token"
        else:
            cur = idx  # 无缓存时 cur 会不断变长，每步重算整段
        out_ids = []  # 列表，装生成出来的 id

        for _ in range(max_new_tokens):
            if use_cache:
                logits, _, caches = self.forward(cur, kv_caches=caches, use_cache=True)
            else:
                # 只取最后 block_size 个：超过上下文长度就必须截断，否则位置编码越界
                logits, _, _ = self.forward(cur[:, -self.cfg.block_size:])
            logits = logits[:, -1, :]                    # 只关心最后一个位置的分布 (B,V)
            if greedy:
                # 贪心：永远选概率最大的那个。确定性强，但容易陷入复读循环
                nxt = logits.argmax(-1, keepdim=True)  # (B,1)
            else:
                # 重复惩罚只支持 batch=1；多 batch 时给 None，等于不惩罚
                prev = torch.tensor(out_ids) if (out_ids and logits.size(0) == 1) else None
                probs = self.filter_logits(logits, temperature, top_k, top_p, rep_penalty, prev, ban_ids)
                nxt = torch.multinomial(probs, num_samples=1)  # 按概率抽一个，(B,1)
            if use_cache:
                cur = nxt                                # 有缓存时只喂新 token
            else:
                cur = torch.cat([cur, nxt], dim=1)  # (B,T) + (B,1) → (B,T+1)
            if stop_at_eos and nxt.size(0) == 1 and int(nxt.item()) == EOS:
                break                                    # 课程："在每个句子后面放停止符"
            # batch=1 时直接转 int，多 batch 时保留成 list，方便调用方处理
            out_ids.append(int(nxt[0].item()) if nxt.size(0) == 1 else nxt.squeeze(1).tolist())
            if on_token:
                on_token(out_ids[-1])  # 流式输出：拿到一个就推给前端，不等全部生成完
        return out_ids
    #endregion
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 5. 存取
# ══════════════════════════════════════════════════════════════════════════════
#region 5.1 保存权重：模型 + 配置 + 词表一起存   【函数:save_ckpt(路径,模型,分词器,附加信息),return None】
def save_ckpt(path, model, tokenizer, extra=None):
    """存成一个 dict：权重、超参、词表、附加信息（比如 PPL / token 数）。

    为什么不只存 state_dict：权重文件必须能"自解释"，否则下次加载时
    不知道该建多宽的模型、词表是什么顺序，等于白存。
    """
    torch.save({
        "model": model.state_dict(),  # 权重字典
        "config": vars(model.cfg),    # 超参转 dict，加载时用来重建同构模型
        "itos": tokenizer.itos,       # 词表，必须和权重同源，否则 id 全对不上
        "extra": extra or {},         # 训练指标，纯记录用
    }, path)
#endregion

#region 5.2 加载权重：重建模型 + 分词器   【函数:load_ckpt(路径,设备),return (模型, 分词器, 附加信息)】
def load_ckpt(path, device="cpu"):
    """从 ckpt 恢复出 (model, tokenizer, extra)，三件套齐全才能直接推理。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)  # weights_only=False 才允许读 dict
    cfg = GPTConfig(**ckpt["config"])  # 用存下来的超参重建配置
    model = GPT(cfg)  # 先按配置搭好骨架
    model.load_state_dict(ckpt["model"])  # 再把权重灌进去
    model.to(device).eval()  # 上设备并切推理模式
    # 分词器不走 __init__（它要的是字符列表），直接造对象再塞两个字段，省一次词表重扫
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.itos = ckpt["itos"]
    tok.stoi = {c: i for i, c in enumerate(tok.itos)}  # 反查表由 itos 现推
    return model, tok, ckpt.get("extra", {})
#endregion

#region 5.3 规模预设：Scaling Law 实验用   【函数:scale_config(预设名,词表,上下文),return GPTConfig】
def scale_config(preset, vocab_size, block_size=None):
    """三档预设，方便自己复现 Scaling Law 实验（同样的语料、同样的 token 预算，只改模型大小）。
    参数量的关系近似 N ∝ n_layer · d_model²，这就是"越大越强"的幂律来源。
    """
    table = {
        #        n_layer d_model n_head
        "tiny":  (4,       192,    4),   # 约 2.5M 参数，几分钟就能跑完
        "mini":  (5,       240,    6),
        "small": (6,       288,    6),
        "base":  (8,       384,    6),   # 本项目主力档，16.1M 参数
        "large": (10,      480,    8),
    }
    nl, dm, nh = table[preset]
    # 注意：d_model 必须能被 n_head 整除，上表已按这个规则配好
    return GPTConfig(vocab_size=vocab_size, block_size=block_size or 256,
                     d_model=dm, n_head=nh, n_layer=nl)
#endregion


#region 6. 自检入口：打印参数量 + 可视化 Causal Mask
if __name__ == "__main__":
    # 自检：打印参数量 + 把 Causal Mask 可视化成文字（课程里那张下三角图）
    cfg = scale_config("base", 2579)  # 这里写死 2579 是为了自检快；真实训练的词表是 5153
    m = GPT(cfg)
    print(f"参数量: {m.num_params():,}")  # 打印参数量（逗号分位）
    x = torch.randint(0, 100, (2, 8))  # (B=2, T=8) 随便造点 id，只关心形状
    logits, loss, _ = m(x, x)  # targets 直接传 x，测一下前向能跑通
    print("logits", tuple(logits.shape), "loss", float(loss.detach()))  # 期望 (2,8,2579)，loss 约 ln(2579)≈7.85
    print("\nCausal Mask（1=可见，0=屏蔽），第 i 行=第 i 个 token 的可见范围：")
    mask = torch.tril(torch.ones(8, 8))  # 下三角：tril 把右上角全置 0，就是"不能看未来"
    for r in mask.int().tolist():
        print("  ", " ".join("■" if v else "·" for v in r))  # 打印成文字版的下三角矩阵
#endregion

# 引用pytorch库
import torch
import math
import torch.nn as nn
#引入config.yaml
# from omegaconf import OmegaConf

class TransformerBlock(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # 加载config.yaml文件
        # self.cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml", resolve=True).transformer
        self.cfg = cfg.transformer
        # 初始化transformer所需qkv output fnn 残差连接层
        self.qkv = torch.nn.Linear(self.cfg.model_size, 3 * self.cfg.model_size)
        self.out = torch.nn.Linear(self.cfg.model_size, self.cfg.model_size)
        self.norm1 = nn.LayerNorm(self.cfg.model_size)
        self.norm2 = nn.LayerNorm(self.cfg.model_size)
        self.attn_drop = nn.Dropout(self.cfg.dropout)
        self.dropout1 = nn.Dropout(self.cfg.dropout)
        self.dropout2 = nn.Dropout(self.cfg.dropout)
        self.fc1 = nn.Linear(self.cfg.model_size, self.cfg.d_ff)
        self.fc2 = nn.Linear(self.cfg.d_ff, self.cfg.model_size)
        self.act = nn.ReLU() if self.cfg.act == "relu" else nn.GELU()
    
    def forward(self, x):
        # 经过MHA + 残差1
        if self.cfg.ln: # 是否加 LN 层
            x = self.norm1(x+self.dropout1(self.MHA(x)))
        else:
            x = x + self.dropout1(self.MHA(x))       # 残差1，不加 LN
        # 经过ffn + 残差2
        if self.cfg.ln: # 是否加 LN 层
            x = self.norm2(x+self.dropout2(self.FFN(x)))
        else:
            x = x + self.dropout2(self.FFN(x))       # 残差2，不加 LN
        return x

    def FFN(self,x):
        return self.fc2(self.act(self.fc1(x)))

    def MHA(self,x):
        B, T, C = x.shape # 批次大小，文本长度，文本对应向量的维度 如，64 * 128 * 128
        # 计算qkv
        qkv = self.qkv(x)
        # 计算多头注意力
        q, k, v = qkv.chunk(3, dim=-1)
        def split(t):
            return t.view(B, T, self.cfg.num_heads, C // self.cfg.num_heads).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        # q * kT / 缩放
        attn = q @ k.transpose(-2, -1) / math.sqrt(C//self.cfg.num_heads)
        if self.cfg.causal:
            # mask 上三角
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(attn.device)
            attn = attn.masked_fill(causal_mask, float("-inf"))
        # softmax + dropout
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        # 在乘v并合并头
        attn = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn)
    
    
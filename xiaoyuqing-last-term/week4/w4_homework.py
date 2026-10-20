import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  #每个注意力头的维度768/12=64

        # 分别把输入映射成 Query、Key、Value
        self.q_proj = nn.Linear(d_model, d_model)  #输入128*768 经过768*768的线性层映射成128*768
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)

        # 将多头注意力的结果映射回 d_model 维
        self.out_proj = nn.Linear(d_model, d_model) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状：[batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # 投影得到 Q、K、V，形状都是 [batch_size, seq_len, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 将最后一维拆成“注意力头数”和“每个头的维度”
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算缩放点积注意力分数：
        # Q 和 K 做点积，除以 sqrt(head_dim) 控制数值大小
        # scores 形状：[B, H, T, T]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 对每个位置的注意力分数做 Softmax，得到注意力权重
        # weights 形状：[B, H, T, T]
        weights = torch.softmax(scores, dim=-1)

        # 用注意力权重对 V 加权求和
        # context 形状：[B, H, T, D]
        context = weights @ v

        # 合并多个注意力头：
        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, d_model)

        # 输出映射，得到 MHA(x)
        return self.out_proj(context)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ffn: int,
    ):
        super().__init__()

        self.mha = MultiHeadSelfAttention(d_model, num_heads)

        # FFN：先扩展维度，再经过 GELU，最后映射回 d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, d_model),
        )

        # LayerNorm 对最后一个维度 d_model 做归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步：z = LayerNorm(x + MHA(x))
        # 加上 x 是残差连接
        z = self.norm1(x + self.mha(x))

        # 第二步：output = LayerNorm(z + FFN(z))
        # 同样使用残差连接
        output = self.norm2(z + self.ffn(z))

        return output


# 示例：输入形状为 [batch_size, seq_len, d_model]
x = torch.randn(2, 10, 64)

layer = TransformerLayer(
    d_model=64,
    num_heads=4,
    dim_ffn=128,
)

output = layer(x)
print(output.shape)  # torch.Size([2, 10, 64])
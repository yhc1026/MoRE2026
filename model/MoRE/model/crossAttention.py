import torch
from torch import nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制 - 从 MO-Hate 借鉴
    让一个模态在参考另一个模态时，能保留自己的上下文
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # 投影矩阵
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # ===== 新增：两个 dropout =====
        self.attn_dropout = nn.Dropout(dropout)  # 注意力权重上的 dropout
        self.out_dropout = nn.Dropout(dropout)  # 输出上的 dropout
        # ============================

        # 门控机制（MO-Hate的特色）
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value, return_weights=False):
        """
        query: 当前模态 (B, Lq, D)
        key, value: 要参考的模态 (B, Lk, D)
        """
        B, Lq, _ = query.shape
        B, Lk, _ = key.shape

        # 分头计算
        Q = self.q_proj(query).reshape(B, Lq, self.num_heads, -1).permute(0, 2, 1, 3)
        K = self.k_proj(key).reshape(B, Lk, self.num_heads, -1).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, Lk, self.num_heads, -1).permute(0, 2, 1, 3)

        # 注意力分数
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)  # 🔥 使用 attention dropout

        # 加权求和
        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, Lq, -1)
        out = self.out_proj(out)
        out = self.out_dropout(out)  # 🔥 使用输出 dropout

        # 门控融合（保留原始query的信息）
        gate = self.gate(torch.cat([out, query], dim=-1))
        out = gate * out + (1 - gate) * query

        if return_weights:
            return out, attn
        return out
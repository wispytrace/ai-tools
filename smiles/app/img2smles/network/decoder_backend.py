# # models/decoder.py
# import torch
# import torch.nn as nn
# import math


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=768):
#         super().__init__()
#         self.max_len = max_len
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         x: [B, T, D]
#         """
#         T = x.size(1)
#         assert T <= self.max_len, f"Length {T} > {self.max_len}"
#         pe = self.pe[:T].unsqueeze(0)  # [1, T, D]
#         return x + pe


# class RotaryPositionalEmbedding(nn.Module):
#     """
#     修正版本 RoPE：
#     - 保证每对 (even, odd) 做 2D 旋转，并回写到原交错维度
#     - 避免你原实现中的“通道错位拼接”导致的不正交混合与数值放大
#     """
#     def __init__(self, dim, max_seq_len=768):
#         super().__init__()
#         assert dim % 2 == 0, "RoPE requires dim to be even."
#         self.dim = dim

#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # [D/2]
#         t = torch.arange(max_seq_len).float()  # [T]
#         freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, D/2]

#         # 只存 D/2 的 cos/sin，计算时按 (even, odd) 配对旋转
#         self.register_buffer("cos", freqs.cos()[None, :, :], persistent=False)  # [1, T, D/2]
#         self.register_buffer("sin", freqs.sin()[None, :, :], persistent=False)  # [1, T, D/2]

#     def forward(self, x):
#         """
#         x: [B, T, D]
#         returns: [B, T, D]
#         """
#         B, T, D = x.shape
#         cos = self.cos[:, :T, :]  # [1, T, D/2]
#         sin = self.sin[:, :T, :]  # [1, T, D/2]

#         # [B, T, D/2, 2]，最后一维为 (even, odd)
#         x_ = x.view(B, T, D // 2, 2)
#         x_even = x_[..., 0]
#         x_odd = x_[..., 1]

#         out_even = x_even * cos - x_odd * sin
#         out_odd = x_even * sin + x_odd * cos

#         out = torch.stack([out_even, out_odd], dim=-1)  # [B, T, D/2, 2]
#         return out.view(B, T, D)


# class GatedTransformerDecoderLayer(nn.TransformerDecoderLayer):
#     """
#     稳定性改进：
#     1) 使用 Pre-LN（norm_first=True）以提升深层训练稳定性
#     2) 修正门控残差：只门控 cross-attn 输出，不在 residual 里再次混入 tgt（避免增益>1导致爆炸）
#     3) 保持类名/函数名不变，外界调用不受影响
#     """
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
#         # 关键：启用 Pre-LN
#         super().__init__(
#             d_model, nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=batch_first,
#             norm_first=True
#         )
#         # 标量门控：输出 [B, T, 1]
#         self.gate_proj = nn.Linear(d_model, 1)
#         self._reset_gate_bias()

#     def _reset_gate_bias(self):
#         # 初始化为负值，初期让 gate 更小 => cross-attn 注入更保守（更稳）
#         nn.init.constant_(self.gate_proj.bias, -2.0)
#         nn.init.zeros_(self.gate_proj.weight)

#     def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
#                 tgt_key_padding_mask=None, memory_key_padding_mask=None,
#                 **kwargs):
#         """
#         说明：
#         - 仍然兼容外部传参（**kwargs）
#         - 使用 Pre-LN 流程：
#           tgt = tgt + SA(LN(tgt))
#           tgt = tgt + gate * CA(LN(tgt), memory)
#           tgt = tgt + FFN(LN(tgt))
#         """
#         # ---- Self-Attention (Pre-LN) ----
#         x = tgt
#         x_norm = self.norm1(x)
#         sa_out = self.self_attn(
#             x_norm, x_norm, x_norm,
#             attn_mask=tgt_mask,
#             key_padding_mask=tgt_key_padding_mask,
#             need_weights=False
#         )[0]
#         x = x + self.dropout1(sa_out)

#         # ---- Cross-Attention (Pre-LN) ----
#         x_norm = self.norm2(x)
#         ca_out = self.multihead_attn(
#             x_norm, memory, memory,
#             attn_mask=memory_mask,
#             key_padding_mask=memory_key_padding_mask,
#             need_weights=False
#         )[0]

#         # 门控：只控制 cross-attn 注入量（避免把 x 再混一次导致残差增益偏大）
#         gate = torch.sigmoid(self.gate_proj(x_norm))  # [B, T, 1]
#         x = x + self.dropout2(gate * ca_out)

#         # ---- FFN (Pre-LN) ----
#         x_norm = self.norm3(x)
#         ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
#         x = x + self.dropout3(ffn_out)

#         return x


# class TransformerDecoder(nn.Module):
#     def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=7):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoder = RotaryPositionalEmbedding(d_model)

#         # 使用自定义的门控层（内部已更稳定）
#         layer = GatedTransformerDecoderLayer(
#             d_model, nhead, dim_feedforward=2048, batch_first=True
#         )
#         self.transformer = nn.TransformerDecoder(layer, num_layers)
#         self.out = nn.Linear(d_model, vocab_size)
#         self.d_model = d_model

#         # 初始化 embedding（保持你的习惯）
#         self.embedding.weight.data.normal_(mean=0.0, std=0.02)

#     def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
#         """
#         tgt: [B, T]
#         memory: [B, S, D]
#         """
#         tgt_emb = self.embedding(tgt)
#         tgt_emb = self.pos_encoder(tgt_emb)

#         output = self.transformer(
#             tgt_emb, memory,
#             tgt_mask=tgt_mask,
#             tgt_key_padding_mask=tgt_key_padding_mask
#         )
#         return self.out(output)
    

import torch
import torch.nn as nn

# ==========================================
# 1. 核心组件 (RoPE, Gated Layer) - 保持逻辑，更新默认参
# ==========================================

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048): # 建议把max_len调大一点，防万一
        super().__init__()
        assert dim % 2 == 0, "RoPE requires dim to be even."
        self.dim = dim
        
        # 预计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        self.register_buffer("cos", freqs.cos()[None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, :], persistent=False)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        # 防止序列超长
        if T > self.cos.shape[1]:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.cos.shape[1]}")
            
        cos = self.cos[:, :T, :]
        sin = self.sin[:, :T, :]

        x_ = x.view(B, T, D // 2, 2)
        x_even = x_[..., 0]
        x_odd = x_[..., 1]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.stack([out_even, out_odd], dim=-1)
        return out.view(B, T, D)


class GatedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    配置已升级为 Large 规格默认值:
    d_model=1024, nhead=16, dim_feedforward=4096
    """
    def __init__(self, d_model=1024, nhead=16, dim_feedforward=4096, dropout=0.1, batch_first=True):
        super().__init__(
            d_model, nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=True # Pre-LN 训练更稳定
        )
        self.gate_proj = nn.Linear(d_model, 1)
        self._reset_gate_bias()

    def _reset_gate_bias(self):
        # 初始化为负值，让 gate 初始较小，保护训练稳定性
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.zeros_(self.gate_proj.weight)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                **kwargs):
        
        # 1. Self-Attention
        x = tgt
        x_norm = self.norm1(x)
        sa_out = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )[0]
        x = x + self.dropout1(sa_out)

        # 2. Cross-Attention
        x_norm = self.norm2(x)
        ca_out = self.multihead_attn(
            x_norm, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )[0]

        # Gating Mechanism
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x = x + self.dropout2(gate * ca_out)

        # 3. FFN
        x_norm = self.norm3(x)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout3(ffn_out)

        return x


# ==========================================
# 2. Decoder 主体 (升级为 Large 配置)
# ==========================================

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=1024, nhead=16, num_layers=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = RotaryPositionalEmbedding(d_model)

        # Decoder Layers
        # 显式指定 dim_feedforward 为 4倍 d_model (4096)
        layer = GatedTransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers)
        
        # Output Head
        self.out = nn.Linear(d_model, vocab_size)
        
        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: [B, T_dec]
        memory: [B, T_enc, d_model]  <-- 注意：这里的 memory 必须已经是 1024 维
        """
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb) # Apply RoPE

        output = self.transformer(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.out(output)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 核心组件: 严谨的 RoPE 实现 (LLaMA Style)
# ==========================================

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 计算频率 theta_i
        # dim 必须是 head_dim，而不是 d_model
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 初始化缓存
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 外积: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # 拼接: [Left, Right] 共享频率 -> [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 缓存 [1, 1, seq_len, dim] 以便广播
        self.register_buffer("cos_cached", emb.cos().to(dtype).unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype).unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [B, H, T, D]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len + 128, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x):
    """[-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, T, D]
    # cos, sin: [1, 1, T, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==========================================
# 2. 自定义 Attention 层
# ==========================================

class RotarySelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, max_len=2048):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        assert self.head_dim % 2 == 0, "Head dim must be even for RoPE"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 🔥 RoPE 初始化: 使用 head_dim
        self.rope = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=max_len)
        self.dropout = dropout

    def forward(self, x, is_causal=True):
        B, T, C = x.shape
        
        # 1. 投影 + 分头: [B, T, C] -> [B, H, T, head_dim]
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 2. 应用 RoPE
        cos, sin = self.rope(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Scaled Dot Product Attention (FlashAttention compatible)
        # is_causal=True 会自动处理对角掩码
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=is_causal
        )

        # 4. 合并头
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ==========================================
# 3. Decoder Layer (Pre-Norm + Gated)
# ==========================================

class GatedDecoderLayer(nn.Module):
    def __init__(self, d_model=1024, nhead=16, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        
        # --- Self Attention Block (RoPE) ---
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RotarySelfAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        # --- Cross Attention Block (Standard) ---
        self.norm2 = nn.LayerNorm(d_model)
        # Cross Attn 不需要 RoPE，因为 Key/Value 来自图像特征
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # 门控机制
        self.gate_proj = nn.Linear(d_model, 1)
        self._reset_gate_bias()

        # --- FFN Block ---
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU() # 或 nn.SiLU()
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)

    def _reset_gate_bias(self):
        # 初始化为 -2.0，使 gate 初始值较小 (sigmoid(-2) ≈ 0.12)
        # 帮助模型在初期先关注自身语法，再慢慢引入图像信息
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.zeros_(self.gate_proj.weight)

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        # tgt: [B, T, D]
        # memory: [B, L, D]
        
        # 1. Pre-Norm Self Attention (RoPE inside)
        x = tgt
        x_norm = self.norm1(x)
        # is_causal=True 处理了 autoregressive mask
        sa_out = self.self_attn(x_norm, is_causal=True)
        x = x + self.dropout1(sa_out)

        # 2. Pre-Norm Gated Cross Attention
        x_norm = self.norm2(x)
        ca_out, _ = self.cross_attn(
            query=x_norm, 
            key=memory, 
            value=memory, 
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        
        # 应用 Gate
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x = x + self.dropout2(gate * ca_out)

        # 3. Pre-Norm FFN
        x_norm = self.norm3(x)
        ffn_out = self.linear2(self.dropout_ffn(self.activation(self.linear1(x_norm))))
        x = x + self.dropout3(ffn_out)

        return x


# ==========================================
# 4. 主 Decoder 类
# ==========================================

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=1024, nhead=16, num_layers=16, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        
        # Word Embedding (无位置编码！位置编码在 Attention 层动态添加)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            GatedDecoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Final Norm (Pre-Norm 架构必须)
        self.norm = nn.LayerNorm(d_model)
        
        # Output Projection
        self.out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: [B, T] (Token IDs)
        memory: [B, L, D] (Image Features)
        """
        # 1. Embedding
        x = self.embedding(tgt) # [B, T, D]
        
        # 2. Layers
        for layer in self.layers:
            x = layer(x, memory, memory_key_padding_mask=memory_key_padding_mask)
            
        # 3. Final Norm & Logits
        x = self.norm(x)
        return self.out(x)
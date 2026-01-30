# # models/img2smiles_model.py
# import torch
# import torch.nn as nn

# class Img2SMILESModel(nn.Module):
#     def __init__(self, encoder, decoder, enc_dim=1536, dec_dim=768):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         # 🔑 关键：将 encoder 输出投影到 decoder 的 d_model 维度
#         self.proj = nn.Sequential(
#                     nn.Linear(enc_dim, dec_dim),
#                     nn.LayerNorm(dec_dim),      # 👈 核心：归一化到 Decoder 的期望分布
#                     nn.Dropout(0.1)             # 👈 可选：增强鲁棒性
#                 )
        
#     def forward(self, images, tgt_tokens):
#         memory = self.encoder(images)      # [B, L, 1536]
#         memory = self.proj(memory)         # [B, L, 512] ← 必须加这行！
        
#         tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(images.device)
#         tgt_pad_mask = (tgt_tokens == self.decoder.embedding.padding_idx)
#         return self.decoder(tgt_tokens, memory, tgt_mask, tgt_pad_mask)

#     def generate_square_subsequent_mask(self, sz):
#         mask = torch.triu(torch.ones(sz, sz), diagonal=1)
#         return mask.masked_fill(mask == 1, float('-inf'))

#     @torch.no_grad()
#     def generate(self, images, sos_id, eos_id, max_len=256):
#         """自回归生成 SMILES（推理专用）"""
#         device = images.device
#         batch_size = images.size(0)

#         memory = self.encoder(images)   # [B, L, enc_dim]
#         memory = self.proj(memory)      # [B, L, dec_dim]

#         # 初始化输入
#         tgt = torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long)

#         finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

#         for _ in range(max_len):
#             tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
#             tgt_key_padding_mask = None  # 推理时通常不需要 pad mask

#             logits = self.decoder(
#                 tgt, memory,
#                 tgt_mask=tgt_mask,
#                 tgt_key_padding_mask=tgt_key_padding_mask
#             )  # [B, T, vocab]

#             next_token = logits[:, -1].argmax(dim=-1)  # [B]
#             # 对已结束样本，保持 eos（避免继续生成乱跑）
#             next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)

#             tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
#             finished |= (next_token == eos_id)

#             if finished.all():
#                 break

#         return tgt

#     @torch.no_grad()
#     def generate_with_conf(self, images, sos_id, eos_id, max_len=768):
#         """支持置信度返回的自回归生成"""
#         device = images.device
#         batch_size = images.size(0)

#         memory = self.encoder(images)   # [B, L, enc_dim]
#         memory = self.proj(memory)      # [B, L, dec_dim]

#         # 初始化输入
#         tgt = torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long)
        
#         # 用于存储每个 token 的置信度 (B, T)
#         confidences = torch.zeros((batch_size, 0), device=device)

#         finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

#         for _ in range(max_len):
#             tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
#             logits = self.decoder(
#                 tgt, memory,
#                 tgt_mask=tgt_mask,
#                 tgt_key_padding_mask=None
#             )  # [B, T, vocab]

#             # 1. 提取最后一个时间步的 Logits 并转为概率
#             last_logits = logits[:, -1] # [B, vocab]
#             probs = torch.softmax(last_logits, dim=-1) # [B, vocab]
            
#             # 2. 获取最大概率及其对应的索引
#             max_probs, next_token = torch.max(probs, dim=-1) # [B], [B]
            
#             # 3. 对已结束样本，强制设为 eos 且概率设为 1.0 (或保持不变)
#             next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            
#             # 4. 记录当前 step 的置信度
#             # 如果样本已经 finished，我们不希望已结束部分的低概率影响平均分
#             step_conf = torch.where(finished, torch.ones_like(max_probs), max_probs)
#             confidences = torch.cat([confidences, step_conf.unsqueeze(1)], dim=1)

#             tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
#             finished |= (next_token == eos_id)

#             if finished.all():
#                 break

#         # 计算整句的平均置信度（排除 padding 和 eos 之后的部分，可选）
#         # 这里提供一个简单的 batch 平均值
#         avg_confidence = confidences.mean(dim=1).cpu().numpy().tolist()

#         return tgt, avg_confidence

import torch
import torch.nn as nn

class Img2SMILESModel(nn.Module):
    def __init__(self, encoder, decoder, enc_dim=None, dec_dim=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # 自动探测维度
        if enc_dim is None: enc_dim = getattr(encoder, 'num_features', 1536)
        if dec_dim is None: dec_dim = getattr(decoder, 'd_model', 1024)
            
        print(f"✅ Model Config: Enc({enc_dim}) -> Bridge -> Dec({dec_dim}) with RoPE & Gating")

        # Bridge Layer: 投影 + 归一化
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),    # 先在原维度做一次非线性变换 [1536 -> 1536]
            nn.GELU(),                      # 激活函数
            nn.Linear(enc_dim, dec_dim),    # 再压缩到 decoder 维度 [1536 -> 1024]
            nn.LayerNorm(dec_dim),
            nn.Dropout(0.1)
        )
        
        # 探测 Padding ID
        if hasattr(decoder, 'embedding') and hasattr(decoder.embedding, 'padding_idx'):
            self.pad_idx = decoder.embedding.padding_idx
        else:
            self.pad_idx = 0 

    def forward(self, images, tgt_tokens):
        # 1. Encode
        memory = self.encoder(images)
        if memory.dim() == 4: # [B, H, W, C] -> [B, L, C]
            B, H, W, C = memory.shape
            memory = memory.view(B, H * W, C)
            
        # 2. Bridge
        memory = self.proj(memory)
        
        # 3. Masks
        # 生成 Padding Mask (非 0 即 True, 0 即 False -> 视 PyTorch 版本而定，这里使用 bool mask)
        # PyTorch MHA: True 也就是被 mask 掉 (忽略)
        tgt_pad_mask = (tgt_tokens == self.pad_idx)
        
        # 4. Decode
        # 注意: 这里不再需要传入 tgt_mask (sequence mask)，因为 RotarySelfAttention 内部处理了
        return self.decoder(
            tgt_tokens, 
            memory, 
            tgt_key_padding_mask=tgt_pad_mask
        )

    # --- 生成函数 (保持逻辑不变) ---
    @torch.no_grad()  # 1. 关键装饰器：推理模式，不计算梯度，节省显存并加速
    def generate(self, images, sos_id, eos_id, max_len=256):
        device = images.device
        batch_size = images.size(0)
        
        # --- 2. 编码阶段 (Encoder) ---
        # 作用：提取图像的视觉特征，只计算一次
        memory = self.encoder(images) 
        
        # 维度调整：Transformer 需要 [Batch, Seq_Len, Dim] 的形状
        # 如果 Encoder 输出的是 CNN 风格的 [B, H, W, C] 或 [B, C, H, W]，需要展平
        if memory.dim() == 4:
            B, H, W, C = memory.shape
            memory = memory.view(B, H * W, C) # 展平成 [Batch, H*W, Feature_Dim]
            
        memory = self.proj(memory) # 线性层：把 Encoder 的维度映射到 Decoder 需要的维度

        # --- 3. 初始化解码输入 (Init Decoder Input) ---
        # 创建一个初始序列，里面只有起始符 <sos>
        # 形状: [Batch_Size, 1] -> [[sos], [sos], ...]
        tgt = torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long)
        
        # 标记完成状态：记录哪些样本已经生成了 <eos>
        # 初始全是 False (0)
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

        # --- 4. 自回归循环 (Autoregressive Loop) ---
        for _ in range(max_len):
            # A. 前向传播
            # 注意：这里把当前的整个序列 tgt 都喂进去了 (这就是为什么慢的原因)
            # logits 形状: [Batch, Curr_Seq_Len, Vocab_Size]
            logits = self.decoder(tgt, memory)
            
            # B. 预测下一个词
            # 取最后一个时间步 ([: , -1]) 的输出，因为我们只关心下一个词是什么
            # argmax: 贪婪策略，直接选概率最大的那个词的 ID
            next_token = logits[:, -1].argmax(dim=-1) # 形状: [Batch]
            
            # C. 处理已完成的样本 (Masking Logic)
            # 逻辑：如果这个样本之前已经 finished 了，那强制把它的下一个词设为 <eos> (相当于 padding)
            # 如果没 finished，就用刚才预测出来的 next_token
            next_token = torch.where(finished, torch.tensor(eos_id, device=device), next_token)
            
            # D. 更新完成状态
            # 如果这次预测出了 <eos>，或者之前已经是 finished，那么状态变为 True
            finished |= (next_token == eos_id)
            
            # E. 拼接序列
            # 把预测出来的词拼接到 tgt 后面，作为下一次循环的输入
            # tgt 形状变化: [B, 1] -> [B, 2] -> [B, 3] ...
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            # F. 早停机制 (Early Stopping)
            # 如果 Batch 里所有的样本都 finish 了，直接退出循环，不用等到 max_len
            if finished.all(): break
            
        return tgt

    @torch.no_grad()
    def generate_with_conf(self, images, sos_id, eos_id, max_len=256):
        device = images.device
        batch_size = images.size(0)
        
        # --- 2. 编码阶段 ---
        memory = self.encoder(images)
        if memory.dim() == 4:
            B, H, W, C = memory.shape
            memory = memory.view(B, H * W, C)
        memory = self.proj(memory)

        # --- 3. 初始化 ---
        tgt = torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
        
        # [新增] 初始化置信度记录
        #以此记录每一步生成的概率。SOS token 的置信度默认为 1.0
        confidences = torch.ones((batch_size, 1), device=device, dtype=torch.float)

        # --- 4. 自回归循环 ---
        for _ in range(max_len):
            # A. 前向传播
            logits = self.decoder(tgt, memory)
            
            # B. 获取最后一个时间步的 Logits
            next_token_logits = logits[:, -1, :] # [Batch, Vocab]
            
            # [新增] C. 计算概率分布 (Softmax)
            # 这一步将 Logits 转为 0~1 的概率
            probs = torch.softmax(next_token_logits, dim=-1) # [Batch, Vocab]
            
            # [新增] D. 同时获取 最大概率(values) 和 对应的词(indices)
            # step_conf: [Batch], step_token: [Batch]
            step_conf, next_token = probs.max(dim=-1)
            
            # E. 处理已完成的样本 (Masking Logic)
            # 如果样本已完成，强制将 token 设为 EOS
            next_token = torch.where(finished, torch.tensor(eos_id, device=device), next_token)
            
            # [新增] 如果样本已完成，强制将该步置信度设为 1.0
            # 理由：EOS 之后的 padding 是确定的，不应该拉低整句的平均分数
            step_conf = torch.where(finished, torch.tensor(1.0, device=device), step_conf)
            
            # F. 更新状态
            finished |= (next_token == eos_id)
            
            # G. 拼接 Token 和 Confidence
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            # 将这一步的概率拼接到记录中
            confidences = torch.cat([confidences, step_conf.unsqueeze(1)], dim=1)
            
            if finished.all(): break
        
        # [新增] H. 计算整句平均置信度 (Sequence-level Confidence)
        # 排除掉第一个 SOS (索引 0)，计算剩余生成的有效部分的平均值
        # 注意：这里简单的取 mean 包含了 EOS 和 padding (因为上面设为了 1.0，不会拉低分数)
        seq_scores = confidences[:, 1:].mean(dim=1) # [Batch]
            
        # 返回：生成的序列, 每个词的概率, 整句平均分
        return tgt, seq_scores
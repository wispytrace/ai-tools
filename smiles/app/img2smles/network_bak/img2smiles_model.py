# models/img2smiles_model.py
import torch
import torch.nn as nn

class Img2SMILESModel(nn.Module):
    def __init__(self, encoder, decoder, enc_dim=1536, dec_dim=768):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 🔑 关键：将 encoder 输出投影到 decoder 的 d_model 维度
        self.proj = nn.Sequential(
                    nn.Linear(enc_dim, dec_dim),
                    nn.LayerNorm(dec_dim),      # 👈 核心：归一化到 Decoder 的期望分布
                    nn.Dropout(0.1)             # 👈 可选：增强鲁棒性
                )
        
    def forward(self, images, tgt_tokens):
        memory = self.encoder(images)      # [B, L, 1536]
        memory = self.proj(memory)         # [B, L, 512] ← 必须加这行！
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(images.device)
        tgt_pad_mask = (tgt_tokens == self.decoder.embedding.padding_idx)
        return self.decoder(tgt_tokens, memory, tgt_mask, tgt_pad_mask)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    @torch.no_grad()
    def generate(self, images, sos_id, eos_id, max_len=256):
        """自回归生成 SMILES（推理专用）"""
        device = images.device
        batch_size = images.size(0)

        memory = self.encoder(images)   # [B, L, enc_dim]
        memory = self.proj(memory)      # [B, L, dec_dim]

        # 初始化输入
        tgt = torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long)

        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            tgt_key_padding_mask = None  # 推理时通常不需要 pad mask

            logits = self.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )  # [B, T, vocab]

            next_token = logits[:, -1].argmax(dim=-1)  # [B]
            # 对已结束样本，保持 eos（避免继续生成乱跑）
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)

            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            finished |= (next_token == eos_id)

            if finished.all():
                break

        return tgt

    @torch.no_grad()
    def generate_with_conf(self, images, sos_id, eos_id, max_len=768):
        """支持置信度返回的自回归生成"""
        device = images.device
        batch_size = images.size(0)

        memory = self.encoder(images)   # [B, L, enc_dim]
        memory = self.proj(memory)      # [B, L, dec_dim]

        # 初始化输入
        tgt = torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long)
        
        # 用于存储每个 token 的置信度 (B, T)
        confidences = torch.zeros((batch_size, 0), device=device)

        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            logits = self.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None
            )  # [B, T, vocab]

            # 1. 提取最后一个时间步的 Logits 并转为概率
            last_logits = logits[:, -1] # [B, vocab]
            probs = torch.softmax(last_logits, dim=-1) # [B, vocab]
            
            # 2. 获取最大概率及其对应的索引
            max_probs, next_token = torch.max(probs, dim=-1) # [B], [B]
            
            # 3. 对已结束样本，强制设为 eos 且概率设为 1.0 (或保持不变)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            
            # 4. 记录当前 step 的置信度
            # 如果样本已经 finished，我们不希望已结束部分的低概率影响平均分
            step_conf = torch.where(finished, torch.ones_like(max_probs), max_probs)
            confidences = torch.cat([confidences, step_conf.unsqueeze(1)], dim=1)

            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            finished |= (next_token == eos_id)

            if finished.all():
                break

        # 计算整句的平均置信度（排除 padding 和 eos 之后的部分，可选）
        # 这里提供一个简单的 batch 平均值
        avg_confidence = confidences.mean(dim=1).cpu().numpy().tolist()

        return tgt, avg_confidence
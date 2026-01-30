import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer
from torch.amp import autocast, GradScaler
from datetime import datetime, timedelta
import time
# --- 导入您的自定义模块 ---
from network.encoder import SwinEncoder
from network.decoder import TransformerDecoder
from network.img2smiles_model import Img2SMILESModel
from data_process.chem_dataset import ChemDataset
import random
# --- 参数配置 ---
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")

# --- 1. 数据准备与 Tokenizer ---
def prepare_tokenizer():
    tokenizer_path = "tokenizer-selfies-ultimate.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()
    print(f"✅ Tokenizer loaded. Vocab size: {vocab_size}, padid:{pad_id}")
    return tokenizer, pad_id, vocab_size

# --- 2. 模型构建 (分层梯度控制) ---
def build_model(vocab_size, pad_id, device):
    encoder = SwinEncoder(
        checkpoint_path="/root/binghao/smiles/app/img2smles/pretrained_models/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth"
    )
    # 确保 Encoder 参数可更新（用于微调）
    for param in encoder.parameters():
        param.requires_grad = True
        
    decoder = TransformerDecoder(vocab_size=vocab_size)
    decoder.embedding.padding_idx = pad_id
    
    model = Img2SMILESModel(encoder, decoder).to(device)
    return model

# --- 3. 批处理函数 ---
def collate_fn(batch, pad_id):
    images, tokens = zip(*batch)
    images = torch.stack(images, dim=0)
    max_len = max(len(t) for t in tokens)
    padded_tokens = [t + [pad_id] * (max_len - len(t)) for t in tokens]
    tokens = torch.tensor(padded_tokens, dtype=torch.long)
    return images, tokens

def compute_custom_loss(logits, tgt_out, ce_criterion, pad_id=0, silent_weight=1):
    raw_criterion = nn.CrossEntropyLoss(reduction='none')
    loss_ce = ce_criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
    
    # 2. 找到 EOS 后的第一个位置并惩罚
    # logits shape: [B, L, V], tgt_out shape: [B, L]
    batch_size = tgt_out.size(0)
    first_pad_mask = (tgt_out == pad_id).float()
    first_pad_idx = first_pad_mask.argmax(dim=1) # 找到每行第一个 pad
    
    # 构造一个单点掩码
    one_hot_mask = torch.zeros_like(tgt_out, dtype=torch.float)
    for i in range(batch_size):
        one_hot_mask[i, first_pad_idx[i]] = 1.0
        
    # 计算原始点损失
    loss_raw = raw_criterion(logits.transpose(1, 2), tgt_out) # [B, L]
    loss_silent = (loss_raw * one_hot_mask).sum() / batch_size
    
    return loss_ce + silent_weight * loss_silent

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch, log_interval=10):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    num_batches = min(num_batches, 10000)
    
    accumulation_steps = 1
    optimizer.zero_grad() 
    current_grad_norm = 0.0
    start_time = time.time()
    
    for i, (images, tokens) in enumerate(dataloader):

        if i > num_batches:
            break

        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]

        # 混合精度训练
        with autocast('cuda'):
            logits = model(images, tgt_in)
            # loss = compute_custom_loss(logits, tgt_out, criterion, pad_id=2)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            # 重要：损失需要除以累加步数，以保持梯度量级一致
            loss = loss / accumulation_steps

        # 反向传播，累积梯度
        scaler.scale(loss).backward()

        # 当达到累加步数，或者到达最后一个 batch 时，更新参数
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
            # 1. 解算梯度
            scaler.unscale_(optimizer)
            
            # 2. 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            current_grad_norm = grad_norm.item() # 转为标量用于显示
            # 3. 检查梯度并更新

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n⚠️ 警告: Epoch {epoch} Step {i} 发现梯度异常 (norm={grad_norm})，跳过更新")
                scaler.update()
                optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 更新完后清空梯度
                
                # 注意：如果是每个 step 更新一次的学习率，放这里
                if scheduler is not None:
                    scheduler.step()

            if current_grad_norm > 2000.0:
                old_scale = scaler.get_scale()
                new_scale = old_scale * 0.5
                scaler._scale = torch.tensor([new_scale]).to(device)
                print(f"检测到高 Norm ({current_grad_norm:.2f}), 主动将 Scale 从 {old_scale} 降至 {new_scale}")

        total_loss += loss.item() * accumulation_steps # 恢复原始 loss 用于统计

        try:
            if (i > 0) and (i % 1000 == 0):
                checkpoint_path = os.path.join("/root/binghao/smiles/app/img2smles/temp_models", f"checkpoint_latest.pth")
                state = {
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss.item() * accumulation_steps
                }
                torch.save(state, checkpoint_path)
                print(f"\n💾 已自动保存中间断点: {checkpoint_path}")
        except Exception as e:
            print(f"\n❌ 自动保存断点失败: {e}")

        if log_interval and i % log_interval == 0 and i > 0:
            avg_loss = total_loss / (i + 1)
            lr_enc = optimizer.param_groups[0]['lr']
            lr_dec = optimizer.param_groups[1]['lr']
            
            elapsed_time = time.time() - start_time
            it_per_sec = (i + 1) / elapsed_time
            remaining_steps = num_batches - (i + 1)
            remaining_time = remaining_steps / it_per_sec
            
            eta_str = str(timedelta(seconds=int(remaining_time)))
            elapsed_str = str(timedelta(seconds=int(elapsed_time)))
            current_scale = scaler.get_scale()

            print(f"Epoch {epoch} [{i}/{num_batches}] | Loss: {avg_loss:.4f} | Norm: {current_grad_norm:.2f} | "
                            f"{it_per_sec:.2f} it/s | Elapsed: {elapsed_str} | ETA: {eta_str} | EncLR: {lr_enc:.1e} | DecLR: {lr_dec:.1e} | Scale: {current_scale:.2f}")


    return total_loss / num_batches

# --- 5. 评估函数 ---
# def evaluate(model, dataloader, criterion, device, tokenizer, max_samples=100):
#     """
#     仅在测试集的子集上进行评估，节省时间。
#     max_samples: 评估的最大样本数
#     """
#     model.eval()
#     total_loss = 0.0
#     processed_samples = 0
#     num_display = 10 # 依然只预览 3 条对比
#     displayed = 0

#     # 计算需要跑多少个 batch
#     total_batches = len(dataloader)
#     max_batches = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
#     selected_batches = set(random.sample(range(total_batches), min(max_batches, total_batches)))
#     with torch.no_grad():
#         for i, (images, tokens) in enumerate(dataloader):

#             if i not in selected_batches:
#                             continue

#             images, tokens = images.to(device), tokens.to(device)
#             tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]

#             with autocast('cuda'):
#                 logits = model(images, tgt_in)
#                 # loss = compute_custom_loss(logits, tgt_out, criterion)
#                 loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
#                 total_loss += loss.item()

#             # 解码预览逻辑
#             if displayed < num_display:
#                 pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
#                 tgt_ids = tgt_out[0].cpu().tolist()

#                 def decode_tokens(ids):
#                     tokens_list = [tokenizer.id_to_token(idx) for idx in ids]
#                     return "".join([t for t in tokens_list if t not in ["<pad>", "<sos>", "<eos>"]])

#                 print(f"\n  🔎 Sample {displayed + 1} Preview:")
#                 print(f"    🎯 GT: {decode_tokens(tgt_ids)}")
#                 print(f"    🤖 PR: {decode_tokens(pred_ids)}")
#                 displayed += 1
            
#             processed_samples += images.size(0)

#     avg_loss = total_loss / (i + 1)
#     print(f"\n🧪 Partial Evaluation Done on {processed_samples} samples. Avg Loss: {avg_loss:.4f}")
#     return avg_loss

def evaluate(model, dataloader, criterion, device, tokenizer, max_samples=100):
    model.eval()
    total_loss = 0.0
    processed_samples = 0
    num_display = 10
    displayed = 0

    # 1. 核心修改：从 dataloader 的索引中随机选出要跑的 batch 索引
    total_batches = len(dataloader)
    max_batches = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
    # 随机选出 max_batches 个序号
    selected_batches = set(random.sample(range(total_batches), min(max_batches, total_batches)))

    with torch.no_grad():
        for i, (images, tokens) in enumerate(dataloader):
            # 2. 最小修改：如果当前 batch 不在随机选中的序号里，直接跳过
            if i not in selected_batches:
                continue

            images, tokens = images.to(device), tokens.to(device)
            tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]

            with autocast('cuda'):
                logits = model(images, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                total_loss += loss.item()

            # 解码预览逻辑
            if displayed < num_display:
                # 遍历当前 batch 确保展示多样性
                for b in range(logits.size(0)):
                    if displayed >= num_display: break
                    
                    pred_ids = logits[b].argmax(dim=-1).cpu().tolist()
                    tgt_ids = tgt_out[b].cpu().tolist()

                    def decode_tokens(ids):
                        tokens_list = [tokenizer.id_to_token(idx) for idx in ids]
                        return "".join([t for t in tokens_list if t not in ["<pad>", "<sos>", "<eos>"]])

                    print(f"\n  🔎 Sample {displayed + 1} Preview:")
                    print(f"    🎯 GT: {decode_tokens(tgt_ids)}")
                    print(f"    🤖 PR: {decode_tokens(pred_ids)}")
                    displayed += 1
            
            processed_samples += images.size(0)
            # 达到预定 batch 数量后退出
            if len(selected_batches) > 0 and processed_samples >= max_samples:
                break

    avg_loss = total_loss / len(selected_batches) if len(selected_batches) > 0 else 0
    print(f"\n🧪 Randomized Evaluation Done on {processed_samples} samples. Avg Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    resume = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training started on: {device}")
    # --- 初始化 ---
    tokenizer, pad_id, vocab_size = prepare_tokenizer()
    model = build_model(vocab_size, pad_id, device)

    eos_id = tokenizer.token_to_id("<eos>")
    weights = torch.ones(vocab_size).to(device)
    weights[eos_id] = 5.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.009, weight=weights)
    
    # 数据加载
    # full_dataset = ChemDataset("dataset/images_24", "dataset/images_24/label.txt", tokenizer, max_samples=2500000)
    # excpt_dataset = ChemDataset("dataset/images_24_except", "dataset/images_24_except/label.txt", tokenizer, max_samples=1200000)

    full_dataset = ChemDataset("dataset/images_24", "dataset/images_24/label.txt", tokenizer, max_samples=2500000,is_aug=True)
    excpt_dataset = ChemDataset("dataset/images_24_except", "dataset/images_24_except/label.txt", tokenizer, max_samples=1200000,is_aug=True)

    train_size = int(0.92 * len(full_dataset))
    excpt_size = int(0.92 * len(excpt_dataset))
    train_dataset, test_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_expect_dateset, test_excpect_dataset = random_split(excpt_dataset, [excpt_size, len(excpt_dataset) - excpt_size])
    
    final_train_dataset = train_dataset + train_expect_dateset

    final_test_dataset = test_dataset + test_excpect_dataset

    train_loader = DataLoader(final_train_dataset, batch_size=14, shuffle=True, 
                              collate_fn=lambda b: collate_fn(b, pad_id), num_workers=4, pin_memory=True)
    test_loader = DataLoader(final_test_dataset, batch_size=14, shuffle=False, 
                             collate_fn=lambda b: collate_fn(b, pad_id), num_workers=4)

    # epochs = 8
    # optimizer = torch.optim.AdamW([
    #     {'params': model.encoder.parameters(), 'lr': 5e-6},
    #     {'params': model.decoder.parameters(), 'lr': 1e-3}
    # ], weight_decay=1e-2)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr=[5e-5, 1e-3],
    #     steps_per_epoch=len(train_loader), 
    #     epochs=epochs,
    #     pct_start=0.1
    # )

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 2e-5},
        {'params': model.decoder.parameters(), 'lr': 2e-4}
    ], weight_decay=1e-2)

    epochs = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[5e-5, 5e-4], # 峰值学习率建议下调 2-3 倍，防止在大数据集遇到异常样本时直接炸掉
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.05,      # 大数据集通常缩短 warmup 占比（因为总 step 多了）
        div_factor=20,       # 初始 LR = max_lr / 20
        final_div_factor=1e4 # 训练结束时 LR 降得更低，彻底消除 Sample 6 的复读机现象
    )
    # optimizer = torch.optim.AdamW([
    #     {'params': model.encoder.parameters(), 'lr': 1e-6},
    #     {'params': model.decoder.parameters(), 'lr': 1e-5}
    # ], weight_decay=1e-2)

    # epochs = 10
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr=[1e-5, 1e-4], # 峰值学习率建议下调 2-3 倍，防止在大数据集遇到异常样本时直接炸掉
    #     steps_per_epoch=len(train_loader), 
    #     epochs=epochs,
    #     pct_start=0.05,      # 大数据集通常缩短 warmup 占比（因为总 step 多了）
    #     div_factor=20,       # 初始 LR = max_lr / 20
    #     final_div_factor=1e4 # 训练结束时 LR 降得更低，彻底消除 Sample 6 的复读机现象
    # )
    scaler = GradScaler(init_scale=2**14)

    # --- 核心：断点加载逻辑 ---
    start_epoch = 0
    best_test_loss = float('inf')
    
    if resume:
        resume_path = "/root/binghao/smiles/app/img2smles/temp_models/checkpoint_latest.pth"
        
        if os.path.exists(resume_path):
            print(f"🔄 发现断点，正在从 {resume_path} 恢复训练...")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 恢复模型和优化器状态
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复混合精度 scaler
            # if 'scaler_state_dict' in checkpoint:
            #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # 恢复调度器 (注意：OneCycleLR 恢复时对 step 的处理比较敏感)
            # if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            # 恢复起始 Epoch 
            start_epoch = 0
            print(f"✅ 成功从 Epoch {start_epoch} 恢复。")
        else:
            print("🆕 未发现断点，将从头开始训练。")

    # --- 训练循环 ---
    # 修改 range 范围，从 start_epoch 开始
    for epoch in range(start_epoch, epochs):
        print(f"\n🗓️ Epoch {epoch}/{epochs-1}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch)
        test_loss = evaluate(model, test_loader, criterion, device, tokenizer)

        # 保存最新的模型 (用于主循环后的持久化)
        checkpoint_state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch + 1, # 下次从下一轮开始
            'test_loss': test_loss
        }
        torch.save(checkpoint_state, LAST_MODEL_PATH)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(checkpoint_state, BEST_MODEL_PATH)
            print(f"⭐ Best Model Updated! Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
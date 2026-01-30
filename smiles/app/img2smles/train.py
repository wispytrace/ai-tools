import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tokenizers import Tokenizer
from torch.amp import autocast, GradScaler
from datetime import timedelta
import time
import random
import numpy as np
# 导入自定义模块 (假设路径正确)
from train_config import Config
from network.encoder import SwinEncoder
from network.decoder import TransformerDecoder
from network.img2smiles_model import Img2SMILESModel
from network.loss import FocalLoss
from data_process.chem_dataset import ChemDataset

import datetime

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k] = v

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, val in self.meters.items():
            loss_str.append(f"{name}: {val:.4f}")
        return self.delimiter.join(loss_str)

class SmoothedValue:
    """平滑统计 Loss 和计算 ETA"""
    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.total += value * n
        self.count += n

    @property
    def median(self):
        return torch.tensor(self.deque).median().item()

    @property
    def avg(self):
        return torch.tensor(self.deque).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

# --- 辅助工具函数 ---
def collate_fn(batch, pad_id):
    images, tokens = zip(*batch)
    images = torch.stack(images, dim=0)
    max_len = max(len(t) for t in tokens)
    padded_tokens = [t + [pad_id] * (max_len - len(t)) for t in tokens]
    return images, torch.tensor(padded_tokens, dtype=torch.long)

def decode_tokens(ids, tokenizer):
    tokens_list = [tokenizer.id_to_token(idx) for idx in ids]
    prsed_token_list = []
    for token in tokens_list:
        if token == "<eos>":
            break
        prsed_token_list.append(token)
    return "".join([t for t in prsed_token_list if t not in ["<pad>", "<sos>", "<eos>"]])

def get_log_scaled_weights(freq_file: str, min_w: float = 1.0, max_w: float = 10.0):
    """
    🌟 推荐方案：先取对数，再归一化。
    这样可以防止 [C] 的数量太大，导致中间频数的原子权重全部被挤压到最高点。
    """
    print(f"⚖️ 正在计算权重 (线性反向归一化, 范围: {min_w} - {max_w})...")
    data = []
    # 1. 读取频数文件
    try:
        with open(freq_file, "r") as f:
            header = next(f, None) # 跳过表头
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    token = parts[0]
                    count = int(parts[1])
                    data.append({'token': token, 'count': count})
    except FileNotFoundError:
        print(f"❌ 找不到文件: {freq_file}")
        return {}

    if not data:
        return {}
    # ... (读取文件代码省略，同上) ...
    # 假设 data 已经读好了: [{'token': 'C', 'count': 100000}, ...]
    
    # 1. 提取 count 并取对数 (log10 或 loge 都可以)
    counts = np.array([item['count'] for item in data], dtype=np.float32)
    log_counts = np.log10(counts + 1e-8) # 加一点点防止 log(0)
    
    # 2. 获取 Log 的最大最小值
    # 注意：counts 是降序的，所以 log_counts 也是降序
    log_max = log_counts[0]  # log(Count_C)
    log_min = log_counts[-1] # log(Count_Last)
    
    print(f"   📊 Log频数范围: Max={log_max:.2f}, Min={log_min:.2f}")
    
    weight_dict = {}
    print(f"{'Token':<15} | {'Count':<10} | {'Weight':<10}")
    print("-" * 45)
    
    denom = log_max - log_min
    
    for i, item in enumerate(data):
        current_log = log_counts[i]
        
        if denom == 0:
            final_weight = min_w
        else:
            # 使用 Log 值计算比例
            ratio = (log_max - current_log) / denom
            final_weight = min_w + ratio * (max_w - min_w)
            
        weight_dict[item['token']] = final_weight
        
        if i < 5 or i >= len(data) - 5:
            print(f"{item['token']:<15} | {item['count']:<10} | {final_weight:.4f}")

    return weight_dict

# --- 核心逻辑 ---

def evaluate(model, dataloader, criterion, device, tokenizer, max_samples=1000):
    model.eval()
    total_loss = 0.0
    processed_samples = 0
    num_display = 10 # 减少一点预览，保持日志整洁
    displayed = 0

    total_batches = len(dataloader)
    max_batches = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
    selected_batches = set(random.sample(range(total_batches), min(max_batches, total_batches)))

    with torch.no_grad():
        for i, (images, tokens) in enumerate(dataloader):
            if i not in selected_batches: continue

            images, tokens = images.to(device), tokens.to(device)
            tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]

            with autocast('cuda'):
                logits = model(images, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                total_loss += loss.item()

            if displayed < num_display:
                for b in range(min(logits.size(0), 1)): # 每个选中 batch 预览 1 条
                    pred_ids = logits[b].argmax(dim=-1).cpu().tolist()
                    tgt_ids = tgt_out[b].cpu().tolist()
                    print(f"\n  🔎 Sample {displayed + 1} Preview:")
                    print(f"    🎯 GT: {decode_tokens(tgt_ids, tokenizer)}")
                    print(f"    🤖 PR: {decode_tokens(pred_ids, tokenizer)}")
                    displayed += 1
            
            processed_samples += images.size(0)
            if processed_samples >= max_samples: break

    avg_loss = total_loss / len(selected_batches) if selected_batches else 0
    print(f"\n🧪 Evaluation Loss: {avg_loss:.4f}")
    return avg_loss

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch, cfg):
    model.train()
    
    # 定义指标统计器
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    print_freq = cfg.LOG_STEP_INTERVAL

    # 梯度累积步数 (建议在 Config 中定义，默认为 1)
    accum_steps = getattr(cfg, 'ACCUMULATION_STEPS', 1) 
    
    optimizer.zero_grad()
    
    start_time = time.time()
    end = time.time()
    
    num_batches = len(dataloader)
    if hasattr(cfg, 'STEPS_PER_EPOCH'):
        num_batches = min(num_batches, cfg.STEPS_PER_EPOCH)

    for i, (images, tokens) in enumerate(dataloader):
        if i >= num_batches: break

        # 数据移动
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]

        # --- 1. Forward ---
        with autocast('cuda'):
            logits = model(images, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            
            # 关键：Loss 必须除以累积步数，否则梯度会偏大
            loss = loss / accum_steps

        # --- 2. Backward ---
        scaler.scale(loss).backward()

        # --- 3. Optimizer Step (仅在累积满足时执行) ---
        if (i + 1) % accum_steps == 0:
            # Unscale 只有在 step 前做一次
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)

            # 检查是否有非法梯度
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                scaler.step(optimizer)
                scaler.update()
                if scheduler: scheduler.step()
            else:
                print(f"⚠️ [Epoch {epoch} Step {i}] Skipped step due to Inf/NaN grad (Norm: {grad_norm.item():.2f})")
                scaler.update() # 即使跳过也要更新 scaler 状态
            
            # 您的自定义 Scale 调整逻辑 (保留)
            if grad_norm.item() > 2000.0:
                try:
                    old_scale = scaler.get_scale()
                    scaler.update(new_scale=old_scale * 0.5)
                    print(f"📉 High Norm detected. Scale dropped: {old_scale} -> {scaler.get_scale()}")
                except:
                    pass # 兼容不同 PyTorch 版本

            optimizer.zero_grad()

        # --- 4. Logging & ETA Calculation ---
        # 还原 Loss 数值用于显示 (乘回去)
        loss_val = loss.item() * accum_steps
        
        # 计算时间
        batch_time = time.time() - end
        end = time.time()
        
        # 记录指标
        metric_logger.update(loss=loss_val)
        metric_logger.update(time=batch_time)

        if i % print_freq == 0 and i > 0:
            # 计算预计剩余时间 (ETA)
            steps_remaining = num_batches - i
            eta_seconds = metric_logger.meters['time'] * steps_remaining # 使用移动平均时间
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            # 获取显存占用
            mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            
            # 获取双学习率
            enc_lr = optimizer.param_groups[0]['lr']
            dec_lr = optimizer.param_groups[1]['lr']
            
            print(
                f"{header} [{i}/{num_batches}] "
                f"eta: {eta_string} | "
                f"loss: {metric_logger.meters['loss']:.4f} | "
                f"grad_norm: {grad_norm.item() if 'grad_norm' in locals() else 0:.1f} | "
                f"enc_lr: {enc_lr:.1e} | "
                f"dec_lr: {dec_lr:.1e} | "
                f"mem: {mem_mb:.0f}MB"
            )

        # --- Auto Save ---
        if i % cfg.SAVE_STEP_INTERVAL == 0 and i > 0:
            # 确保目录存在
            save_dir = os.path.dirname(cfg.RESUME_PATH)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            state = {
                'epoch': epoch,
                'step': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }
            torch.save(state, cfg.RESUME_PATH)
            # 不打印这行，避免刷屏，除非你很想看

    # Epoch 结束统计
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'{header} Total time: {total_time_str}')
    
    return metric_logger.meters['loss'] # 返回最后的移动平均 Loss

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tokenizer & Data
    tokenizer = Tokenizer.from_file(cfg.TOKENIZER_PATH)
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()

    train_sets, test_sets = [], []
    for img_dir, label_file, max_s in cfg.DATA_SOURCES:
        ds = ChemDataset(img_dir, label_file, tokenizer, max_samples=max_s, is_aug=cfg.IS_AUG)
        tr_sz = int(cfg.TRAIN_RATIO * len(ds))
        tr, te = random_split(ds, [tr_sz, len(ds) - tr_sz])
        train_sets.append(tr)
        test_sets.append(te)

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=cfg.BATCH_SIZE, shuffle=True, 
                              collate_fn=lambda b: collate_fn(b, pad_id), num_workers=4, pin_memory=True)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=cfg.BATCH_SIZE, shuffle=False, 
                             collate_fn=lambda b: collate_fn(b, pad_id), num_workers=4)

    # 2. Model & Loss
    encoder = SwinEncoder(checkpoint_path=cfg.PRETRAINED_PATH)
    decoder = TransformerDecoder(vocab_size=vocab_size)
    decoder.embedding.padding_idx = pad_id
    model = Img2SMILESModel(encoder, decoder).to(device)
    
    try:
        print("🚀 Compiling model with torch.compile...")
        model = torch.compile(model, mode="default") 
    except Exception as e:
        print(f"⚠️ torch.compile failed (ignored): {e}")


    weights = torch.ones(vocab_size).to(device)
    weights_map = get_log_scaled_weights(cfg.FREQUENCT_PARH)
    eos_id = tokenizer.token_to_id("<eos>")
    weights[eos_id] = cfg.EOS_WEIGHT
    for t, w in weights_map.items():
            tid = tokenizer.token_to_id(t)
            if tid is not None:
                weights[tid] = float(w)

    criterion = FocalLoss(
        gamma=2.0,            # 聚焦力度，2.0 是标准值
        alpha=weights,  # 结合类别权重
        ignore_index=pad_id,
        reduction='mean'
    )

    # criterion = nn.CrossEntropyLoss(
    #     ignore_index=pad_id, 
    #     label_smoothing=cfg.LABEL_SMOOTHING, 
    #     weight=weights
    # )

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg.ENCODER_LR},
        {'params': model.decoder.parameters(), 'lr': cfg.DECODER_LR}
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.MAX_LR, epochs=cfg.EPOCHS,
        steps_per_epoch=min(len(train_loader), cfg.STEPS_PER_EPOCH), pct_start=cfg.PCT_START
    )
    scaler = GradScaler()

    # 4. Resume
    start_epoch = 0
    if cfg.RESUME and os.path.exists(cfg.RESUME_PATH):
        print(f"🔄 Loading checkpoint from {cfg.RESUME_PATH}")
        ckpt = torch.load(cfg.RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        # start_epoch = ckpt['epoch'] # 如果需要从特定 epoch 开始

    # 5. Loop
    best_loss = float('inf')
    for epoch in range(start_epoch, cfg.EPOCHS):
        print(f"\n--- Epoch {epoch} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch, cfg)
        test_loss = evaluate(model, test_loader, criterion, device, tokenizer)
        if test_loss < best_loss:
            checkpoint_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch + 1, # 下次从下一轮开始
                'test_loss': test_loss
            }
            best_loss = test_loss
            best_model_path = os.path.join(cfg.CHECKPOINT_DIR)
            os.makedirs(best_model_path, exist_ok=True) 
            torch.save(checkpoint_state, os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"))
            print("⭐ Best model saved!")

if __name__ == "__main__":
    main()
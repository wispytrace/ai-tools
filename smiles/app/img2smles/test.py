import torch
import sys
import os

# 假设你的模型定义在以下路径
from network.encoder import SwinEncoder
from network.decoder import TransformerDecoder
from network.img2smiles_model import Img2SMILESModel

DEVICE = "cpu"  # 用 CPU 避免 GPU 内存问题
MODEL_PATH = "/root/binghao/smiles/app/img2smles/last_model.pth"

def load_trained_img2smiles_model(model_path, vocab_size, pad_id, device="cuda"):
    """
    仅加载模型权重用于推理
    
    Args:
        model_path: 模型 checkpoint 路径
        vocab_size: SMILES 词表大小（必须与训练时一致！）
        enc_dim: 编码器输出维度（必须与训练时一致）
        dec_dim: 解码器隐藏维度（必须与训练时一致）
        pad_id: padding token ID
        device: 设备 ("cuda" or "cpu")
    """
    # 1. 重建模型结构（确保与训练时完全一致！）
    encoder = SwinEncoder(
        checkpoint_path=None
    )
    
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
    )

    decoder.embedding.padding_idx = pad_id

    # ⚠️ 关键修复：只创建一次模型，并立即移到 device
    model = Img2SMILESModel(
        encoder=encoder,
        decoder=decoder,
    ).to(device)  # ← 立即移到设备！

    # 2. 加载 state_dict
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 3. 尝试严格加载（推荐先 strict=True 调试）
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Strict loading succeeded.")
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("🔄 Trying non-strict loading...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"❌ Missing keys: {missing}")
        if unexpected:
            print(f"❓ Unexpected keys: {unexpected}")

    print(f"✅ Model loaded from {model_path} on {device}")
    return model.eval()  # 切换为推理模式


def get_param_stats(param):
    """获取参数的统计信息"""
    if param.numel() == 0:
        return {"mean": 0, "std": 0, "max": 0, "min": 0}
    return {
        "mean": param.mean().item(),
        "std": param.std().item(),
        "max": param.max().item(),
        "min": param.min().item()
    }

def print_param_comparison(original_state, loaded_model, top_k=20):
    """打印前 top_k 个参数的对比"""
    print("="*100)
    print(f"{'Parameter Name':<50} | {'Original (mean±std)':<20} | {'Loaded (mean±std)':<20} | Match?")
    print("-"*100)
    
    mismatch_count = 0
    checked = 0

    print("First 20 keys in checkpoint:")
    for i, key in enumerate(list(original_state.keys())[:20]):
        print(f"{i+1:2d}. {key}")

        
    for name, param in loaded_model.named_parameters():
        if name not in original_state:
            print(f"{name:<50} | {'MISSING IN CKPT':<20} | .... | ❌")
            continue
            
        orig_param = original_state[name]
        loaded_param = param
        
        # 获取统计信息
        orig_stats = get_param_stats(orig_param)
        loaded_stats = get_param_stats(loaded_param)
        
        # 检查是否匹配
        is_match = torch.allclose(orig_param, loaded_param, atol=1e-6)
        match_str = "✅" if is_match else "❌"
        if not is_match:
            mismatch_count += 1
        
        # 打印
        orig_str = f"{orig_stats['mean']:.4f}±{orig_stats['std']:.4f}"
        loaded_str = f"{loaded_stats['mean']:.4f}±{loaded_stats['std']:.4f}"
        print(f"{name[:48]:<50} | {orig_str:<20} | {loaded_str:<20} | {match_str}")
        
        checked += 1
        if checked >= top_k:
            break
    
    print("-"*100)
    print(f"Total checked: {checked}, Mismatches: {mismatch_count}")
    if mismatch_count == 0:
        print("🎉 ALL PARAMETERS MATCH!")
    else:
        print("⚠️  Some parameters do NOT match!")

def main():
    # 1. 加载原始 checkpoint
    print("📥 Loading original checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    original_state = checkpoint['model_state_dict']
    
    # 2. 获取 vocab_size 和 pad_id（从 checkpoint 或 tokenizer）
    # 这里假设你知道 vocab_size（或从 checkpoint 中保存）
    vocab_size = 273  # ← 替换为你的实际 vocab_size
    pad_id = 2         # ← 替换为你的实际 pad_id
    
    # 3. 重建模型并加载权重
    print("🔧 Rebuilding model and loading weights...")
    model = load_trained_img2smiles_model(
        model_path=MODEL_PATH,
        vocab_size=vocab_size,
        pad_id=pad_id,
        device="cpu"
    )
    model.load_state_dict(original_state, strict=True)
    
    # 4. 打印关键参数对比
    print_param_comparison(original_state, model, top_k=30)

if __name__ == "__main__":
    main()

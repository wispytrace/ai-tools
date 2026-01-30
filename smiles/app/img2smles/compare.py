import torch
import sys
import os

# ====== 配置区（按你的路径修改）======
CKPT_PATH = "/root/binghao/smiles/app/img2smles/last_model.pth"
# 如果你知道训练时的模型定义，导入它（关键！）
# 例如：
# from train_script import Img2SMILESModel  # ← 替换为你的训练模型定义

# 如果无法导入训练模型，就在这里重建（必须与训练时完全一致！）
from network.encoder import SwinEncoder
from network.decoder import TransformerDecoder
from network.img2smiles_model import Img2SMILESModel

def build_inference_model(vocab_size=1000, pad_id=0):
    """重建推理模型（必须与训练时结构一致！）"""
    encoder = SwinEncoder(
        checkpoint_path=None
    )
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
    )
    model = Img2SMILESModel(
        encoder=encoder,
        decoder=decoder,
    )
    return model

# ====== 执行区（无需修改）======
def main():
    # 1. 加载 checkpoint 的 keys
    print("🔍 Loading checkpoint keys...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if 'model_state_dict' in ckpt:
        ckpt_keys = set(ckpt['model_state_dict'].keys())
    else:
        ckpt_keys = set(ckpt.keys())
    
    print(f"✅ Checkpoint has {len(ckpt_keys)} parameters.")
    
    # 2. 获取当前模型的 keys
    print("\n🔍 Building inference model and getting its keys...")
    try:
        model = build_inference_model()
        model = torch.compile(model)
        
        model_keys = set(model.state_dict().keys())
        print(f"✅ Inference model has {len(model_keys)} parameters.")
    except Exception as e:
        print(f"❌ Failed to build model: {e}")
        return

    # 3. 对比
    only_in_ckpt = ckpt_keys - model_keys
    only_in_model = model_keys - ckpt_keys
    
    print(f"\n📊 Comparison Result:")
    print(f"- Keys only in checkpoint: {len(only_in_ckpt)}")
    print(f"- Keys only in model:      {len(only_in_model)}")
    print(f"- Common keys:             {len(ckpt_keys & model_keys)}")
    
    if only_in_ckpt:
        print("\n🔑 Keys missing in model (from checkpoint):")
        for k in sorted(only_in_ckpt)[:20]:  # 只打印前20个
            print(f"  - {k}")
        if len(only_in_ckpt) > 20:
            print(f"  ... and {len(only_in_ckpt)-20} more")
    
    if only_in_model:
        print("\n🔑 Keys missing in checkpoint (from model):")
        for k in sorted(only_in_model)[:20]:
            print(f"  - {k}")
        if len(only_in_model) > 20:
            print(f"  ... and {len(only_in_model)-20} more")
    
    if not only_in_ckpt and not only_in_model:
        print("\n🎉 PERFECT MATCH! Model structure is consistent.")
    else:
        print("\n⚠️  MISMATCH DETECTED! Model structure differs from checkpoint.")

if __name__ == "__main__":
    main()

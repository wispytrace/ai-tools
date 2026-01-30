# inference.py
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import sys
import os

from network.encoder import SwinEncoder
from network.decoder import TransformerDecoder  # 确保路径正确
from network.img2smiles_model import Img2SMILESModel
import torchvision
from tokenizers import Tokenizer
import selfies as sf
# ======================
# 配置（根据你的训练设置修改）
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/root/binghao/smiles/app/img2smles/checkpoints/last_model.pth"

# 从你的训练配置中获取这些值
MAX_LEN = 512

# ======================
# 加载完整模型（包括你训练的权重）
# ======================
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

    model = torch.compile(model)
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

# ======================
# 自回归生成 SMILES
# ======================

def prepare_tokenizer(tokenizer_path = "tokenizer-selfies-ultimate.json"):
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    sos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()

    print(f"✅ Tokenizer loaded. Vocab size: {vocab_size}")
    return tokenizer, vocab_size

def generate_smiles(model, image_tensor, tokenizer, max_len=MAX_LEN):
    sos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")
    device = image_tensor.device

    with torch.no_grad():
        generated_ids = model.generate(image_tensor, sos_id, eos_id, max_len).tolist()
                
    return generated_ids

def ids_to_smiles(token_ids, tokenizer):
    """
    使用训练好的 tokenizer 将 token IDs 转为 SMILES 字符串
    """
    smiles_str = tokenizer.decode(token_ids, skip_special_tokens=True)
    smiles_str.strip()
    
    return smiles_str
# ======================
# 主函数
# ======================
def main(image_path):
    # 1. 加载模型

    tokenizer, vocab_size = prepare_tokenizer()

    model = load_trained_img2smiles_model(
        model_path=MODEL_PATH,
        vocab_size=vocab_size,
        pad_id=tokenizer.token_to_id("<pad>"),
        device=DEVICE
    )
    
    # 2. 加载并预处理图像
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((384, 384)),
        torchvision.transforms.ToTensor(),  # 转为 [0,1] 的 tensor
        torchvision.transforms.Normalize(   # ← 关键！归一化到 ImageNet 分布
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB") 
    img_tensor = transform(image)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    img_tensor = img_tensor.to(DEVICE)        # ← 移到 GPU/CPU 与模型一致

    # 3. 生成 token IDs
    print("🤖 Generating SMILES...")
    token_ids = generate_smiles(model, img_tensor, tokenizer)
    print("Token IDs:", token_ids)
    
    smiles_str = ids_to_smiles(token_ids, tokenizer)
    result = [char for char in list(smiles_str) if char != ' ']
    print("Decoded SMILES string:", ''.join(result))
    smiles_str = ''.join(result)
    smiles_str = sf.decoder(smiles_str.strip())
    print(f"\n🎉 Generated SMILES: {smiles_str}")
    return smiles_str

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    main(args.image)

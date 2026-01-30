# models/encoder.py
import sys
sys.path.append("/mnt/smiles2img/Swin-Transformer")  # 替换为实际路径

import torch
import torch.nn as nn
from config import get_config, _C
from models import build_model

def get_swinv2_large_384_config():
    cfg = _C.clone()
    cfg.MODEL.TYPE = "swinv2"
    cfg.MODEL.NAME = "swinv2_large_patch4_window12to24_192to384"
    cfg.MODEL.SWINV2.EMBED_DIM = 192
    cfg.MODEL.SWINV2.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWINV2.NUM_HEADS = [6, 12, 24, 48]
    cfg.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [12, 12, 12, 6]
    cfg.MODEL.SWINV2.WINDOW_SIZE = 24
    cfg.DATA.IMG_SIZE = 384
    cfg.MODEL.NUM_CLASSES = 1000
    return cfg

class SwinEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        config = get_swinv2_large_384_config()
        self.swin = build_model(config)
        if checkpoint_path:
            self.load_pretrained(checkpoint_path)
        # 移除分类头
        self.swin.head = nn.Identity()

    def load_pretrained(self, checkpoint_path):
        print(f"Loading: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model']
        # 官方 checkpoint key 与模型完全匹配
        missing, unexpected = self.swin.load_state_dict(state_dict, strict=False)
        print(f"Loaded! Missing: {len(missing)}")

    def forward(self, x):
        x = self.swin.patch_embed(x)
        if hasattr(self.swin, 'absolute_pos_embed') and self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)
        for layer in self.swin.layers:
            x = layer(x)
        x = self.swin.norm(x)  # [B, L, C]
        return x
        # return self.swin.forward_features(x)

if __name__ == "__main__":
    import os
    # 替换为你的 checkpoint 路径
    CHECKPOINT_PATH = "/root/binghao/smiles/app/img2smles/ckpt/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("Please download it from:")
        print("https://github.com/microsoft/Swin-Transformer/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth")
        exit(1)

    print("🔧 Building SwinEncoder...")
    encoder = SwinEncoder(checkpoint_path=CHECKPOINT_PATH)
    encoder.eval()  # 切换到评估模式

    print("🧠 Model structure:")
    print(f"  - Number of stages: {len(encoder.swin.layers)}")
    for i, layer in enumerate(encoder.swin.layers):
        dim = layer.blocks[0].attn.qkv.in_features
        has_downsample = hasattr(layer, 'downsample') and layer.downsample is not None
        ds_shape = layer.downsample.reduction.weight.shape if has_downsample else "None"
        print(f"  - Stage {i}: dim={dim}, downsample={ds_shape}")

    print("\n🧪 Testing forward pass...")
    x = torch.randn(2, 3, 384, 384)  # batch_size=2, 3 channels, 384x384
    with torch.no_grad():
        features = encoder(x)

    print(f"✅ Forward pass successful!")
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {features.shape}")  # 应为 [2, 144, 1536] (因为 384//32=12, 12*12=144)

    # 验证输出维度是否合理
    expected_seq_len = (384 // 32) ** 2  # 12 * 12 = 144
    expected_dim = 1536
    assert features.shape == (2, expected_seq_len, expected_dim), \
        f"Unexpected output shape! Expected ({2}, {expected_seq_len}, {expected_dim})"

    print("\n🎉 All tests passed! SwinEncoder is working correctly.")

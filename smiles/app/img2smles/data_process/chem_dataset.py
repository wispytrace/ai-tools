# data/chem_dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os
from tokenizers import Tokenizer
import torchvision
import torchvision.transforms as T
import re
import torch
import random
import torch
import torchvision.transforms as T

class MobilePhotoAugmentor:
    def __init__(self, magnitude=0.3):
        self.m = magnitude

    def get_transforms(self):
        if self.m <= 0:
            return T.Compose([T.ToTensor()])

        aug_blocks = [
            # 1. 几何畸变：全角度旋转 + 动态缩放 + 平移
            T.RandomApply([
                T.RandomChoice([
                    T.RandomAffine(
                        degrees=180,              # 全方向旋转
                        # translate 表示平移，比例随 magnitude 调整
                        translate=(0.1 * self.m, 0.1 * self.m), 
                        # scale 是缩放核心：(0.7, 1.2) 代表缩小到 70% 或放大到 120%
                        # 缩小能让模型学会处理细小特征，放大能增强局部细节
                        scale=(1.0 - 0.3 * self.m, 1.0 + 0.2 * self.m), 
                        fill=255                  # 白色背景填充
                    ),
                    T.RandomPerspective(distortion_scale=0.3 * self.m, p=1.0, fill=255),
                ])
            ], p=0.8 * self.m),

            # 2. 局部缩放/裁剪模拟：模拟拍摄距离远近
            # 随机裁剪掉边缘，再重新填充回原尺寸，变相实现了“放大”
            T.RandomApply([
                T.RandomResizedCrop(
                    size=(224, 224), # 这里建议填你模型输入的实际尺寸
                    scale=(0.8, 1.0), 
                    ratio=(0.95, 1.05),
                    interpolation=T.InterpolationMode.BILINEAR
                )
            ], p=0.3 * self.m),

            # 3. 光学损伤
            T.RandomApply([
                T.RandomChoice([
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4 + 0.8 * self.m)),
                    T.RandomAdjustSharpness(sharpness_factor=1.5 * self.m, p=1.0),
                ])
            ], p=0.4 * self.m),

            # 4. 复杂光影
            T.RandomApply([
                T.ColorJitter(brightness=0.3 * self.m, contrast=0.3 * self.m),
            ], p=0.5 * self.m),

            # 5. 环境干扰
            T.RandomApply([
                T.Pad(padding=int(30 * self.m), fill=255),
            ], p=0.3 * self.m),
        ]

        return T.Compose([
            T.RandomOrder(aug_blocks), 
            T.ToTensor(),
            # 6. 传感器底噪
            T.RandomApply([
                T.Lambda(lambda x: (x + torch.randn_like(x) * (0.02 * self.m)).clamp(0, 1))
            ], p=0.2 * self.m),
        ])
    
class ChemDataset(Dataset):
    def __init__(self, img_dir, smiles_file, tokenizer, max_length=256, max_samples=100000, is_aug=False):
        """
        Args:
            img_dir: 图像目录
            smiles_file: 包含 SELFIES/SMILES 的文件
            tokenizer: 分词器
            max_length: 序列最大长度
            max_samples: 最终需要保留的最大随机样本数量
        """
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.is_aug = is_aug

        # 1. 一次性读取所有行并记录原始行号（行号用于对应文件名）
        with open(smiles_file, "r") as f:
            # 使用 enumerate 记录原始索引 (idx, content)
            all_data = [(idx, line.strip()) for idx, line in enumerate(f)]

        # 2. 随机打乱数据顺序，以实现随机抽取
        random.seed(42)  # 固定随机种子，保证实验可复现
        random.shuffle(all_data)

        self.base_transform = [
            T.Resize((384, 384)),
        ]
        # 归一化（必须放在 ToTensor 之后）
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # 获取增强流水线
        self.augmentor = MobilePhotoAugmentor(magnitude=0.3).get_transforms()

        self.smiles_list = []
        self.valid_indices = []

        print(f"🔍 开始从 {img_dir}文件夹下 {len(all_data)} 条原始数据中随机筛选，目标保留最大数量: {max_samples}")
        
        kept_count = 0
        for original_idx, smiles in all_data:
            # 达到目标数量则停止
            if kept_count >= max_samples:
                break

            if not smiles:
                continue

            # 3. 校验图像是否存在 (注意：文件名可能对应原始行号 original_idx + 1)
            img_name = f"{original_idx + 1}.png"
            img_path = os.path.join(self.img_dir, img_name)
            
            if not os.path.exists(img_path):
                # 如果缺失样本太多，频繁打印会很慢，可以考虑降低打印频率
                continue

            # 4. 编码测试与长度检查
            try:
                # 假设类中存在 encode 方法
                token_ids = self.encode(smiles) 
                if len(token_ids) > self.max_length - 2:
                    continue
                
                self.smiles_list.append(smiles)
                self.valid_indices.append(original_idx)
                kept_count += 1
                
                # 每 10000 条打印一次进度
                if kept_count % 10000 == 0:
                    print(f"⏳ 已加载 {kept_count} 条有效样本...")
                    
            except Exception:
                continue

        print(f"✅ 数据集加载完成！最终随机保留 {len(self.smiles_list)} 条有效数据")
        print(f"当前数据增强开启状态: {is_aug}")

    def __len__(self):
        return len(self.smiles_list)

    # def encode(self, smiles):
    #     """
    #     编码 SMILES 为 token IDs 列表，添加 <sos> 和 <eos>
    #     """
    #     char_list = list(smiles)  # 按字符拆分
    #     tokens = []
    #     for char in char_list:
    #         id_ = self.tokenizer.token_to_id(char)
    #         if id_ is None:
    #             id_ = self.tokenizer.token_to_id("<unk>")
    #         tokens.append(id_)

    #     return tokens

    def encode(self, text):
        vocab = self.tokenizer.get_vocab()
        unk_id = vocab.get("<unk>", 3)
        segments = [s[0] if s[0] else s[1] for s in re.findall(r"(\[[^\]]+\])|(.)", text)]
        final_ids = []
        for seg in segments:
            if seg in vocab:
                final_ids.append(vocab[seg])
            elif seg.startswith("[") and seg.endswith("]"):
                for char in seg:
                    final_ids.append(vocab.get(char, unk_id))
            else:
                final_ids.append(unk_id)
        return final_ids

    def __getitem__(self, item):
        # 注意：item 是处理后的索引（从 0 到 len-1）
        smiles = self.smiles_list[item]
        raw_idx = self.valid_indices[item]  # 原始文件行号或图片 ID

        img_path = os.path.join(self.img_dir, f"{raw_idx+1}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"未找到图像: {img_path}")
        
        img = Image.open(img_path).convert("RGB")

        if self.is_aug:
            image = self.apply_train_transforms(img)
        else:
            image = T.Compose([
                T.Resize((384, 384)),
                T.ToTensor(),
                self.normalize])(img)

        try:
            token_ids = self.encode(smiles)
            if len(token_ids) > self.max_length - 2:
                raise ValueError(f"序列太长 ({len(token_ids)})，已超出 max_length={self.max_length}")

            sos_id = self.tokenizer.token_to_id("<sos>")
            eos_id = self.tokenizer.token_to_id("<eos>")
            tokens = [sos_id] + token_ids + [eos_id]

        except Exception as e:
            print(f"❌ Tokenizer 失败: {smiles}, error: {e}")
            tokens = [sos_id, eos_id]

        return image, tokens

    def apply_train_transforms(self, img):

        aug_logic = MobilePhotoAugmentor(magnitude=0.35).get_transforms()
        img_tensor = aug_logic(img)
        final_process = T.Compose([
            T.Resize((384, 384)), # 确保尺寸统一
            self.normalize         # 确保分布统一
        ])
        
        return final_process(img_tensor)
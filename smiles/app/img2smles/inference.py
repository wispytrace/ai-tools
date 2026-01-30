import torch
import os
import time
from tokenizers import Tokenizer
from PIL import Image
from torchvision import transforms
import selfies as sf
import random
import mimetypes
from pathlib import Path
import requests
import shutil

# --- 导入自定义模块 ---
from network.encoder import SwinEncoder
from network.decoder import TransformerDecoder
from network.img2smiles_model import Img2SMILESModel

random.seed(888)

class SmilesPredictor:
    def __init__(self, ckpt_path, tokenizer_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing Predictor on: {self.device}")

        # 1. 加载 Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.sos_id = self.tokenizer.token_to_id("<sos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        vocab_size = self.tokenizer.get_vocab_size()

        # 2. 构建模型结构
        print("\n🔍 Building inference model structure...")
        encoder = SwinEncoder()
        decoder = TransformerDecoder(vocab_size=vocab_size)
        decoder.embedding.padding_idx = self.pad_id
        # 请确保这里的 enc_dim 和 dec_dim 与训练时一致
        self.model = Img2SMILESModel(encoder, decoder).to(self.device)

        # 3. 验证并加载参数
        self._verify_and_load(ckpt_path)
        
        self.model.eval()
        
        # 4. 预处理定义
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _verify_and_load(self, ckpt_path):
        """对比 Checkpoint 和当前模型的参数差异并加载"""
        print(f"\n🔍 Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # 提取 state_dict
        ckpt_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        ckpt_keys = set(ckpt_state_dict.keys())
        
        # 检查是否包含 compile 前缀
        has_compile_prefix = any(k.startswith('_orig_mod.') for k in ckpt_keys)
        if has_compile_prefix:
            print("💡 Detected 'torch.compile' prefix in checkpoint, stripping for comparison...")
            ckpt_state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt_state_dict.items()}
            ckpt_keys = set(ckpt_state_dict.keys())

        # 获取当前模型参数
        model_state_dict = self.model.state_dict()
        model_keys = set(model_state_dict.keys())

        # --- 对比逻辑 ---
        only_in_ckpt = ckpt_keys - model_keys
        only_in_model = model_keys - ckpt_keys
        common_keys = ckpt_keys & model_keys

        print(f"📊 Comparison Result:")
        print(f"   - Common keys:             {len(common_keys)}")
        print(f"   - Keys only in checkpoint: {len(only_in_ckpt)}")
        print(f"   - Keys only in model:      {len(only_in_model)}")

        if only_in_ckpt:
            print("\n🔑 Keys missing in model (from checkpoint):")
            for k in sorted(only_in_ckpt)[:10]: print(f"     - {k}")
            if len(only_in_ckpt) > 10: print(f"     ... and {len(only_in_ckpt)-10} more")

        if only_in_model:
            print("\n🔑 Keys missing in checkpoint (from model):")
            for k in sorted(only_in_model)[:10]: print(f"     - {k}")
            if len(only_in_model) > 10: print(f"     ... and {len(only_in_model)-10} more")

        if not only_in_ckpt and not only_in_model:
            print("\n🎉 PERFECT MATCH! Model structure is consistent.")
        else:
            print("\n⚠️  MISMATCH DETECTED! Loading with strict=False.")

        # 执行加载
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        print("✅ Weights loaded into model.")

    @torch.no_grad()
    def predict_confidence(self, image_path, max_len=256):
        """预测单张图片"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        # 调用模型内置的 generate 逻辑
        output_ids, confidence = self.model.generate_with_conf(
            img_tensor, 
            sos_id=self.sos_id, 
            eos_id=self.eos_id, 
            max_len=max_len
        )
        
        # 处理返回维度 [1, seq_len] 或 [seq_len]
        if output_ids.dim() > 1:
            output_ids = output_ids[0]
        
        decode_chars = [self.tokenizer.decode([tid]) for tid in output_ids.cpu().tolist()]
        decoded = ''.join(decode_chars).replace('<sos>', '').replace('<eos>', '').strip()
        
        # decoded = self.tokenizer.decode(output_ids.cpu().tolist())
        elapsed = time.time() - start_time
        
        return decoded, elapsed, confidence

    @torch.no_grad()
    def predict(self, image_path, max_len=256):
        """预测单张图片"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        # 调用模型内置的 generate 逻辑
        output_ids = self.model.generate(
            img_tensor, 
            sos_id=self.sos_id, 
            eos_id=self.eos_id, 
            max_len=max_len
        )
        
        # 处理返回维度 [1, seq_len] 或 [seq_len]
        if output_ids.dim() > 1:
            output_ids = output_ids[0]
        
        decode_chars = [self.tokenizer.decode([tid]) for tid in output_ids.cpu().tolist()]
        decoded = ''.join(decode_chars).replace('<sos>', '').replace('<eos>', '').strip()
        
        # decoded = self.tokenizer.decode(output_ids.cpu().tolist())
        elapsed = time.time() - start_time
        
        return decoded, elapsed

    @torch.no_grad()
    def predict_batch(self, image_paths, batch_size=8, max_len=256):
        """
        支持单张或多张图片路径的批量预测
        image_paths: str 或 List[str]
        """
        # 1. 统一输入格式为列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        all_results = []
        all_times = []
        
        # 2. 分批次处理，防止显存溢出
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            start_time = time.time()
            
            # 加载并转换图片
            batch_tensors = []
            for path in batch_paths:
                if os.path.exists(path):
                    img = Image.open(path).convert('RGB')
                    batch_tensors.append(self.transform(img))
            
            # 堆叠为 [B, C, H, W]
            imgs_tensor = torch.stack(batch_tensors).to(self.device)
            
            # 3. 调用批量生成逻辑 (generate 或 generate_beam)
            # 注意：此处确保你的 model.generate 支持 batch 输入
            output_ids_batch = self.model.generate(
                imgs_tensor, 
                sos_id=self.sos_id, 
                eos_id=self.eos_id, 
                max_len=max_len,
            )
            
            elapsed = (time.time() - start_time) / len(batch_paths)

            # 4. 后处理解码
            for output_ids in output_ids_batch:
                # 过滤特殊字符并解码
                ids_list = output_ids.cpu().tolist()
                decode_chars = [
                    self.tokenizer.decode([tid]) for tid in ids_list 
                    if tid not in [self.sos_id, self.eos_id, 0] # 0 通常是 padding
                ]
                decoded = ''.join(decode_chars).strip()
                all_results.append(decoded)
                all_times.append(elapsed)
        
        # 如果输入是单张，返回单个结果；否则返回列表
        if len(all_results) == 1:
            return all_results[0], all_times[0]
        return all_results, all_times

def similarity_ratio(str1, str2):
    import Levenshtein # 需要安装 python-Levenshtein
    
    # 计算编辑距离
    distance = Levenshtein.distance(str1, str2)
    # 归一化为 0 到 1 之间的得分
    max_len = max(len(str1), len(str2))
    return 1 - (distance / max_len)

def load_compare_dataset(dataset_path):
    label_path = os.path.join(dataset_path, "label.txt")
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    dataset_index = random.sample(range(int(len(lines)/2)), min(1000, len(lines)))
    dataset = []
    for idx in dataset_index:
        seflies_str = lines[idx].strip()
        img_path = os.path.join(dataset_path, str(idx+1) + ".png")
        dataset.append((img_path, seflies_str))
    return dataset


def convert_image_to_smiles(image_path: str):
    url = "http://192.168.1.239:30869/ocr_api/img_to_smiles"
    headers = {"accept": "application/json"}
    
    # 推测 MIME 类型
    mime_type, _ = mimetypes.guess_type(image_path)
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        print(f"[SMILES] Unsupported image type: {ext}")
        return None
    file_name = Path(image_path).name
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (file_name, f, mime_type)}
            response = requests.post(url, headers=headers, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # 假设返回格式: {"smiles": "C1=CC=..."} 或带 confidence 的对象
            if isinstance(result, dict) and "smiles" in result:
                return result["smiles"]
            else:
                print(f"[SMILES] Invalid response format: {result}")
                return ''
        else:
            print(f"[SMILES] API Error {response.status_code}")
            return ''
    except Exception as e:
        print(f"[SMILES] Request failed: {str(e)}")
        return ''

def evaluate_model_on_dataset(predictor, dataset):
    total = len(dataset)
    score = 0
    total_time = 0.0
    correct = 0

    for img_path, true_selfies in dataset:
        pred_selfies, elapsed = predictor.predict(img_path)
        total_time += elapsed
        score += similarity_ratio(pred_selfies, true_selfies)
        if pred_selfies == true_selfies:
            correct += 1
        try:
            pred_smiles = sf.decoder(pred_selfies)
        except Exception:
            pred_smiles = ""
        true_smiles = sf.decoder(true_selfies)
        print(f"Image: {img_path} | Predicted: {pred_smiles} | True: {true_smiles} | Time: {elapsed:.2f}s")

    accuracy = correct / total
    avg_time = total_time / total
    score = score / total
    return accuracy, avg_time, score


# def evaluate_model_on_dataset_batch(predictor, dataset, batch_size=96):
#     """
#     针对批量预测优化的评估函数
#     dataset: List of (img_path, true_selfies)
#     """
#     total = len(dataset)
#     total_correct = 0
#     total_similarity = 0.0
#     total_time = 0.0
    
#     # 将数据集按 batch_size 切分
#     for i in range(0, total, batch_size):
#         batch_data = dataset[i : i + batch_size]
#         img_paths = [item[0] for item in batch_data]
#         true_selfies_list = [item[1] for item in batch_data]
        
#         # 1. 调用批量预测 (返回列表)
#         # 这里的 predictor.predict 内部应该已经实现了 imgs_tensor = torch.stack(...)
#         pred_selfies_list, batch_elapsed_list = predictor.predict_batch(img_paths, batch_size=batch_size)
        
#         # 处理单张返回为列表的兼容性
#         if isinstance(pred_selfies_list, str):
#             pred_selfies_list = [pred_selfies_list]
#             batch_elapsed_list = [batch_elapsed_list]

#         # 2. 批量处理评估逻辑
#         for j in range(len(pred_selfies_list)):
#             pred_selfies = pred_selfies_list[j]
#             true_selfies = true_selfies_list[j]
#             elapsed = batch_elapsed_list[j]
            
#             total_time += elapsed
            
#             # 计算相似度 (SELFIES 相似度)
#             sim = similarity_ratio(pred_selfies, true_selfies)
#             total_similarity += sim
            
#             # 计算完全匹配
#             if pred_selfies == true_selfies:
#                 total_correct += 1
            
#             # 解码为 SMILES 用于打印显示
#             try:
#                 pred_smiles = sf.decoder(pred_selfies)
#             except Exception:
#                 pred_smiles = "INVALID_SELFIES"
            
#             try:
#                 true_smiles = sf.decoder(true_selfies)
#             except Exception:
#                 true_smiles = "INVALID_TRUE_SELFIES"
                
#             print(f"[{i+j+1}/{total}] Time: {elapsed:.2f}s | Sim: {sim:.2f} | Pred: {pred_smiles}")

#     # 3. 计算最终指标
#     avg_accuracy = total_correct / total
#     avg_time = total_time / total
#     avg_score = total_similarity / total
    
#     print("-" * 30)
#     print(f"Final Report: Accuracy: {avg_accuracy:.4f} | Avg Time: {avg_time:.4f}s | Avg Sim: {avg_score:.4f}")
    
#     return avg_accuracy, avg_time, avg_score

def evaluate_model_on_dataset_batch(predictor, dataset, batch_size=96, error_save_dir="eval_errors"):
    """
    针对批量预测优化的评估函数，并自动保存识别失败的图片
    """
    # 创建存放错误图片的文件夹
    if not os.path.exists(error_save_dir):
        os.makedirs(error_save_dir)
        print(f"📂 Created error directory: {error_save_dir}")
    else:
        # 清空旧的错误图片（可选，视你需求而定）
        shutil.rmtree(error_save_dir)
        os.makedirs(error_save_dir)

    total = len(dataset)
    total_correct = 0
    total_similarity = 0.0
    total_time = 0.0
    
    for i in range(0, total, batch_size):
        batch_data = dataset[i : i + batch_size]
        img_paths = [item[0] for item in batch_data]
        true_selfies_list = [item[1] for item in batch_data]
        
        pred_selfies_list, batch_elapsed_list = predictor.predict_batch(img_paths, batch_size=batch_size)
        
        if isinstance(pred_selfies_list, str):
            pred_selfies_list = [pred_selfies_list]
            batch_elapsed_list = [batch_elapsed_list]

        for j in range(len(pred_selfies_list)):
            pred_selfies = pred_selfies_list[j]
            true_selfies = true_selfies_list[j]
            img_path = img_paths[j]
            elapsed = batch_elapsed_list[j]
            
            total_time += elapsed
            sim = similarity_ratio(pred_selfies, true_selfies)
            total_similarity += sim
            
            is_correct = (pred_selfies == true_selfies)
            
            if is_correct:
                total_correct += 1
            else:
                # --- 新增：保存识别失败的图片 ---
                # 为了防止文件名过长或非法，我们截取部分 selfies 字符串
                safe_true_val = true_selfies[:30].replace("[", "").replace("]", "")
                error_filename = f"err_{i+j+1}_sim_{sim:.2f}_{safe_true_val}.png"
                dst_path = os.path.join(error_save_dir, error_filename)
                
                try:
                    shutil.copy(img_path, dst_path)
                except Exception as e:
                    print(f"⚠️ Failed to copy error image: {e}")
                # ------------------------------

            try:
                pred_smiles = sf.decoder(pred_selfies)
            except Exception:
                pred_smiles = "INVALID_SELFIES"
            
            # 只有失败时才多打印一些信息辅助查看
            status = "✅" if is_correct else "❌"
            print(f"[{i+j+1}/{total}] {status} Time: {elapsed:.2f}s | Sim: {sim:.2f} | Pred: {pred_smiles}")

    avg_accuracy = total_correct / total
    avg_time = total_time / total
    avg_score = total_similarity / total
    
    print("-" * 30)
    print(f"Final Report: Accuracy: {avg_accuracy:.4f} | Avg Time: {avg_time:.4f}s | Avg Sim: {avg_score:.4f}")
    print(f"📁 Misclassified images saved to: {error_save_dir}")
    
    return avg_accuracy, avg_time, avg_score


def evaluate_model_on_decimer(dataset):
    total = len(dataset)
    correct = 0
    total_time = 0.0
    score = 0

    for img_path, true_selfies in dataset:
        start = time.time()
        smiles_str = convert_image_to_smiles(img_path)
        pred_selfies = sf.encoder(smiles_str)
        elapsed = time.time() - start
        score += similarity_ratio(pred_selfies, true_selfies)
        if pred_selfies == true_selfies:
            correct += 1
        total_time += elapsed
        print(f"Image: {img_path} | Predicted: {pred_selfies} | True: {true_selfies} | Time: {elapsed:.2f}s")
    
    accuracy = correct / total
    avg_time = total_time / total
    score = score / total
    return accuracy, avg_time, score

        

# --- 执行示例 ---
if __name__ == "__main__":
    # 配置你的路径
    CKPT = "/root/binghao/smiles/app/img2smles/checkpoints-0112-new/best_model.pth"
    TOKENIZER = "tokenizer-selfies-ultimate.json"
    IMAGE = "/root/binghao/smiles/app/img2smles/dataset/images_24/388.png"

    predictor = SmilesPredictor(CKPT, TOKENIZER)

    # print(sf.decoder(predictor.predict("/root/binghao/smiles/app/img2smles/企业微信截图_17670611721793.png")[0]))
    
    compare_dataset = load_compare_dataset("/root/binghao/smiles/app/img2smles/dataset/images_24_except")
    # accuracy, avg_time, score = evaluate_model_on_dataset(predictor, compare_dataset)
    accuracy, avg_time, score = evaluate_model_on_dataset_batch(predictor, compare_dataset)
    # accuracy_decimer, avg_time_decimer, score_decimer = evaluate_model_on_decimer(compare_dataset)
    # print(f"\n✨ Custom Model - Accuracy: {accuracy*100:.2f}%, Avg Time: {avg_time:.2f}s, Score: {score:.4f}")
    # print(f"\n✨ Decimer API - Accuracy: {accuracy_decimer*100:.2f}%, Avg Time: {avg_time_decimer:.2f}, Score: {score_decimer:.4f}")
    # if os.path.exists(IMAGE):
    #     result, speed = predictor.predict(IMAGE)
    #     print(f"\n✨ Prediction: {result}")
    #     print(f"⏱️  Time taken: {speed:.2f}s")
import json
import os
import requests
from pathlib import Path
from tqdm import tqdm
# ================== 配置区 ==================
JSON_PATH = "output_images/result.json"          # 你的测试集 JSON 路径
# API_URL = "http://192.168.1.239:6789/detect_to_smiles"  # 你的 FastAPI 接口
API_URL = "http://192.168.1.239:30869/ocr_api/img_to_smiles"  # 你的 FastAPI 接口
TIMEOUT = 30  # 请求超时（秒）
USE_STANDARDIZED_COMPARE = True      # 是否使用标准化 SMILES 比较（推荐开启）
OUTPUT_FAILURES = "failures.json"   # 失败样本输出文件
# ==========================================

def standardize_smiles(smiles: str) -> str:
    """
    尝试标准化 SMILES（可选，需安装 rdkit）
    如果未安装 rdkit，则直接返回原字符串
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except ImportError:
        return smiles

def call_api(image_path: str) -> tuple[str, str]:
    """
    调用 FastAPI 接口，返回 (预测smiles, 错误信息)
    成功时 error 为空，失败时 smiles 为空
    """
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            response = requests.post(API_URL, files=files, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            pred_smiles = result.get("smiles", "").strip()
            return pred_smiles, ""
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API 调用失败 {image_path}: {error_msg}")
        return "", error_msg

def main():
    # 1. 加载测试数据
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"🧪 加载 {len(test_cases)} 个测试样本...")
    correct = 0
    total_valid = 0  # 实际参与评估的样本数（排除图像不存在的）
    failures = []    # 记录所有失败项

    # 2. 遍历每个样本
    for item in tqdm(test_cases, desc="Testing"):
        image_path = item["path"]
        true_smiles = item["smiles"].strip()

        # 检查图像是否存在
        if not Path(image_path).exists():
            failures.append({
                "image_path": image_path,
                "true_smiles": true_smiles,
                "pred_smiles": "",
                "error": "Image file not found",
                "failure_type": "file_missing"
            })
            continue

        total_valid += 1

        # 调用 API
        pred_smiles, api_error = call_api(image_path)

        # 情况1: API 调用失败
        if api_error:
            failures.append({
                "image_path": image_path,
                "true_smiles": true_smiles,
                "pred_smiles": "",
                "error": api_error,
                "failure_type": "api_error"
            })
            continue

        # 情况2: API 返回空 SMILES
        if not pred_smiles:
            failures.append({
                "image_path": image_path,
                "true_smiles": true_smiles,
                "pred_smiles": pred_smiles,
                "error": "Empty SMILES returned by API",
                "failure_type": "empty_prediction"
            })

        # 比较 SMILES
        if USE_STANDARDIZED_COMPARE:
            true_std = standardize_smiles(true_smiles)
            pred_std = standardize_smiles(pred_smiles)
            is_correct = (true_std == pred_std)
        else:
            is_correct = (true_smiles == pred_smiles)

        if is_correct:
            correct += 1
        else:
            # 情况3: SMILES 不匹配
            failures.append({
                "image_path": image_path,
                "true_smiles": true_smiles,
                "pred_smiles": pred_smiles,
                "error": "SMILES mismatch",
                "failure_type": "incorrect_prediction"
            })
            print(f"\n🛑 不匹配:\n  真实: {true_smiles}\n  预测: {pred_smiles}")

    # 3. 输出结果
    accuracy = correct / total_valid if total_valid > 0 else 0
    print("\n" + "="*50)
    print(f"✅ 有效样本数: {total_valid}")
    print(f"✅ 正确数:     {correct}")
    print(f"🎯 准确率:     {accuracy:.2%}")
    print(f"💥 失败总数:   {len(failures)}")
    print("="*50)

    # 4. 保存失败记录
    with open(OUTPUT_FAILURES, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    print(f"📄 失败样本已保存至: {OUTPUT_FAILURES}")

if __name__ == "__main__":
    main()

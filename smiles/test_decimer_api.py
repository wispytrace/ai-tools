import requests

def decimer_predict(image_path: str) -> str:
    """
    调用 DECIMER.ai API 识别化学结构图
    
    Args:
        image_path (str): 本地图像路径（PNG/JPG/SVG）
    
    Returns:
        str: 预测的 SMILES 字符串
    """
    url = "https://decimer.ai/api/predict/"
    
    with open(image_path, "rb") as f:
        files = {"file": (f.name, f, "image/png")}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("smiles", "").strip()
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")

# 使用示例
if __name__ == "__main__":
    smiles = decimer_predict("/root/binghao/smiles/failed_images/19.png")
    print("Predicted SMILES:", smiles)

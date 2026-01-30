import os
import io
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import selfies as sf
from PIL import Image
import cairosvg  # 用于将 SVG 转换为 PNG
import cv2
import numpy as np

from inference import SmilesPredictor 

app = FastAPI(title="Img2SMILES API", description="支持 PNG/JPG/SVG 的化学分子识别")

predictor = None

# 配置
CKPT_PATH = "/root/binghao/smiles/app/img2smles/checkpoints-0112-new/best_model.pth "
TOKENIZER_PATH = "tokenizer-selfies-ultimate.json"
DEVICE = "cuda"

@app.on_event("startup")
async def load_model():
    global predictor
    print("⏳ Loading model...")
    predictor = SmilesPredictor(CKPT_PATH, TOKENIZER_PATH, device=DEVICE)
    print("✅ Model loaded.")

# def preprocess_image_to_binary(image_path: str):
#     """
#     读取图片，将其转换为灰度图，并应用二值化处理。
#     """
#     # 1. 使用 OpenCV 读取
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("Could not read image for binarization")

#     # 2. 转为灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 3. 二值化处理
#     # 使用 Otsu 自适应阈值，THRESH_BINARY 代表白底黑字（如果需要反色可用 THRESH_BINARY_INV）
#     # 通常化学分子图是白底黑字，我们确保背景是纯白 (255)，线条是纯黑 (0)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # 如果图片由于某种原因反色了（黑底白字），可以强制检测并修正
#     # 假设背景应该是白色的（255），统计边缘像素
#     edge_pixels = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
#     if np.mean(edge_pixels) < 127: # 说明背景大多是黑色
#         binary = cv2.bitwise_not(binary)

#     # 4. 写回原路径或覆盖原图
#     cv2.imwrite(image_path, binary)

@app.post("/predict")
async def predict_smiles(file: UploadFile = File(...)):
    # 1. 验证类型
    allowed_types = ["image/jpeg", "image/png", "image/svg+xml"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and SVG are supported.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, file.filename)
        process_path = os.path.join(tmp_dir, "processed_final.png")

        # 保存原始文件
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # 2. 格式转换：SVG -> PNG
            if "svg" in file.content_type or file.filename.lower().endswith(".svg"):
                cairosvg.svg2png(url=input_path, write_to=process_path, output_width=512, output_height=512)
            else:
                # JPG/PNG 复制到 process_path 准备处理
                shutil.copy(input_path, process_path)

            # 3. 【新增】二值化处理
            # 经过这一步，图片将变成纯粹的黑白两色
            # preprocess_image_to_binary(process_path)

            # 4. 模型推理
            # 注意：如果你的 predictor.predict_confidence 内部还会做一次 Image.open
            # 它读到的将是已经二值化后的黑白图
            pred_selfies, elapsed, confidence = predictor.predict_confidence(process_path)

            # 5. 解码
            try:
                pred_smiles = sf.decoder(pred_selfies)
            except:
                pred_smiles = "SMILES Conversion Error"

            return {
                "success": True,
                "smiles": pred_smiles,
                "selfies": pred_selfies,
                "confidence": confidence.item(),
                "inference_time_sec": round(elapsed, 4),
                "is_binary": True,
                "format": file.content_type
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
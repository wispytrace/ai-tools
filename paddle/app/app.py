from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from paddleocr import TextRecognition
import os
from datetime import datetime
import numpy as np
import cv2

# 初始化 FastAPI 应用
app = FastAPI(title="PaddleOCR Text Recognition API (Pipeline Mode)", version="1.0")

# 初始化模型（启动时加载一次）
model = TextRecognition()

# 创建输出目录
OUTPUT_VIS_DIR = "./output"
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

# 根路径欢迎页
@app.get("/")
async def home():
    return {"message": "PaddleOCR Text Recognition API is running. Use POST /recognize to upload images."}


class OCRResult:
    def __init__(self, text: str, score: float, image=None):
        self.rec_text = text
        self.rec_score = score
        self.image = image  # NumPy array

    def print(self):
        print(f"Text: {self.rec_text}, Score: {self.rec_score}")

    def save_to_img(self, save_path: str):
        if self.image is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{timestamp}_rec_result.jpg"
        path = os.path.join(save_path, fname)
        cv2.imwrite(path, self.image)
        return path

    def save_to_json(self, save_path: str):
        import json
        result = {
            "text": self.rec_text,
            "score": self.rec_score,
            "timestamp": datetime.now().isoformat()
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{timestamp}_result.json"
        path = os.path.join(save_path, fname)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return path


@app.post("/recognize")
async def recognize_images(images: list[UploadFile] = File(...)):
    """
    接收一个或多个裁剪好的文字图像，执行纯文本识别。
    返回识别文本和置信度，并保存可视化图与 JSON。
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    valid_images = []
    file_names = []

    # 1. 验证并读取图像
    for file in images:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Invalid image type for {file.filename}")

        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decode failed")
            valid_images.append(img)
            file_names.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image {file.filename}: {str(e)}")

    # 2. 执行识别
    try:
        outputs = model.predict(input=valid_images)  # 支持批量输入
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    # 3. 构造响应 & 保存结果
    results = []
    saved_paths = []

    for i, res in enumerate(outputs):
        text = res.get('rec_text', '')
        score = res.get('rec_score', 0.0)

        # 包装为类对象（模拟原生 output 对象行为）
        ocr_res = OCRResult(text=text, score=score, image=valid_images[i])

        # 打印日志
        ocr_res.print()

        # 保存可视化图像
        vis_path = ocr_res.save_to_img(save_path=OUTPUT_VIS_DIR)
        if vis_path:
            saved_paths.append({"image": vis_path})

        # 保存 JSON
        json_path = ocr_res.save_to_json(save_path=OUTPUT_VIS_DIR)
        if json_path:
            saved_paths.append({"json": json_path})

        # 添加到返回结果
        results.append({
            "filename": file_names[i],
            "text": text,
            "score": float(score)
        })

    # 4. 返回统一 JSON 响应
    return JSONResponse(content={
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "results": results,
        "saved_files": saved_paths
    })

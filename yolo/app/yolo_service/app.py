from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
from datetime import datetime
import cv2
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import numpy as np
import requests
import mimetypes
from pathlib import Path

# 初始化 FastAPI 应用
app = FastAPI(title="YOLO Object Detection API", version="1.0")

# 配置
MODEL_PATH = "last.pt"
UPLOAD_DIR = "uploaded"
VIS_DIR = "uploaded_vis"
CROPS_DIR = "crops"  # 新增：用于保存检测框裁剪图

BASE_URL = "http://192.168.1.239:6789"  # ← 改成你的实际 IP 或域名

# 创建目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# 加载模型（启动时加载一次）
model = YOLO(MODEL_PATH)


def get_class_names() -> Dict[int, str]:
    """返回类别 ID 到名称的映射"""
    return model.model.names


def extract_detections(result) -> List[Dict[str, Any]]:
    """
    从 YOLOv8 的推理结果中提取检测信息。
    
    Args:
        result: YOLOv8 的单张图像推理结果（results[0]）

    Returns:
        List of detection dicts with keys: class_id, class_name, confidence, bbox
    """
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    class_names = get_class_names()

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(round(coord, 2)) for coord in boxes[i]]
        conf = float(confs[i])
        cls_id = int(classes[i])
        cls_name = class_names[cls_id]

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
    return detections


def postprocess_detection_crops(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    base_filename: str,
    save_dir: str = CROPS_DIR
) -> List[str]:
    """
    对每个检测框裁剪图像，并保存到指定目录。
    你可以在此函数中添加自定义操作（如 OCR、特征提取、上传等）。

    Args:
        image: 原始 BGR 图像 (H, W, C)
        detections: 检测结果列表
        base_filename: 基础文件名（不含扩展名）
        save_dir: 裁剪图保存目录

    Returns:
        List of saved crop file paths (relative or absolute)
    """
    saved_paths = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        # 确保坐标不越界
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue  # 跳过无效框

        crop = image[y1:y2, x1:x2]
        crop_filename = f"{base_filename}_crop_{idx}_{det['class_name']}.jpg"
        crop_path = os.path.join(save_dir, crop_filename)
        cv2.imwrite(crop_path, crop)
        saved_paths.append({"class_id": det["class_id"], "class_name": det["class_name"], "crop_image": crop_path, "confidence": det["confidence"]})
    return saved_paths


async def _process_single_image(file: UploadFile) -> Dict[str, Any]:
    """
    处理单张上传图像的核心逻辑（被单图和批量接口共用）
    """
    # 1. 保存原图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式: {ext}")

    original_filename = f"{timestamp}_original{ext}"
    vis_filename = f"{timestamp}_detected{ext}"
    base_name = f"{timestamp}"

    original_path = os.path.join(UPLOAD_DIR, original_filename)
    vis_path = os.path.join(VIS_DIR, vis_filename)

    contents = await file.read()
    with open(original_path, "wb") as f:
        f.write(contents)

    # 2. 推理
    results = model(original_path)
    result = results[0]

    # 3. 提取检测结果
    detections = extract_detections(result)

    # 4. 保存可视化图
    plotted_img = result.plot()  # BGR
    cv2.imwrite(vis_path, plotted_img)

    # 5. 【新增】对检测框进行裁剪和后处理
    original_img = cv2.imread(original_path)  # BGR
    crop_paths = postprocess_detection_crops(original_img, detections, base_name)

    return {
        "filename": file.filename,
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "visualized_image": vis_filename,
        "crop_images": crop_paths,  # ✅ 新增：返回裁剪图 URL
        "detections": detections,
        "total_objects": len(detections)
    }

async def convert_image_to_smiles(image_path: str):
    """
    将化学结构图像转换为SMILES表示
    
    Args:
        image_path: 本地图像路径
        
    Returns:
        {"smiles": "..."} 或 None（失败时）
    """
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
            print(f"[SMILES] API Error {response.status_code}: {error}")
            return ''
    except Exception as e:
        print(f"[SMILES] Request failed: {str(e)}")
        return ''


@app.post("/detect_to_smiles", summary="上传单张图片并进行目标检测，返回SMILES")
async def detect_image(file: UploadFile = File(...)):
    # 初始化默认响应
    response_data = {
        "success": False,
        "smiles": "",
        "message": ""
    }

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必须是图片格式")

        result = await _process_single_image(file)
        crop_images = result.get("crop_images", [])
        
        max_confidence = 0.0
        best_crop_path = ""

        # 寻找 class_id == 0 且置信度最高的裁剪图
        for crop in crop_images:
            if crop.get("class_id") == 0 and crop.get("confidence", 0) > max_confidence:
                max_confidence = crop["confidence"]
                best_crop_path = crop.get("crop_image", "")

        smiles = ""
        if best_crop_path and os.path.exists(best_crop_path):
            smiles = await convert_image_to_smiles(best_crop_path)

        if smiles:  # 非空字符串视为成功
            response_data.update({
                "success": True,
                "smiles": smiles
            })
        else:
            response_data["message"] = "未检测到有效分子结构或无法转换为SMILES"

    except HTTPException as he:
        # 重新抛出 HTTP 异常（如 400）
        raise he
    except Exception as e:
        response_data["message"] = f"处理过程中发生错误: {str(e)}"

    return JSONResponse(content=response_data)


@app.post("/detect", summary="上传单张图片并进行目标检测")
async def detect_image_to_smiles(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")

    try:
        result = await _process_single_image(file)
        return JSONResponse(content={
            "success": True,
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/detect_batch", summary="批量上传图片并进行目标检测")
async def detect_batch_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="未接收到任何文件")

    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "非图片文件",
                "timestamp": datetime.now().isoformat()
            })
            continue

        try:
            res = await _process_single_image(file)
            results.append(res)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    return JSONResponse(content={
        "success": True,
        "processed_count": len(files),
        "results": results
    })


# ✅ 提供可视化图像访问
@app.get("/image/{filename}", summary="获取检测后的可视化图像")
async def get_visualized_image(filename: str):
    file_path = os.path.join(VIS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="图像未找到")
    return FileResponse(file_path, media_type="image/jpeg")


# ✅ 新增：提供裁剪图像访问
@app.get("/crop/{filename}", summary="获取检测框裁剪后的子图")
async def get_crop_image(filename: str):
    file_path = os.path.join(CROPS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="裁剪图像未找到")
    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/", summary="API 主页")
async def home():
    return {
        "message": "YOLO Detection API is running",
        "docs": "/docs",
        "endpoints": {
            "single": "/detect (POST)",
            "batch": "/detect_batch (POST)",
            "view images": "/image/{filename}",
            "view crops": "/crop/{filename}"
        }
    }

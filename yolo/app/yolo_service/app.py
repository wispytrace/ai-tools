from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
import os
from datetime import datetime
import cv2
from ultralytics import YOLO
import json
from typing import List

# 初始化 FastAPI 应用
app = FastAPI(title="YOLO Object Detection API", version="1.0")

# 加载 YOLO 模型（启动时加载一次）
MODEL_PATH = "last.pt"
model = YOLO(MODEL_PATH)

# 创建目录
UPLOAD_DIR = "uploaded"
VIS_DIR = "uploaded_vis"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# 基础 URL（可根据部署环境修改）
BASE_URL = "http://192.168.1.239:6789"  # ← 改成你的实际 IP 或域名


def get_class_names():
    """返回类别名"""
    return model.model.names


@app.post("/detect", summary="上传图片并进行目标检测")
async def detect_image(file: UploadFile = File(...)):
    # 1. 检查是否为图像文件
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")

    # 2. 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    original_filename = f"{timestamp}_original{ext}"
    vis_filename = f"{timestamp}_detected{ext}"

    original_path = os.path.join(UPLOAD_DIR, original_filename)
    vis_path = os.path.join(VIS_DIR, vis_filename)

    # 3. 保存上传的原图
    try:
        contents = await file.read()
        with open(original_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存原图失败: {str(e)}")

    # 4. 使用 YOLO 进行推理
    try:
        results = model(original_path)
        result = results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

    # 5. 解析检测结果
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

    # 6. 保存带标注的可视化图像
    try:
        plotted_img = result.plot()  # BGR to RGB
        cv2.imwrite(vis_path, plotted_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存可视化图像失败: {str(e)}")

    # 7. 构造响应（包含可访问的图片 URL）
    visualized_image_url = f"{BASE_URL}/image/{vis_filename}"  # 可直接在浏览器打开

    response = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "original_image": original_path,
        "visualized_image": visualized_image_url,  # ✅ 返回可访问链接
        "detections": detections,
        "total_objects": len(detections)
    }

    return JSONResponse(content=response)

async def detect_batch_images(files: List[UploadFile] = File(...)):
    # 1. 检查是否所有文件都是图像
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是有效的图片格式")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="未接收到任何文件")

    # 存储每张图的结果
    results = []

    for file in files:
        try:
            # 2. 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{hash(file.filename)}"
            ext = os.path.splitext(file.filename)[1] or ".jpg"
            original_filename = f"{timestamp}_original{ext}"
            vis_filename = f"{timestamp}_detected{ext}"

            original_path = os.path.join(UPLOAD_DIR, original_filename)
            vis_path = os.path.join(VIS_DIR, vis_filename)

            # 3. 保存原图
            contents = await file.read()
            with open(original_path, "wb") as f:
                f.write(contents)

            # 4. YOLO 推理
            results_model = model(original_path)
            result = results_model[0]

            # 5. 解析检测结果
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

            # 6. 保存带框的可视化图像
            plotted_img = result.plot()
            cv2.imwrite(vis_path, plotted_img)

            # 7. 构造每张图的响应
            visualized_image_url = f"{BASE_URL}/image/{vis_filename}"

            results.append({
                "filename": file.filename,
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "visualized_image": visualized_image_url,
                "detections": detections,
                "total_objects": len(detections)
            })

        except Exception as e:
            # 单个文件出错时不中断整体流程
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    # 返回整体响应
    return JSONResponse(content={
        "success": True,
        "processed_count": len(files),
        "results": results
    })

# ✅ 新增：提供可视化图像访问接口
@app.get("/image/{filename}", summary="获取检测后的可视化图像")
async def get_visualized_image(filename: str):
    file_path = os.path.join(VIS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="图像未找到")
    
    # 返回图片（浏览器可直接显示）
    return FileResponse(file_path, media_type="image/jpeg")


# ✅ 可选：主页提示
@app.get("/")
async def home():
    return {
        "message": "YOLO Detection API is running",
        "docs": "前往 /docs 查看交互式文档"
    }

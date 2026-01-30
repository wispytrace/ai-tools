from ultralytics import YOLO

# 加载你自己训练的模型权重
model = YOLO("/ultralytics/runs/detect/train/weights/best.pt")  # 替换为你的 .pt 文件路径

# 进行预测（可以是图片路径、URL、numpy 数组等）
results = model.predict(
    source="/ultralytics/359b777d73ed25301cdb44332e5270e9_compress.jpg",        # 输入源：图片/视频路径，或目录，或 'https://url/to/video.mp4'
    save=True,                # 保存结果图
    imgsz=640,                # 推理图像大小
    conf=0.1,                # 置信度阈值
    device="cpu"             # 使用 GPU (或 device=0 / device=[0,1])
)

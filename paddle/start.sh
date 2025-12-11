#!/bin/bash

# =============================
# 启动 YOLO + FastAPI 服务的脚本
# =============================

# 配置参数
IMAGE_NAME="192.168.1.101:7443/top/paddle"
CONTAINER_NAME="paddle_test"

# ----------------------------
# 步骤 1：停止并删除已有容器
# ----------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 停止并删除已有容器: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

# ----------------------------
# 步骤 2：检查镜像是否存在，不存在则构建
# ----------------------------
# if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}\$"; then
#     echo "🏗️  构建 Docker 镜像: $IMAGE_NAME"
#     docker build -t "$IMAGE_NAME" .
# else
#     echo "✅ 镜像 $IMAGE_NAME 已存在，跳过构建。"
# fi

# ----------------------------
# 步骤 3：启动容器，运行 Uvicorn
# ----------------------------
# echo "🚀 启动容器 $CONTAINER_NAME 并运行 FastAPI 服务..."

# docker run -d \
#   --name "$CONTAINER_NAME" \
#   -p 6788:8000 \
#   "$IMAGE_NAME" \
#   tail -f /dev/null

docker run -d \
  --name "$CONTAINER_NAME" \
  -p 6788:8000 \
  -v "$(pwd)"/app:/root/app \
  -w /root/app \
  "$IMAGE_NAME" \
  uvicorn app:app --host 0.0.0.0 --port 8000 

# docker run -d \
#   --name "$CONTAINER_NAME" \
#   -p 8000:8000 \
#   -v "$(pwd)"/app:/ultralytics \
#   -w /ultralytics/yolo_service \
#   "$IMAGE_NAME" \
#   tail -f /dev/null

# docker exec -it yolo bash

# cd /ultralytics/yolo_service
# 
# ----------------------------
# 步骤 4：输出提示信息
# ----------------------------
if [ $? -eq 0 ]; then
    echo "🎉 容器已成功启动！"
    echo "📄 API 文档请访问: http://localhost:6788/docs"
    echo "🖼️  测试 /detect 接口上传图片"
else
    echo "❌ 启动失败，请检查日志: docker logs $CONTAINER_NAME"
    exit 1
fi
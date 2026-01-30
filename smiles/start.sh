#!/bin/bash

# =============================
# 启动 YOLO + FastAPI 服务的脚本
# =============================

# 配置参数
IMAGE_NAME="192.168.1.101:7443/top/smiles:latest"
CONTAINER_NAME="smiles"

# ----------------------------
# 步骤 1：停止并删除已有容器
# ----------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 停止并删除已有容器: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

echo "🚀 启动容器 $CONTAINER_NAME 并运行 FastAPI 服务..."



docker run -d \
  --name "$CONTAINER_NAME" \
  -p 9999:8000 \
  --gpus all \
  -v "$(pwd)"/app:/root/app \
  "$IMAGE_NAME" \
  tail -f /dev/null

if [ $? -eq 0 ]; then
    echo "🎉 容器已成功启动！"
    echo "📄 API 文档请访问: http://localhost:9999/docs"
    echo "🖼️  测试 /detect 接口上传图片"
else
    echo "❌ 启动失败，请检查日志: docker logs $CONTAINER_NAME"
    exit 1
fi
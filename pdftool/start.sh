#!/bin/bash

# =============================
# 启动 pdf2zh 服务的脚本
# =============================

# 配置参数
IMAGE_NAME="192.168.1.101:7443/top/byaidu/pdf2zh:latest"
CONTAINER_NAME="pdf2zh"

# ----------------------------
# 步骤 1：停止并删除已有容器
# ----------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 停止并删除已有容器: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

echo "🚀 启动容器 $CONTAINER_NAME 并运行 pdf2zh 服务..."

docker run -d \
  --name "$CONTAINER_NAME" \
  -p 7860:7860 \
  -p 6791:8000 \
  -v "$(pwd)"/app:/root/app \
  -w /root/app \
  "$IMAGE_NAME" \
  uvicorn app:app --host 0.0.0.0 --port 8000 

# ----------------------------
# 步骤 4：输出提示信息
# ----------------------------
if [ $? -eq 0 ]; then
    echo "🎉 容器已成功启动！"
    echo "📄 API 文档请访问: http://localhost:6791/docs"
else
    echo "❌ 启动失败，请检查日志: docker logs $CONTAINER_NAME"
    exit 1
fi
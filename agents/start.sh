#!/bin/bash

# =============================
# 启动 YOLO + FastAPI 服务的脚本
# =============================

# 配置参数
IMAGE_NAME="192.168.1.101:7443/top/agents:latest"
CONTAINER_NAME="agent"

# ----------------------------
# 步骤 1：停止并删除已有容器
# ----------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 停止并删除已有容器: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  -p 6790:8000 \
  -v "$(pwd)"/app:/root/app \
  -w /root/app \
  "$IMAGE_NAME" \
  uvicorn agent_service.app:app --host 0.0.0.0 --port 8000

if [ $? -eq 0 ]; then
    echo "🎉 容器已成功启动！"
    echo "📄 API 文档请访问: http://localhost:6790/docs"
else
    echo "❌ 启动失败，请检查日志: docker logs $CONTAINER_NAME"
    exit 1
fi

IMAGE_NAME="192.168.1.101:7443/top/labelu:latest"
CONTAINER_NAME="labelu"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 停止并删除已有容器: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

# docker run -d -p 6777:8000 --name labelu labelu:latest


docker run -d \
  -p 6777:8000 \
  --name "$CONTAINER_NAME"  \
  -v $(pwd)/data:/root/.local/share/labelu \
  "$IMAGE_NAME"

if [ $? -eq 0 ]; then
    echo "🎉 容器已成功启动！"
    echo "📄 API 文档请访问: http://localhost:6777/docs"
else
    echo "❌ 启动失败，请检查日志: docker logs $CONTAINER_NAME"
    exit 1
fi
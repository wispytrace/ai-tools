docker run -d \
  --name yolo \
  -p 8000:8000 \
  -v "$(pwd)"/app:/ultralytics \
  yolo:latest \
  tail -f /dev/null
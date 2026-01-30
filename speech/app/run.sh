#!/bin/bash

# 添加 cuDNN lib 路径
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 可选：打印验证
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

# 启动你的程序
exec python app.py "$@"
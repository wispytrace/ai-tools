from modelscope import snapshot_download

MODEL_ID = "AI-ModelScope/XTTS-v2"  # 替换为你的模型
MODEL_DIR = f"/root/binghao/models/{MODEL_ID.split('/')[-1]}"

print(f"下载 {MODEL_ID} 到 {MODEL_DIR}")

model_path = snapshot_download(
    MODEL_ID,
    cache_dir=f"{MODEL_DIR}/.cache",
    local_dir=MODEL_DIR
)

print(f"完成! 路径: {model_path}")
import json
import shutil
from pathlib import Path

# ================== 配置 ==================
FAILURE_JSON = "failures.json"      # 你的失败记录 JSON 文件
OUTPUT_DIR = "failed_images"       # 输出文件夹名
# =========================================

def main():
    # 1. 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"📁 输出目录: {output_dir.absolute()}")

    # 2. 读取失败记录
    with open(FAILURE_JSON, "r", encoding="utf-8") as f:
        failures = json.load(f)

    print(f"📋 共找到 {len(failures)} 个失败样本")

    copied = 0
    for item in failures:
        src_path = Path(item["image_path"])
        if not src_path.exists():
            print(f"⚠️ 跳过（文件不存在）: {src_path}")
            continue

        dst_path = output_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)  # copy2 保留元数据
            copied += 1
        except Exception as e:
            print(f"❌ 复制失败 {src_path}: {e}")

    print(f"\n✅ 成功导出 {copied} 张失败图片到 '{OUTPUT_DIR}' 文件夹")
    print(f"💡 你可以直接打开该文件夹查看所有识别错误的图像")

if __name__ == "__main__":
    main()

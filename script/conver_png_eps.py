import os
import numpy as np
from PIL import Image

def convert_png_to_lightweight_pdf(folder_path, scale_factor=0.6, quality=85):
    """
    folder_path: 图片所在文件夹路径
    scale_factor: 缩放比例 (0.1-1.0)，建议 0.6，既保证清晰度又能显著减小体积
    quality: 压缩质量 (1-100)，PDF 内部使用 JPEG 压缩，85 是一个平衡点
    """
    output_folder = os.path.join(folder_path, "TASE_PDF_Figures")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png") or True:
            png_path = os.path.join(folder_path, filename)
            pdf_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + ".pdf")

            try:
                with Image.open(png_path) as img:
                    # 1. 统一转为 RGB (PDF 压缩必须在 RGB 模式下效果最好)
                    img = img.convert("RGB")
                    
                    # 2. 缩放数组尺寸 (降低像素总量)
                    if scale_factor < 1.0:
                        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                        img = img.resize(new_size, resample=Image.LANCZOS)
                    
                    # 3. 保存为 PDF
                    # resolution=300 保证了在 LaTeX 打印时的清晰度
                    # optimize=True 会进一步优化 PDF 内部结构
                    img.save(pdf_path, "PDF", resolution=300.0, quality=quality, optimize=True)
                
                print(f"转换成功: {filename} -> {os.path.basename(pdf_path)} (缩放: {scale_factor})")
            except Exception as e:
                print(f"转换 {filename} 出错: {e}")

if __name__ == "__main__":
    # 请修改为你存放图片的实际路径
    target_folder = './eps/figure' 
    convert_png_to_lightweight_pdf(target_folder, scale_factor=0.6, quality=85)
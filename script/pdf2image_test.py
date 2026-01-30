import os
from pdf2image import convert_from_path

def split_pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    pdf_path: PDF文件路径
    output_folder: 导出图片的文件夹
    dpi: 分辨率，300是论文打印标准
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将 PDF 页面转换为 PIL 图片对象列表
    # thread_count 开启多线程加速
    images = convert_from_path(pdf_path, dpi=dpi, thread_count=4)

    for i, image in enumerate(images):
        # 命名格式：原文件名_第几页.png
        base_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
        image_path = os.path.join(output_folder, f"{base_name}_{i+1}.png")
        
        # 导出为 PNG (无损压缩)
        image.save(image_path, "PNG")
        print(f"已导出: {image_path}")

if __name__ == "__main__":
    # 使用示例
    pdf_file = '/root/binghao/script/93ad3724-d9fa-4ae2-85d0-748dbcefcde7.pdf' 
    output_dir = './exported_figures'
    split_pdf_to_images(pdf_file, output_dir)
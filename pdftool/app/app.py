# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import shutil
from pdf2image import convert_from_path
import img2pdf
from io import BytesIO
from PIL import Image
# 导入 pdf2zh 模块
from pdf2zh import translate
from pdf2zh.doclayout import OnnxModel, ModelInstance
import tempfile

app = FastAPI(title="PDF 中英翻译 API", description="支持生成单栏和双栏翻译 PDF")

# ========== 设置路径 ========== #
ROOT_DIR = Path(__file__).parent.absolute()

UPLOAD_DIR = ROOT_DIR / "upload"
OUTPUT_DIR = ROOT_DIR / "output"
TEMP_MERGE_DIR = ROOT_DIR / "merged"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_MERGE_DIR.mkdir(exist_ok=True)

# 加载模型
ModelInstance.value = OnnxModel.load_available()
print("Loaded BABELDOC_MODEL:", ModelInstance.value)


def run_translation(file_path: str):
    """执行翻译任务，返回 (mono, dual) 路径"""
    params = {
        'lang_in': 'en',
        'lang_out': 'zh',
        'service': 'bing',
        'thread': 4,
        'model': ModelInstance.value,
        'output': str(OUTPUT_DIR),
    }
    try:
        result = translate(files=[file_path], **params)
        if not result:
            return None, None
        return result[0]  # (mono, dual)
    except Exception as e:
        print(f"Translation error: {e}")
        return None, None


@app.post("/translate-dual/", response_class=FileResponse)
async def translate_dual(file: UploadFile = File(...)):
    """上传 PDF，返回中英双栏对照版"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    input_path = UPLOAD_DIR / file.filename

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"Saved to {input_path}")

        mono_pdf, dual_pdf = run_translation(str(input_path))

        if not dual_pdf or not os.path.exists(dual_pdf):
            raise HTTPException(status_code=500, detail="未生成双栏 PDF")

        return FileResponse(
            path=dual_pdf,
            media_type='application/pdf',
            filename=f"dual_{Path(file.filename).stem}.pdf"
        )
    except Exception as e:
        print(f"Error in /translate-dual/: {e}")
        raise HTTPException(status_code=500, detail=f"服务错误: {str(e)}")


@app.post("/translate-mono/", response_class=FileResponse)
async def translate_mono(file: UploadFile = File(...)):
    """上传 PDF，返回单栏中文翻译版"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    input_path = UPLOAD_DIR / file.filename

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"Saved to {input_path}")

        mono_pdf, dual_pdf = run_translation(str(input_path))

        if not mono_pdf or not os.path.exists(mono_pdf):
            raise HTTPException(status_code=500, detail="未生成单栏 PDF")

        return FileResponse(
            path=mono_pdf,
            media_type='application/pdf',
            filename=f"mono_{Path(file.filename).stem}.pdf"
        )
    except Exception as e:
        print(f"Error in /translate-mono/: {e}")
        raise HTTPException(status_code=500, detail=f"服务错误: {str(e)}")

@app.post("/merge-dual-pages/", response_class=FileResponse)
async def merge_dual_pages(file: UploadFile = File(...)):
    """
    将上传的 PDF 每两页合并为一页（左右并排，中间留白）
    使用 img2pdf + 高清图像直接生成 PDF，避免 ReportLab 降质
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 保存上传文件
    temp_input = TEMP_MERGE_DIR / file.filename
    with open(temp_input, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_filename = f"merged_{Path(file.filename).stem}.pdf"
    output_path = TEMP_MERGE_DIR / output_filename

    try:
        # --- 设置 Poppler 路径（仅 Windows 需要）---
        poppler_path = None
        if os.name == 'nt':  # Windows
            poppler_path = r"C:\tools\poppler\bin"  # 修改为你自己的路径！

        # --- 将 PDF 转为图像列表（300 DPI 高清）---
        images = convert_from_path(
            str(temp_input),
            dpi=300,           # 关键：高分辨率渲染
            thread_count=4,
            poppler_path=poppler_path
        )
    except Exception as e:
        print(f"PDF to image conversion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF 渲染失败，请检查 Poppler 是否安装: {str(e)}"
        )

    # --- 设置输出页面尺寸：A4 横向（更宽）---
    from reportlab.lib.pagesizes import A4, landscape
    page_width_total_pt, page_height_pt = landscape(A4)  # 单位：point (约 842 x 595)

    # 我们将以 300 DPI 为基准，计算目标像素大小
    # 1 inch = 72 pt = 300 px → 所以每 point ≈ 300 / 72 = 4.1667 px
    scale_factor = 300 / 72  # pt → px 的换算因子（基于 300 DPI）
    page_width_total_px = int(page_width_total_pt * scale_factor)
    page_height_total_px = int(page_height_pt * scale_factor)

    gap_px = int(40 * scale_factor)  # 中间空白 40pt → 像素
    usable_width_per_side_px = (page_width_total_px - gap_px) // 2

    # 存储所有拼接后的图像（用于生成 PDF）
    list_of_image_bytes = []

    for i in range(0, len(images), 2):
        # 创建一张白色背景的大图（用于左右拼接）
        combined_img = Image.new('RGB', (page_width_total_px, page_height_total_px), 'white')

        # === 左侧页面（第 i 页）===
        img1 = images[i]
        img1_ratio = img1.width / img1.height
        target_height_1 = int(page_height_total_px * 0.95)
        target_width_1 = int(target_height_1 * img1_ratio)

        # 如果太宽，则按最大宽度限制
        if target_width_1 > usable_width_per_side_px:
            target_width_1 = usable_width_per_side_px
            target_height_1 = int(target_width_1 / img1_ratio)

        img1_resized = img1.resize((target_width_1, target_height_1), Image.Resampling.LANCZOS)
        y1 = (page_height_total_px - target_height_1) // 2  # 垂直居中
        combined_img.paste(img1_resized, (0, y1))

        # === 右侧页面（第 i+1 页）===
        if i + 1 < len(images):
            img2 = images[i + 1]
            img2_ratio = img2.width / img2.height
            target_height_2 = int(page_height_total_px * 0.95)
            target_width_2 = int(target_height_2 * img2_ratio)

            if target_width_2 > usable_width_per_side_px:
                target_width_2 = usable_width_per_side_px
                target_height_2 = int(target_width_2 / img2_ratio)

            img2_resized = img2.resize((target_width_2, target_height_2), Image.Resampling.LANCZOS)
            y2 = (page_height_total_px - target_height_2) // 2
            x2 = usable_width_per_side_px + gap_px
            combined_img.paste(img2_resized, (x2, y2))
        # else: 单数页时右侧留白

        # 保存为 JPEG 字节流（高质量）
        img_byte_arr = BytesIO()
        combined_img.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
        img_byte_arr.seek(0)
        list_of_image_bytes.append(img_byte_arr)

    # --- 使用 img2pdf 直接生成 PDF（关键：不经过 canvas 绘图）---
    try:
        pdf_data = img2pdf.convert(
            [img_bytes.getvalue() for img_bytes in list_of_image_bytes]
        )
        with open(output_path, "wb") as f:
            f.write(pdf_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 生成失败: {str(e)}")
    finally:
        # 清理图像字节流
        for buf in list_of_image_bytes:
            buf.close()

    # 删除原始临时文件
    try:
        temp_input.unlink(missing_ok=True)
    except:
        pass

    return FileResponse(
        path=output_path,
        media_type='application/pdf',
        filename=output_filename,
        headers={"Content-Disposition": f"attachment; filename={output_filename}"}
    )

@app.get("/")
def root():
    return {
        "message": "PDF 翻译 & 合并 API",
        "endpoints": [
            "/translate-dual/   → POST 上传 PDF，返回中英双栏翻译",
            "/translate-mono/  → POST 上传 PDF，返回纯中文翻译",
            "/merge-dual-pages/ → POST 上传 PDF，每两页合并为一页（左右对照）"
        ],
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
        "temp_merge_dir": str(TEMP_MERGE_DIR)
    }
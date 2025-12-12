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


def merge_pdf_pages(input_pdf_path: Path, output_filename: str) -> Path:
    """
    将输入 PDF 每两页合并为一页（左右并排），返回输出路径
    复用逻辑，供多个接口调用
    """
    output_path = TEMP_MERGE_DIR / output_filename

    try:
        poppler_path = r"C:\tools\poppler\bin" if os.name == 'nt' else None

        images = convert_from_path(
            str(input_pdf_path),
            dpi=300,
            thread_count=4,
            poppler_path=poppler_path
        )
    except Exception as e:
        raise RuntimeError(f"PDF to image conversion failed: {e}")

    from reportlab.lib.pagesizes import A4, landscape
    page_width_pt, page_height_pt = landscape(A4)
    scale_factor = 300 / 72
    page_width_px = int(page_width_pt * scale_factor)
    page_height_px = int(page_height_pt * scale_factor)
    gap_px = int(40 * scale_factor)
    usable_width_per_side = (page_width_px - gap_px) // 2

    merged_image_buffers = []

    for i in range(0, len(images), 2):
        combined = Image.new('RGB', (page_width_px, page_height_px), 'white')

        # Left: page i
        img1 = images[i]
        ratio1 = img1.width / img1.height
        h1 = int(page_height_px * 0.95)
        w1 = int(h1 * ratio1)
        if w1 > usable_width_per_side:
            w1 = usable_width_per_side
            h1 = int(w1 / ratio1)
        img1_resized = img1.resize((w1, h1), Image.Resampling.LANCZOS)
        y1 = (page_height_px - h1) // 2
        combined.paste(img1_resized, (0, y1))

        # Right: page i+1 (if exists)
        if i + 1 < len(images):
            img2 = images[i + 1]
            ratio2 = img2.width / img2.height
            h2 = int(page_height_px * 0.95)
            w2 = int(h2 * ratio2)
            if w2 > usable_width_per_side:
                w2 = usable_width_per_side
                h2 = int(w2 / ratio2)
            img2_resized = img2.resize((w2, h2), Image.Resampling.LANCZOS)
            y2 = (page_height_px - h2) // 2
            x2 = usable_width_per_side + gap_px
            combined.paste(img2_resized, (x2, y2))

        buf = BytesIO()
        combined.save(buf, format='JPEG', quality=95, optimize=True)
        buf.seek(0)
        merged_image_buffers.append(buf)

    try:
        pdf_data = img2pdf.convert([buf.getvalue() for buf in merged_image_buffers])
        with open(output_path, "wb") as f:
            f.write(pdf_data)
    except Exception as e:
        raise RuntimeError(f"PDF generation failed: {e}")
    finally:
        for buf in merged_image_buffers:
            buf.close()

    return output_path


@app.post("/translate-dual/", response_class=FileResponse)
async def translate_dual(file: UploadFile = File(...)):
    """上传 PDF，返回中英双栏对照版（交错页面）"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    input_path = UPLOAD_DIR / file.filename
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
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
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
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
    """将上传的 PDF 每两页合并为一页（左右并排）"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    temp_input = TEMP_MERGE_DIR / file.filename
    with open(temp_input, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        output_filename = f"merged_{Path(file.filename).stem}.pdf"
        output_path = merge_pdf_pages(temp_input, output_filename)
        return FileResponse(
            path=output_path,
            media_type='application/pdf',
            filename=output_filename
        )
    except Exception as e:
        print(f"Error in /merge-dual-pages/: {e}")
        raise HTTPException(status_code=500, detail=f"合并失败: {str(e)}")
    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/translate-dual-and-merge/", response_class=FileResponse)
async def translate_dual_and_merge(file: UploadFile = File(...)):
    """
    上传 PDF → 生成交错双栏 PDF → 自动合并每两页为一页左右对照 → 返回结果
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    input_path = UPLOAD_DIR / file.filename
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Step 1: 翻译生成 dual PDF（交错页面）
        mono_pdf, dual_pdf = run_translation(str(input_path))
        if not dual_pdf or not os.path.exists(dual_pdf):
            raise HTTPException(status_code=500, detail="翻译失败，未生成 dual PDF")

        # Step 2: 合并交错页面
        output_filename = f"merged_dual_{Path(file.filename).stem}.pdf"
        merged_path = merge_pdf_pages(Path(dual_pdf), output_filename)

        return FileResponse(
            path=merged_path,
            media_type='application/pdf',
            filename=output_filename
        )
    except Exception as e:
        print(f"Error in /translate-dual-and-merge/: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "PDF 翻译 & 合并 API",
        "endpoints": [
            "/translate-dual/           → POST 上传 PDF，返回中英交错页面 PDF",
            "/translate-mono/          → POST 上传 PDF，返回纯中文翻译",
            "/merge-dual-pages/        → POST 上传 PDF，每两页合并为一页",
            "/translate-dual-and-merge/ → POST 上传 PDF，自动翻译+合并，返回最终对照 PDF"
        ],
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
        "temp_merge_dir": str(TEMP_MERGE_DIR)
    }

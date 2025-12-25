# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import shutil
import uuid
import threading
import queue
import time
from typing import Dict
from pdf2image import convert_from_path
import img2pdf
from io import BytesIO
from PIL import Image
from pdf2zh import translate
from pdf2zh.doclayout import OnnxModel, ModelInstance
import traceback

app = FastAPI(title="PDF 中英翻译 API", description="支持生成单栏和双栏翻译 PDF")

# ========== 设置路径 ========== #
ROOT_DIR = Path(__file__).parent.absolute()
UPLOAD_DIR = ROOT_DIR / "upload"
OUTPUT_DIR = ROOT_DIR / "output"
TEMP_MERGE_DIR = ROOT_DIR / "merged"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_MERGE_DIR.mkdir(exist_ok=True)

# ========== 全局任务系统 ========== #
TASKS: Dict[str, dict] = {}
TASK_QUEUE = queue.Queue()  # 线程安全队列

# 加载模型（全局一次）
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
        return result[0]
    except Exception as e:
        traceback.print_exc()
        print(f"Translation error: {e}")
        return None, None


def merge_pdf_pages(input_pdf_path: Path, output_filename: str) -> Path:
    """将输入 PDF 每两页合并为一页（左右并排）"""
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


# ========== 任务执行函数（支持所有类型） ========== #
def execute_task(task: dict):
    task_id = task["task_id"]
    input_path = task["input_path"]
    original_filename = task["original_filename"]
    task_type = task["task_type"]  # "dual", "mono", "merge", "dual_and_merge"

    print(f"[WORKER] Executing {task_type} task {task_id}...")
    TASKS[task_id]["status"] = "processing"

    try:
        if task_type == "dual":
            mono_pdf, dual_pdf = run_translation(str(input_path))
            if not dual_pdf or not os.path.exists(dual_pdf):
                raise Exception("未生成双栏 PDF")
            result_path = Path(dual_pdf)
            filename = f"dual_{Path(original_filename).stem}.pdf"

        elif task_type == "mono":
            mono_pdf, dual_pdf = run_translation(str(input_path))
            if not mono_pdf or not os.path.exists(mono_pdf):
                raise Exception("未生成单栏 PDF")
            result_path = Path(mono_pdf)
            filename = f"mono_{Path(original_filename).stem}.pdf"

        elif task_type == "merge":
            output_filename = f"merged_{Path(original_filename).stem}.pdf"
            result_path = merge_pdf_pages(input_path, output_filename)
            filename = output_filename

        elif task_type == "dual_and_merge":
            mono_pdf, dual_pdf = run_translation(str(input_path))
            if not dual_pdf or not os.path.exists(dual_pdf):
                raise Exception("翻译未生成 dual PDF")
            output_filename = f"merged_dual_{Path(original_filename).stem}.pdf"
            result_path = merge_pdf_pages(Path(dual_pdf), output_filename)
            filename = output_filename

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        TASKS[task_id].update({
            "status": "completed",
            "result_path": str(result_path),
            "filename": filename
        })
        print(f"[WORKER] Task {task_id} ({task_type}) completed.")

    except Exception as e:
        error_msg = str(e)
        print(f"[WORKER] Task {task_id} ({task_type}) failed: {error_msg}")
        TASKS[task_id].update({
            "status": "failed",
            "error": error_msg
        })
    finally:
        input_path.unlink(missing_ok=True)


# ========== 后台工作线程 ========== #
def worker():
    while True:
        try:
            task = TASK_QUEUE.get(timeout=1)
            execute_task(task)
            TASK_QUEUE.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[WORKER] Unexpected error: {e}")
            time.sleep(1)


worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()
print("[SYSTEM] Background worker thread started.")


# ========== 统一提交函数 ========== #
def _submit_task(file: UploadFile, task_type: str):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    task_id = str(uuid.uuid4())
    safe_chars = "".join(c if c.isalnum() or c in "._-" else "_" for c in file.filename)
    input_path = UPLOAD_DIR / f"{task_id}_{safe_chars}"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    TASKS[task_id] = {
        "status": "pending",
        "filename": file.filename,
        "task_type": task_type
    }

    TASK_QUEUE.put({
        "task_id": task_id,
        "input_path": input_path,
        "original_filename": file.filename,
        "task_type": task_type
    })

    print(f"[API] {task_type} task {task_id} queued.")
    return {
        "task_id": task_id,
        "status": "submitted",
        "message": f"{task_type} 任务已提交，请通过 GET /task/{{task_id}} 查询状态"
    }


# ========== 所有接口（全部异步） ========== #

@app.post("/translate-dual/")
async def translate_dual(file: UploadFile = File(...)):
    return _submit_task(file, "dual")


@app.post("/translate-mono/")
async def translate_mono(file: UploadFile = File(...)):
    return _submit_task(file, "mono")


@app.post("/merge-dual-pages/")
async def merge_dual_pages(file: UploadFile = File(...)):
    return _submit_task(file, "merge")


@app.post("/translate-dual-and-merge/")
async def translate_dual_and_merge(file: UploadFile = File(...)):
    return _submit_task(file, "dual_and_merge")


# ========== 状态查询 & 下载 ========== #

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    task = TASKS[task_id]
    if task["status"] == "completed":
        return {
            "task_id": task_id,
            "status": "completed",
            "download_url": f"/download/{task_id}"
        }
    elif task["status"] == "failed":
        return {
            "task_id": task_id,
            "status": "failed",
            "error": task.get("error", "未知错误")
        }
    else:
        return {
            "task_id": task_id,
            "status": task["status"]
        }


@app.get("/download/{task_id}", response_class=FileResponse)
async def download_result(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = TASKS[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    result_path = Path(task["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")
    return FileResponse(path=result_path, media_type='application/pdf', filename=task["filename"])


# ========== 根路径 ========== #

@app.get("/")
def root():
    return {
        "message": "PDF 翻译 & 合并 API（全异步）",
        "endpoints": [
            "POST /translate-dual/          → 提交双栏翻译任务",
            "POST /translate-mono/         → 提交单栏翻译任务",
            "POST /merge-dual-pages/       → 提交 PDF 合并任务",
            "POST /translate-dual-and-merge/ → 提交翻译+合并任务",
            "GET  /task/{task_id}          → 查询任务状态",
            "GET  /download/{task_id}      → 下载结果"
        ],
        "worker_thread_alive": worker_thread.is_alive(),
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
        "temp_merge_dir": str(TEMP_MERGE_DIR)
    }


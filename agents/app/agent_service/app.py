# # chemistry_extraction/main.py
# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from fastapi.responses import JSONResponse
# import os
# import json
# from typing import Dict, Any
# from datetime import datetime
# from .utils.draw_result_export import export_text_draw_result, export_yolo_compound_draw_rsult, export_extported_result_draw_result
# from .state import ChemistryExtractionState
# from .workflow import ChemistryWorkflow
# from .config import Config

# # 配置目录
# UPLOAD_DIR = "upload_pdf"
# OUTPUT_DIR = "output"

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# app = FastAPI(title="Chemistry Information Extraction API", version="1.0")

# # 验证配置
# Config.validate()

# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 生产环境改为具体域名
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/compound-extraction/", summary="Extract compounds from uploaded PDF")
# async def compound_extraction(pdf_file: UploadFile = File(...)):
#     """
#     上传一个PDF文件，执行化合物信息提取。
#     """
#     response_data = await run_extraction(pdf_file, agent="compound_extraction", is_traslate=True)
#     return JSONResponse(content=response_data)

# @app.post("/reaction-extraction/", summary="Extract reactions from uploaded PDF")
# async def reaction_extraction(pdf_file: UploadFile = File(...)):
#     """
#     上传一个PDF文件，执行化学反应式信息提取。
#     """
#     return await run_extraction(pdf_file, agent="reaction_extraction")


# @app.post("/compound/", summary="Extract compounds from uploaded PDF")
# async def compound(pdf_file: UploadFile = File(...)):
#     """
#     上传一个PDF文件，执行化合物信息提取。
#     """
#     response_data = run_extraction(pdf_file, agent="compound_extraction")
#     parsed_response_data = {}
#     keys = ["success", "reactions"]
#     for key in keys:
#         parsed_response_data[key] = response_data[key]
#     return JSONResponse(content=parsed_response_data)

# @app.post("/reaction-extraction/", summary="Extract reactions from uploaded PDF")
# async def reaction(pdf_file: UploadFile = File(...)):
#     """
#     上传一个PDF文件，执行化学反应式信息提取。
#     """
#     response_data = run_extraction(pdf_file, agent="reaction_extraction")
#     parsed_response_data = {}
#     keys = ["success", "compounds"]
#     for key in keys:
#         parsed_response_data[key] = response_data[key]
#     return JSONResponse(content=parsed_response_data)

# async def run_extraction(pdf_file: UploadFile, agent: str, is_traslate=False) -> JSONResponse:
#     # 检查文件类型
#     if not pdf_file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

#     # 保存上传的文件
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe_filename = f"{pdf_file.filename}"
#     file_path = os.path.join(UPLOAD_DIR, safe_filename)

#     with open(file_path, "wb") as f:
#         content = await pdf_file.read()
#         f.write(content)

#     print(f"📄 Saved uploaded PDF to: {file_path}")

#     # 初始化状态
#     initial_state = ChemistryExtractionState(
#         pdf_path=file_path,
#         current_stage=["initialized"],
#     )

#     # 创建并运行工作流
#     print(f"🔄 Running {agent} workflow...")
#     workflow = ChemistryWorkflow()
#     workflow.set_workflow(agent)
#     app_instance = workflow.compile()

#     try:
#         print("🚀 Starting extraction process...")
#         final_state = app_instance.invoke(initial_state)
#         final_state_dict = dict(final_state)
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         print(f"❌ Error during extraction: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

#     # 保存结果
#     result_filename = f"{timestamp}_{agent}_results.json"
#     output_path = os.path.join(OUTPUT_DIR, result_filename)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(final_state_dict, f, ensure_ascii=False, indent=2)

#     print(f"💾 Results saved to: {output_path}")

#     # 构造响应摘要
#     metadata = final_state_dict.get("metadata", {})
#     errors = final_state_dict.get("errors", [])
#     if is_traslate:
#         text_url = export_text_draw_result(final_state_dict["text_jsons"])
#     else:
#         text_url = []

#     reactions_url = export_extported_result_draw_result(final_state_dict.get("extported_result", []))
#     compounds_url = export_yolo_compound_draw_rsult(final_state_dict.get("compounds", []))

#     response_data = {
#         "success": True,
#         "result_file": output_path,
#         "metadata": metadata,
#         "errors": errors,
#         "summary": {
#             "text_sections_processed": metadata.get("text_sections_count", 0),
#             "images_analyzed": metadata.get("image_extractions_count", 0),
#             "fusion_success": metadata.get("fusion_success", False),
#             "error_count": len(errors)
#         },
#         "compounds": final_state_dict.get("compounds", {}),
#         "reactions": final_state_dict.get("extported_result", {}), 
#         "texts_url": text_url,
#         "compounds_url": compounds_url,
#         "reactions_url": reactions_url
#     }

#     if errors:
#         print(f"⚠️  Encountered {len(errors)} errors during extraction.")
#         response_data["success"] = False
    
#     return response_data





# @app.get("/", summary="Root endpoint")
# def root():
#     return {
#         "message": "Welcome to Chemistry Information Extraction API",
#         "endpoints": [
#             "/compound-extraction/ - POST (form-data: pdf_file)",
#             "/reaction-extraction/ - POST (form-data: pdf_file)"
#         ]
#     }

# chemistry_extraction/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import json
from typing import Dict, Any
from datetime import datetime
from .utils.draw_result_export import (
    export_text_draw_result,
    export_yolo_compound_draw_rsult,
    export_extported_result_draw_result
)
from .state import ChemistryExtractionState
from .workflow import ChemistryWorkflow
from .config import Config

# 配置目录
UPLOAD_DIR = "upload_pdf"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Chemistry Information Extraction API", version="1.0")

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 验证配置
Config.validate()


# ======================
# 💡 核心业务逻辑拆分
# ======================

def save_uploaded_pdf(pdf_file: UploadFile) -> str:
    """保存上传的 PDF 文件，返回完整路径"""
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_filename = pdf_file.filename
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as f:
        content = pdf_file.file.read()
        f.write(content)

    print(f"📄 Saved uploaded PDF to: {file_path}")
    return file_path


def execute_workflow(file_path: str, agent: str) -> Dict[str, Any]:
    """执行化学信息提取工作流，返回最终状态字典"""
    initial_state = ChemistryExtractionState(
        pdf_path=file_path,
        current_stage=["initialized"],
    )

    workflow = ChemistryWorkflow()
    workflow.set_workflow(agent)
    app_instance = workflow.compile()

    try:
        print(f"🚀 Running {agent} workflow...")
        final_state = app_instance.invoke(initial_state)
        return dict(final_state)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


def export_results(final_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """导出可视化结果（图片 URL 列表）"""
    text_url = export_text_draw_result(final_state_dict.get("text_jsons", []))
    compounds_url = export_yolo_compound_draw_rsult(final_state_dict.get("compounds", []))
    reactions_url = export_extported_result_draw_result(final_state_dict.get("extported_result", []))
    
    return {
        "texts_url": text_url,
        "compounds_url": compounds_url,
        "reactions_url": reactions_url
    }


def build_response(
    final_state_dict: Dict[str, Any],
    agent: str,
    is_translate: bool = False,
    keys_to_include: list = None,
    is_export_result: bool = False,
) -> Dict[str, Any]:
    """构建标准 API 响应"""
    metadata = final_state_dict.get("metadata", {})
    errors = final_state_dict.get("errors", [])
    success = len(errors) == 0

    # 保存原始结果到 JSON 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{timestamp}_{agent}_results.json"
    output_path = os.path.join(OUTPUT_DIR, result_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_state_dict, f, ensure_ascii=False, indent=2)
    print(f"💾 Results saved to: {output_path}")

    # 构建基础响应
    response_data = {
        "success": success,
        "result_file": output_path,
        "metadata": metadata,
        "errors": errors,
        "summary": {
            "text_sections_processed": metadata.get("text_sections_count", 0),
            "images_analyzed": metadata.get("image_extractions_count", 0),
            "fusion_success": metadata.get("fusion_success", False),
            "error_count": len(errors)
        },
        "compounds": final_state_dict.get("compounds", {}),
        "reactions": final_state_dict.get("extported_result", {}),
    }
    if is_export_result:
        exported_result = export_results(final_state_dict)
        for key, value in exported_result.items():
            response_data[key] = value


    # 如果指定了 keys_to_include，则只返回这些字段（用于 /compound 和 /reaction）
    if keys_to_include is not None:
        filtered_response = {k: response_data[k] for k in keys_to_include if k in response_data}
        return filtered_response

    return response_data


async def handle_extraction_request(
    pdf_file: UploadFile,
    agent: str,
    is_translate: bool = False,
    keys_to_include: list = None
) -> JSONResponse:
    """统一处理提取请求的入口"""
    file_path = save_uploaded_pdf(pdf_file)
    final_state = execute_workflow(file_path, agent)
    response_data = build_response(final_state, agent, is_translate, keys_to_include)
    return JSONResponse(content=response_data)


# ======================
# 🌐 API 路由（保持接口不变）
# ======================

@app.post("/compound-extraction/", summary="Extract compounds from uploaded PDF")
async def compound_extraction(pdf_file: UploadFile = File(...)):
    return await handle_extraction_request(pdf_file, agent="compound_extraction", is_translate=True, is_export_result=True)


@app.post("/reaction-extraction/", summary="Extract reactions from uploaded PDF")
async def reaction_extraction(pdf_file: UploadFile = File(...)):
    return await handle_extraction_request(pdf_file, agent="reaction_extraction", is_export_result=True)


@app.post("/compound/", summary="Extract compounds (minimal response)")
async def compound(pdf_file: UploadFile = File(...)):
    # 注意：原代码中 /compound 返回的是 ["success", "reactions"] —— 这可能是笔误？
    # 根据上下文，应该是返回 compounds。但为保持兼容，按你写的来。
    return await handle_extraction_request(
        pdf_file, 
        agent="compound_extraction",
        keys_to_include=["success", "compounds"]  
    )


@app.post("/reaction/", summary="Extract reactions (minimal response)")
async def reaction(pdf_file: UploadFile = File(...)):
    return await handle_extraction_request(
        pdf_file,
        agent="reaction_extraction",
        keys_to_include=["success", "reactions"]  
    )


@app.get("/", summary="Root endpoint")
def root():
    return {
        "message": "Welcome to Chemistry Information Extraction API",
        "endpoints": [
            "/compound-extraction/ - POST (form-data: pdf_file)",
            "/reaction-extraction/ - POST (form-data: pdf_file)",
            "/compound/ - POST (minimal response)",
            "/reaction/ - POST (minimal response)"
        ]
    }

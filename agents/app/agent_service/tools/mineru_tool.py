# chemistry_extraction/tools/mineru_tool.py

import os
import json
import requests
import zipfile
import tarfile
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from pathlib import Path
import time
from .base_tool import BaseTool, ToolResult

TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIyNDEwMDE4OSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2NTE3MzY2OCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiOTEyNjM2OWItZjkxYi00MWQ3LThkYmUtZjA2ZWRkZTFjYzZiIiwiZW1haWwiOiIiLCJleHAiOjE3NjYzODMyNjh9.pcV4OM8NVm3RnYbU3wZjA5rAaagFngagOkIfj_v5OGCbCDINsXWQKzTzsr-YSsSNw9j4L4v7f8llVXWyzUX8WA"


class MinerUExtractInput:
    """输入数据结构模拟 Pydantic（轻量替代）"""
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MinerUExtractInput':
        pdf_path = data.get("pdf_path")
        if not pdf_path:
            raise ValueError("Missing required field: file_path")
        if not isinstance(pdf_path, str):
            raise ValueError("field 'file_path' must be a string")
        if not os.path.exists(pdf_path):
            raise ValueError(f"File not found: {pdf_path}")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported")
        return cls(pdf_path=pdf_path)


class MinerUExtractTool(BaseTool):
    """
    MCP 工具：使用 MinerU API 从 PDF 中提取化学文本与图像
    支持自动上传、轮询、下载并解压结构化结果
    """

    name: str = "mineru_pdf_extraction"
    description: str = (
        "使用 MinerU 服务从科学PDF中图像与文本信息，"
        "返回解析后的 Markdown 和图片目录路径"
    )

    def __init__(self, config: Optional[Dict[str, Any]] = {}):
        super().__init__(config)
        self.timeout = self.config.get("timeout", 120.0)  # 提取可能耗时较长
        self.poll_interval = self.config.get("poll_interval", 5.0)
        self.api_base_url = self.config.get("api_base_url", "https://mineru.net/api/v4")

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        try:
            MinerUExtractInput.from_dict(input_data)
            return True
        except Exception as e:
            self.logger.warning(f"Input validation failed: {str(e)}")
            return False

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        try:
            # 解析输入
            inp = MinerUExtractInput.from_dict(input_data)
            pdf_path = inp.pdf_path

            self.logger.info(f"Starting extraction for PDF: {pdf_path}")

            # Step 1: 上传任务
            batch_id = self._upload_task(pdf_path)
            if not batch_id:
                return ToolResult(
                    success=False,
                    data={},
                    tool_name=self.name,
                    execution_time=time.time() - start_time,
                    error="Failed to upload PDF to MinerU"
                )
            self.logger.info(f"Upload successful, batch_id={batch_id}")

            # Step 2: 轮询结果
            zip_url = self._poll_result(batch_id)
            if not zip_url:
                return ToolResult(
                    success=False,
                    data={},
                    tool_name=self.name,
                    execution_time=time.time() - start_time,
                    error="Failed to retrieve extraction result (timeout or server error)"
                )
            self.logger.info(f"Extraction completed, downloading from: {zip_url}")

            # Step 3: 下载并解压
            result_paths = self._download_and_extract(zip_url, pdf_path)
            exec_time = time.time() - start_time

            # 成功返回
            return ToolResult(
                success=True,
                data=result_paths,
                tool_name=self.name,
                execution_time=exec_time
            )

        except Exception as e:
            exec_time = time.time() - start_time
            self.logger.error(f"Execution failed: {str(e)}")
            return ToolResult(
                success=False,
                data={},
                tool_name=self.name,
                execution_time=exec_time,
                error=str(e)
            )

    def _upload_task(self, file_path: str) -> Optional[str]:
        url = f"{self.api_base_url}/file-urls/batch"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}"
        }
        file_name = os.path.basename(file_path)
        data = {
            # "enable_formula": True,
            # "language": "ch",
            "model_version": "vlm",
            "enable_table": True,
            "files": [
                {"name": file_name, "is_ocr": True, "data_id": "abcd"}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    batch_id = result["data"]["batch_id"]
                    upload_urls = result["data"]["file_urls"]
                    with open(file_path, 'rb') as f:
                        upload_resp = requests.put(upload_urls[0], data=f, timeout=60)
                    if upload_resp.status_code == 200:
                        return batch_id
                    else:
                        self.logger.error(f"PUT upload failed: {upload_resp.status_code}, {upload_resp.text}")
            else:
                self.logger.error(f"POST /file-urls/batch failed: {response.status_code}, {response.text}")
        except requests.RequestException as e:
            self.logger.error(f"Request error during upload: {str(e)}")
        return None

    def _poll_result(self, batch_id: str) -> Optional[str]:
        url = f"{self.api_base_url}/extract-results/batch/{batch_id}"
        headers = {
            "Authorization": f"Bearer {TOKEN}"
        }

        max_retries = int((self.timeout or 120.0) / self.poll_interval)
        for attempt in range(max_retries):
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code != 200:
                    self.logger.warning(f"Polling failed (attempt {attempt+1}): {res.status_code} {res.text}")
                    time.sleep(self.poll_interval)
                    continue

                data = res.json().get("data", {})
                extract_result = data.get("extract_result", [{}])[0]
                state = extract_result.get("state")

                if state == "done":
                    full_zip_url = extract_result.get("full_zip_url")
                    if full_zip_url:
                        return full_zip_url
                    else:
                        self.logger.error("State 'done' but no full_zip_url returned.")
                        return None
                elif state == "failed":
                    reason = extract_result.get("fail_reason", "Unknown")
                    self.logger.error(f"Extraction failed on server: {reason}")
                    return None
                else:
                    self.logger.debug(f"Extraction in progress: {state}")
                    time.sleep(self.poll_interval)

            except Exception as e:
                self.logger.error(f"Error polling result: {str(e)}")
                time.sleep(self.poll_interval)

        self.logger.error("Polling timed out waiting for extraction result.")
        return None

    def _download_and_extract(self, zip_url: str, source_pdf_path: str) -> Dict[str, str]:
        # 构建目标目录名：基于原始文件名
        pdf_name = Path(source_pdf_path).stem
        target_dir = os.path.join("extracted_results", f"{pdf_name}_mineru_output")
        os.makedirs(target_dir, exist_ok=True)

        # 下载文件名
        fname = os.path.basename(urlparse(zip_url).path)
        if not fname or "." not in fname:
            fname = "output.zip"
        filepath = os.path.join(target_dir, fname)

        self.logger.info(f"Downloading ZIP from {zip_url} to {filepath}")

        try:
            with requests.get(zip_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            self.logger.info("Download completed.")

            # 解压
            self.logger.info("Starting extraction...")
            if fname.endswith(".zip"):
                with zipfile.ZipFile(filepath, 'r') as zf:
                    zf.extractall(target_dir)
            elif fname.endswith((".tar.gz", ".tgz")):
                with tarfile.open(filepath, 'r:gz') as tf:
                    tf.extractall(target_dir)
            elif fname.endswith(".tar"):
                with tarfile.open(filepath, 'r') as tf:
                    tf.extractall(target_dir)
            else:
                raise ValueError(f"Unsupported archive format: {fname}")

            # 清理临时压缩包（可选）
            # os.remove(filepath)

            # 返回关键路径
            # markdown_file = os.path.join(target_dir, "full.md")
            # images_dir = os.path.join(target_dir, "images")

            return {
                "pdf_output_dir": target_dir,
                # "markdown_file": markdown_file,
                # "images_dir": images_dir,
                "archive_path": filepath,
                "source_pdf": source_pdf_path,
                "message": "MinerU extraction and unpacking completed."
            }

        except Exception as e:
            self.logger.error(f"Download or extraction failed: {str(e)}")
            raise


# chemistry_extraction/tools/paddle_ocr_tool.py

import os
import json
import time
from typing import Dict, Any, Optional
from .base_tool import BaseTool, ToolResult
from ..utils.bbox_utlis import crop_image, visualize_bboxes
from ..utils.api_utils import convert_image_to_bboxs, convert_image_to_smiles, convert_image_to_text
from ..config import Config
import copy

class YoloDetector(BaseTool):
    """
    MCP 工具：对识别出的smiles码进行校验
    """
    name: str = "yolo_detector"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.output_dir = Config.get("OUTPUT_DIR", "output")
        self.segment_dir = os.path.join(self.output_dir, "compound", "segments")
        self.visualization_dir = os.path.join(self.output_dir, "compound", "visualized")
        os.makedirs(self.segment_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.logger.info("YoloDetector instance initialized.")

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        try:
            # 解析输入
            self.logger.info(f"Running yolo detector tool")
            image_path = input_data["image_path"]
            results = convert_image_to_bboxs(image_path)
            bboxs = [res['bbox'] for res in results]
            visulized_img = visualize_bboxes(image_path, bboxs, os.path.join(self.visualization_dir, "visualized_" + os.path.basename(image_path)))
            detect_results = []
            
            copounds = [result for result in results if result['class_id'] == 0]
            if len(copounds) == 0:
                exec_time = time.time() - start_time
                self.logger.info("No compounds detected by Yolo.")
                return ToolResult(
                    success=True,
                    data=[],
                    tool_name=self.name,
                    execution_time=exec_time
                )
            
            for idx, result in enumerate(results):
                fomated_reuslt = copy.deepcopy(result)
                bbox = result['bbox']
                basename, ext = os.path.splitext(os.path.basename(image_path))
                ext = os.path.splitext(image_path)[1].lower()
                crop_filename = f"cmp_{basename}_{idx+1}_{ext}"
                crop_path = os.path.join(self.segment_dir, crop_filename)
                saved_path, W, H = crop_image(image_path, bbox, crop_path)
                fomated_reuslt['name'] = ""
                if result['class_id'] == 0:  # 如果是化合物结构，尝试转换为smiles
                    fomated_reuslt['smiles'] = convert_image_to_smiles(saved_path)
                    fomated_reuslt['name'] = ""
                else:
                    fomated_reuslt['smiles'] = ""
                    fomated_reuslt['name'] = convert_image_to_text(saved_path)
                fomated_reuslt['crop_path'] = crop_path
                fomated_reuslt['image_resolution'] = [W, H]
                fomated_reuslt['original_image'] = image_path
                fomated_reuslt['bbox_id'] = idx + 1
                fomated_reuslt['visualized_image'] = visulized_img
                detect_results.append(fomated_reuslt)
                
            # 执行 Yolo 推理
            exec_time = time.time() - start_time
            print(f"Yolo detection found {len(detect_results)} compounds.")
            return ToolResult(
                success=True,
                data=detect_results,
                tool_name=self.name,
                execution_time=exec_time
            )

        except Exception as e:
            exec_time = time.time() - start_time
            self.logger.error(f"OCR execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return ToolResult(
                success=False,
                data={},
                tool_name=self.name,
                execution_time=exec_time,
                error=str(e)
            )


def run_yolo_detector(config: Dict[str, Any] = None) -> ToolResult:
    """
    快捷函数：直接运行 OCR 工具
    """
    tool = YoloDetector(config=config)
    return tool.execute({"image_path": "/mnt/binghao/38f4e387-c730-4ed5-9570-4679ff213785.png"})

if __name__ == "__main__":
    # 测试代码

    results = run_yolo_detector()
    print(results.data)
# chemistry_extraction/state.py
from typing import TypedDict, List, Dict, Any, Optional
from typing import Annotated
from operator import add

def merge_dict(src_dict: Dict[str, Any], dst_dict: Dict[str, Any]):
    merged_dict = {}
    
    for k,v in src_dict.items():
        merged_dict[k] = v
    for k,v in dst_dict.items():
        merged_dict[k] = v
    
    return merged_dict


class ChemistryExtractionState(TypedDict):
    # 原始输入
    raw_text: str
    pdf_path: str
    markdown_file: str
    pdf_output_dir: str
    user_request: str

    # 文本处理结果（用 add 自动 extend）
    text_jsons: Annotated[List[Dict[str, Any]], add]
    text_extractions: Annotated[List[Dict[str, Any]], add]

    # 图像相关
    image_jsons: Annotated[List[Dict[str, Any]], add]
    image_extractions: Annotated[List[Dict[str, Any]], add]
    
    yolo_detections: Annotated[List[Dict[str, Any]], add]
    compounds: Annotated[List[Dict[str, Any]], add]

    # 融合结果（通常只有一个，不合并）
    fusion_result: Optional[Dict[str, Any]]
    extported_result: Optional[Dict[str, Any]]
    # 处理状态
    current_stage: Annotated[List[str], add]
    errors: Annotated[List[str], add]  # 错误日志追加
    metadata: Annotated[Dict[str, Any], merge_dict]  # dict 默认用 |= 合并（Python 3.9+）

    tool_results: Dict[str, Any]

    # 可选：支持动态字段（非必须）
    _extra_fields: Dict[str, Any]


from typing import Dict, Any, List, Tuple
import math
import json
import time
from typing import Dict, Any, List, Optional, Tuple

from ..state import ChemistryExtractionState
from . import BaseAgent
from ..config import Config
import copy

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2
import logging

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_bbox_center(bbox: BBox) -> Tuple[float, float]:
    """获取边界框中心点 (x, y)"""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点间欧氏距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_below(compound_center: Tuple[float, float], text_center: Tuple[float, float], threshold=50) -> bool:
    """判断文本是否在化合物下方"""
    cx, cy = compound_center
    tx, ty = text_center
    vertical_diff = ty - cy
    horizontal_diff = abs(tx - cx)
    return vertical_diff > 0 and vertical_diff < threshold and horizontal_diff < threshold * 0.7


def is_above(compound_center: Tuple[float, float], text_center: Tuple[float, float], threshold=50) -> bool:
    """判断文本是否在上方"""
    cx, cy = compound_center
    tx, ty = text_center
    vertical_diff = ty - cy
    horizontal_diff = abs(tx - cx)
    return vertical_diff < 0 and abs(vertical_diff) < threshold and horizontal_diff < threshold * 0.7


def is_right_of(compound_center: Tuple[float, float], text_center: Tuple[float, float], threshold=50) -> bool:
    """判断是否在右侧"""
    cx, cy = compound_center
    tx, ty = text_center
    horizontal_diff = tx - cx
    vertical_diff = abs(ty - cy)
    return horizontal_diff > 0 and horizontal_diff < threshold and vertical_diff < threshold * 0.7


def is_left_of(compound_center: Tuple[float, float], text_center: Tuple[float, float], threshold=50) -> bool:
    """判断是否在左侧"""
    cx, cy = compound_center
    tx, ty = text_center
    horizontal_diff = tx - cx
    vertical_diff = abs(ty - cy)
    return horizontal_diff < 0 and abs(horizontal_diff) < threshold and vertical_diff < threshold * 0.7


def calculate_score(
    compound_center: Tuple[float, float],
    text_center: Tuple[float, float],
    dist_weight: float = 0.6,
    below_weight: float = 0.4
) -> float:
    """
    计算匹配得分
    :param compound_center: 化合物中心
    :param text_center: 文本中心
    :param dist_weight: 距离权重（0~1）
    :param below_weight: 下方加权系数（0~1）
    :return: 得分 [0, 1]，越高越好
    """
    # 1. 距离项：越近得分越高
    dist = euclidean_distance(compound_center, text_center)
    max_dist = 300  # 可配置，根据图像分辨率调整
    distance_score = max(0, 1 - min(dist / max_dist, 1))

    # 2. 方向项：只有“下方”才加分
    direction_score = 0.0
    if is_below(compound_center, text_center):
        direction_score = below_weight
    elif is_above(compound_center, text_center):
        direction_score = below_weight * 0.3
    elif is_right_of(compound_center, text_center):
        direction_score = below_weight * 0.2
    elif is_left_of(compound_center, text_center):
        direction_score = below_weight * 0.1

    # 总分 = 距离权重 × 距离分 + 方向权重 × 方向分
    total_score = dist_weight * distance_score + direction_score
    return total_score


def match_compound_name(
    compound_detections: List[Dict],
    name_detections: List[Dict],
    dist_weight: float = 0.6,
    below_weight: float = 0.4,
    min_confidence: float = 0.3
) -> List[Dict[str, Any]]:
    """
    使用全局最优策略匹配化合物与其名称：
    1. 计算所有 (compound, name) 对的匹配分数
    2. 按分数从高到低排序
    3. 依次配对，跳过已匹配的 compound 或 name
    """
    logger.info(f"Starting global optimal matching: {len(compound_detections)} compounds, {len(name_detections)} names")

    # Step 1: 构建所有有效匹配对及其分数
    all_pairs = []
    for comp in compound_detections:
        if not isinstance(comp.get("bbox"), (list, tuple)) or len(comp["bbox"]) != 4:
            logger.warning(f"Invalid compound bbox: {comp}")
            continue
        compound_id = comp["bbox_id"]
        compound_center = get_bbox_center(comp["bbox"])
        
        for name_det in name_detections:
            if not isinstance(name_det.get("bbox"), (list, tuple)) or len(name_det["bbox"]) != 4:
                continue
            name_id = name_det["bbox_id"]
            name_center = get_bbox_center(name_det["bbox"])
            score = calculate_score(compound_center, name_center, dist_weight, below_weight)
            
            if score >= min_confidence:
                name_text = name_det.get("text") or name_det.get("name", "")
                all_pairs.append({
                    "compound_id": compound_id,
                    "name_id": name_id,
                    "name": name_text,
                    "confidence": round(score, 3),
                    "score": score  # 用于排序
                })

    # Step 2: 按分数降序排序
    all_pairs.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: 贪心选择（但基于全局排序）
    matches = []
    used_compound_ids = set()
    used_name_ids = set()

    for pair in all_pairs:
        cid = pair["compound_id"]
        nid = pair["name_id"]
        if cid in used_compound_ids or nid in used_name_ids:
            continue  # 跳过已匹配的
        # 接受该匹配
        matches.append({
            "compound_id": cid,
            "name_id": nid,
            "name": pair["name"],
            "confidence": pair["confidence"]
        })
        used_compound_ids.add(cid)
        used_name_ids.add(nid)
        logger.debug(f"Global match: Compound {cid} ↔ Name {nid} ('{pair['name']}') score={pair['score']:.3f}")

    logger.info(f"Global matching complete. Found {len(matches)} matches.")
    return matches


class CompoundNameAgent(BaseAgent):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = Config.get("IMAGE_ANALYSIS_MODEL")
    
    def process(self, state: ChemistryExtractionState):
        """处理文本提取的主要逻辑"""
        try:
            metadata = {}
            metadata['text_agent_start'] = time.time()
            current_stage = ["text_processing"]

            compound_detections = state["yolo_detections"] 
            self.logger.info(f"Found {len(compound_detections)} sections to process")

            
            # 处理每个相关部分
            compound_name_extractions = []
            for i, section in enumerate(compound_detections):
                extraction = copy.deepcopy(section)
                self.logger.info(f"Processing section {i+1}/{len(compound_detections)}")
                input_data = [{"bbox": det["bbox"], "class_id": det["class_id"], "bbox_id": det["bbox_id"], "name": det["name"]} for det in section["detect"]]
                if len(input_data) == 0:
                    self.logger.warning(f"No detection data found in section {i+1}, skipping.")
                    continue
                section_str = json.dumps(input_data, ensure_ascii=False)
                llm_result = self._extract_from_section(section_str)
                if llm_result is None or not isinstance(llm_result, list):
                    self.logger.warning(f"LLM extraction returned invalid result for section {i+1}: {llm_result}")
                    self.logger.warning(f"LLM extraction failed for section {i+1}, skipping.")
                    continue
                for item in llm_result:
                    for detect in extraction["detect"]:
                        if item['compound_id'] == detect['bbox_id'] or item['name_id'] == detect['bbox_id']:
                            detect['name'] = item.get('name', '')
                compound_name_extractions.append(extraction)
            # 更新状态
            metadata['copmpund_name_extractions_count'] = len(compound_name_extractions)
            metadata['copmpund_name_agent_end'] = time.time()

            self.logger.info(f"Successfully extracted {len(compound_name_extractions)} copmpund_names")
            return {
                'compounds': compound_name_extractions,
                'current_stage': current_stage,
                'metadata': metadata
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.handle_error(state, e, "text_extraction_process")
    
    def _extract_from_section(self, section: str):
        try:
            # 将输入数据解析为字典列表
            input_data = json.loads(section)
            # 分离化合物和名称检测
            compound_detections = [d for d in input_data if d["class_id"] == 0]
            name_detections = [d for d in input_data if d["class_id"] == 5]
            print(f"Compound detections: {compound_detections}")
            print(f"Name detections: {name_detections}")
            if not compound_detections or not name_detections:
                self.logger.warning(f"No valid detections found in section.")
                return []

            # 执行匹配（可调节权重）
            matches = match_compound_name(
                compound_detections=compound_detections,
                name_detections=name_detections,
                dist_weight=0.6,      # 距离占 60%
                below_weight=0.4,     # 下方占 40%
                min_confidence=0.0    # 最低置信度
            )

            # 打印匹配结果
            self.logger.info(f"Found {len(matches)} compound-name matches via geometric rules.")

            return matches
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Failed to match compound names: {str(e)}")
            return None

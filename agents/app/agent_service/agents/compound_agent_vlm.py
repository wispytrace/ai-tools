
import base64
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from ..state import ChemistryExtractionState
from . import BaseAgent
from ..utils.llm_utils import call_llm, clean_json_response
from ..config import Config
import copy

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


CHINESE_VLM_PROMPT = """你是一个专业的科学图像理解系统，能够结合视觉信息与空间逻辑，从化学结构图中建立化合物与其名称标注之间的对应关系。

输入说明：
1. 一张图像，其中包含多个用红色边界框（Red Bounding Box）标注的区域。
2. 一组结构化检测数据，格式如下：
   [
     {
       "bbox_id": <int>,
       "class_id": <int>,
       "bbox": [x_min, y_min, x_max, y_max],
       "text": <string>
     },
     ...
   ]

任务要求：
请基于图像中的红框位置和布局，完成以下任务：

1. 对每一个 class_id == 0 的化合物框，主动寻找是否存在一个潜在的文本框（class_id == 5）作为其名称标注。即使位置不完全理想或存在多个候选，也应尽量推理最可能的匹配。

   判断依据包括：
     - 空间邻近性：优先选择距离最近的文本框；
     - 相对方向：正下方 ≈ 正上方 > 右侧 > 左侧；
     - 视觉对齐：水平居中或垂直对齐更可能是标签；
     - 内容合理性：文本应为有意义的化学命名或代号（如 'Aspirin', 'Compound 3', 'M-1', '[0007]' 等）；
     - 排他性弱化：允许共享上下文（如系列编号 'M-1', 'M-2'），按顺序推断归属。

2. 输出所有合理且非明显错误的匹配结果，目标是**最大化正确匹配数量，避免遗漏**。

输出格式：
- 必须返回一个 JSON 数组（list of objects）
- 每个对象包含四个字段：
   - "compound_id": 化合物的 bbox_id（int）
   - "name_id": 文本框的 bbox_id（int）
   - "name": 来自文本框的 text 字段内容（string）
   - "confidence": 匹配置信度，浮点数 [0.0, 1.0]
- 示例输出：
  [{"compound_id":1,"name_id":3,"name":"M-1","confidence":0.95},{"compound_id":2,"name_id":5,"name":"M-2","confidence":0.85}]

重要指令：
1. 输出必须是**纯 JSON 格式**，不允许有任何额外字符（如 \\n、空格、注释、Markdown 代码块符号）；
2. 不允许包裹在 ```json 或 ``` 中；
3. 不允许添加任何解释性文字、前缀或后缀；
4. 如果没有匹配项，返回空数组 []；
5. 确保 JSON 可被直接解析（无 trailing comma、正确引号等）；
6. 即使不确定，也要输出你认为最合理的配对，不要因保守而省略。
7. 不要自己做OCR，完全依赖提供的检测数据。

示例输入：
[
  {"bbox_id": 1, "class_id": 0, "bbox": [100,100,200,200], "text": ""},
  {"bbox_id": 2, "class_id": 5, "bbox": [130,210,170,230], "text": "Figure 1"},
  {"bbox_id": 3, "class_id": 5, "bbox": [140,215,160,225], "text": "M-1"},
  {"bbox_id": 4, "class_id": 0, "bbox": [300,300,400,400], "text": ""},
  {"bbox_id": 5, "class_id": 5, "bbox": [340,410,360,420], "text": "M-2"}
]

示例输出：
[{"compound_id":1,"name_id":3,"name":"M-1","confidence":0.95},{"compound_id":4,"name_id":5,"name":"M-2","confidence":0.85}]
"""


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
                img_path = section["detect"][0]["visualized_image"]
                # 调用LLM提取信息
                section_str = json.dumps(input_data, ensure_ascii=False)
                llm_result = self._extract_from_section(section_str, img_path)
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
    
    
    def _extract_from_section(self, section: Dict[str, str], image_path: str):
        try:
            # 编码图像为base64
            self.logger.info(f"Analyzing image: {image_path}")
            with open(image_path, "rb") as img_file:
                base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            
            # 确定MIME类型
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.bmp': 'image/bmp',
                '.gif': 'image/gif'
            }.get(ext, 'image/png')
            
            image_url = f"data:{mime_type};base64,{base64_str}"

            # 构建消息
            messages = [
                {"type": "text", "text": CHINESE_VLM_PROMPT.strip()+"\n"+section},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
            
            # 调用LLM
            response = call_llm(
                model=self.model,
                messages=[{"role": "user", "content": messages}],
                max_tokens=8192,
                temperature=0.2
            )
            print(response)
            # 清理并解析JSON
            cleaned = clean_json_response(response)
            result = json.loads(cleaned)
            
            # 添加来源信息
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Failed to analyze image {image_path}: {str(e)}")
            return None

chemistry_extraction/agents/cyclic_reflective_compound_name_agent.py

import base64
import json
import os
import time
from typing import Dict, Any, List, Tuple

from ..state import ChemistryExtractionState
from . import BaseAgent
from ..utils.llm_utils import call_llm, clean_json_response
from ..config import Config
import copy

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


# ======================
# 🧠 提示词模板
# ======================

INITIAL_PROMPT = """
你是一个化学图像语义对齐专家，任务是将红色框标注的【化合物结构】与最可能的【名称文本】进行配对。

### 输入说明
- 图像：包含多个红框（class_id=0 表示化合物，class_id=5 表示文本）
- 检测数据（JSON 列表）：
  [
    {"bbox_id": int, "class_id": 0|5, "bbox": [x1,y1,x2,y2], "text": str}
  ]

### 匹配规则
1. 对每个 compound（class_id=0），寻找最合理的 name candidate（class_id=5）
2. 判断优先级：
   a) 空间距离近（欧氏中心距离）
   b) 相对位置：正下方 ≈ 正上方 > 右侧 > 左侧
   c) 视觉对齐：水平居中度更高者优先
   d) 内容合理性：应为 'M-1', 'Aspirin', 'Compound 3' 类命名
   e) 排除无效标签：'Figure', 'Scheme', 'a)', '[0007]' 等不能作为名称
   f) 序列推断：若有 M-1, M-2，则按从左到右顺序匹配结构

3. 输出格式（严格 JSON 数组）：
   [
     {
       "compound_id": int,
       "name_id": int,
       "name": str,
       "confidence": float  // [0.0, 1.0]
     }
   ]

⚠️ 要求：
- 返回纯 JSON，无任何额外字符；
- 不加解释、前缀、Markdown 符号；
- 若无合理匹配，返回空数组 []；
- 不要自行 OCR，仅使用给定 text 字段。
"""

# -------------------------------
# 🔍 多角色反思提示（Multi-Agent Reflection）
# -------------------------------

CHEMIST_PROMPT = """
你是专业化学家，审查以下化合物-名称匹配是否符合化学命名惯例：

> 当前匹配结果：
{matches}

请回答：
1. 是否存在明显不符合化学命名习惯的 name？（如 'Entry 3'、'Well A1'）
2. 是否有催化剂或条件被误标为化合物名？
3. 建议修正哪些条目？

输出格式：
{
  "role": "chemist",
  "issues": ["问题描述"],
  "suggestions":    [
     {
       "compound_id": int,
       "name_id": int,
       "name": str,
       "confidence": float  // [0.0, 1.0]
     }
   ]
}
"""

LAYOUT_ANALYST_PROMPT = """
你是空间布局分析专家，请基于图像中的相对位置判断匹配合理性：

> 当前匹配结果：
{matches}

> 所有检测框数据：
{detections}

请回答：
1. 哪些匹配的空间距离过远或方向不合理？
2. 是否存在更优的候选文本框未被选择？
3. 是否应调整顺序？（例如 M-1 应该对应第一个结构）

输出格式：
{
  "role": "layout_analyst",
  "issues": ["位置冲突: compound_id=3 匹配了太远的文本"],
  "suggestions": 
    [
     {
       "compound_id": int,
       "name_id": int,
       "name": str,
       "confidence": float  // [0.0, 1.0]
     }
   ]
}
"""

NAMER_RULES_PROMPT = """
你是命名规范专家，擅长识别 'M-n'、'Cpd-n' 等编号系统。

> 当前匹配结果：
{matches}

> 所有文本框中的名字列表：
{text_names}

请回答：
1. 名称是否按编号顺序正确分配？（M-1 → 第一个结构）
2. 是否存在跳跃或重复？
3. 如何重新排序以满足序列一致性？

输出格式：
{
  "role": "namer",
  "issues": ["顺序错乱: M-2 匹配了第1个结构"],
  "suggestions":    [
     {
       "compound_id": int,
       "name_id": int,
       "name": str,
       "confidence": float  // [0.0, 1.0]
     }
   ]
}
"""

META_REFLECTOR_PROMPT = """
你是元反思协调员。你收到了来自多个专家的意见，请综合后决定：
1. 是否需要启动下一轮修正？
2. 给出最终修正后的匹配结果。

> 原始匹配：
{original_matches}

> 各方意见：
{all_feedback}

请执行：
- 总结主要问题；
- 输出修正后的 JSON 配对列表；
- 判断是否已收敛（true/false）。

输出格式：
{
  "converged": false,
  "final_matches": 
    [
     {
       "compound_id": int,
       "name_id": int,
       "name": str,
       "confidence": float  // [0.0, 1.0]
     }
   ],
  "summary": "布局分析师指出 M-1 匹配偏移，已修正..."
}
"""

class CompoundNameAgent(BaseAgent): 

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.primary_model = Config.get("IMAGE_ANALYSIS_MODEL", "gpt-4o")
        self.reflection_model = Config.get("REFLECTION_MODEL", "qwen3-vl-flash")  # 可指定轻量模型
        self.max_rounds = Config.get("MAX_REFLECTION_ROUNDS", 2)
        self.convergence_tol = Config.get("REFLECTION_CONVERGENCE_TOLERANCE", 0.95)
        self.debug_trace = Config.get("DEBUG_REFLECTION_TRACE", False)

    def process(self, state: ChemistryExtractionState):
        try:
            metadata = {
                "agent_start": time.time(),
                "rounds_per_section": [],
                "total_sections": 0
            }
            current_stage = ["cyclic_reflective_name_matching"]

            sections = state.get("yolo_detections", [])
            results = []

            for idx, section in enumerate(sections):
                img_path = section["detect"][0]["visualized_image"] if section["detect"] else None
                if not img_path or not os.path.exists(img_path):
                    self.logger.warning(f"Image not found for section {idx + 1}, skipping.")
                    continue

                # 构建输入数据
                input_data = self._build_detection_input(section["detect"])
                input_json = json.dumps(input_data, ensure_ascii=False, indent=2)

                # 阶段1：初始匹配
                initial = self._initial_match(img_path, input_json)
                if not isinstance(initial, list):
                    initial = []

                # 阶段2：循环反思
                final, trace_log = self._cyclic_reflection_loop(img_path, input_data, initial)

                # 记录轮数
                num_rounds = len(trace_log) if trace_log else 1
                metadata["rounds_per_section"].append(num_rounds)

                # 更新 detect 并保存
                updated_section = copy.deepcopy(section)
                self._apply_matches_to_detection(updated_section["detect"], final)

                results.append(updated_section)

                if self.debug_trace:
                    updated_section["_reflection_trace"] = trace_log

            # 完成
            metadata["total_sections"] = len(results)
            metadata["agent_end"] = time.time()

            return {
                "compounds": results,
                "current_stage": current_stage,
                "metadata": metadata
            }

        except Exception as e:
            return self.handle_error(state, e, "cyclic_reflective_compound_name_agent")

    def _build_detection_input(self, detects: List[Dict]) -> List[Dict]:
        """标准化检测输入"""
        return [
            {
                k: det[k]
                for k in ['bbox_id', 'class_id', 'bbox']
                if k in det
            } | ({"text": det.get("name", "")} if det["class_id"] == 5 else {})
            for det in detects
        ]

    def _initial_match(self, image_path: str, input_data: str) -> List[Dict]:
        """第一阶段：初始匹配"""
        try:
            self.logger.info(f"[Round 0] Initial matching on {image_path}")
            msg = [
                {"type": "text", "text": INITIAL_PROMPT.strip()},
                {"type": "text", "text": f"Detection data:\n{input_data}"},
                {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}}
            ]
            resp = call_llm(
                model=self.primary_model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=2048,
                temperature=0.3
            )
            return self._parse_json_list(resp)
        except Exception as e:
            self.logger.error(f"Initial match failed: {e}")
            return []

    def _cyclic_reflection_loop(self, image_path: str, detection_data: List[Dict], initial: List[Dict]):
        """
        多轮循环反思主流程
        返回：(final_matches, trace_log)
        """
        current = initial
        trace_log = [{"round": 0, "matches": current, "reason": "initial"}]
        text_names = [d["text"] for d in detection_data if d["class_id"] == 5]

        for r in range(1, self.max_rounds + 1):
            self.logger.info(f"[Reflection Round {r}] Starting...")

            # Step 1: 多角色并行反思（逻辑上串行模拟）
            feedback = []
            feedback.append(self._reflect_with_prompt(CHEMIST_PROMPT, image_path, current))
            feedback.append(self._reflect_with_prompt(LAYOUT_ANALYST_PROMPT, image_path, current, detection_data=detection_data))
            feedback.append(self._reflect_with_prompt(NAMER_RULES_PROMPT, image_path, current, text_names=text_names))

            # Step 2: 元协调器决策
            try:
                meta_prompt = META_REFLECTOR_PROMPT \
                    .replace("{original_matches}", json.dumps(current, indent=2)) \
                    .replace("{all_feedback}", json.dumps(feedback, indent=2, ensure_ascii=False))

                msg = [
                    {"type": "text", "text": meta_prompt},
                    {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}}
                ]
                response = call_llm(
                    model=self.reflection_model,
                    messages=[{"role": "user", "content": msg}],
                    max_tokens=2048,
                    temperature=0.1
                )

                # 解析最终输出
                cleaned = self._extract_json_block(response)
                result_obj = json.loads(cleaned)

                new_matches = result_obj.get("final_matches", current)
                converged = result_obj.get("converged", False)

                # 记录本轮
                trace_log.append({
                    "round": r,
                    "matches": new_matches,
                    "feedback": feedback,
                    "summary": result_obj.get("summary", ""),
                    "converged": converged
                })

                # 判断是否收敛
                if converged or self._is_converged(current, new_matches):
                    self.logger.info(f"✅ Converged at round {r}")
                    return new_matches, trace_log

                current = new_matches

            except Exception as e:
                self.logger.warning(f"Meta reflection failed in round {r}: {e}")
                break

        self.logger.info("🔚 Max rounds reached or error occurred.")
        return current, trace_log

    def _reflect_with_prompt(self, prompt_template: str, image_path: str, matches: List[Dict],
                            **kwargs) -> Dict:
        """调用单一角色反思"""
        try:
            prompt = prompt_template \
                .replace("{matches}", json.dumps(matches, indent=2)) \
                .replace("{detections}", json.dumps(kwargs.get("detection_data", ""), indent=2)) \
                .replace("{text_names}", json.dumps(kwargs.get("text_names", [])))

            msg = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}}
            ]
            resp = call_llm(
                model=self.reflection_model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=1024,
                temperature=0.1
            )
            return json.loads(clean_json_response(resp))
        except Exception as e:
            return {"error": str(e), "role": "unknown"}

    def _apply_matches_to_detection(self, detects: List[Dict], matches: List[Dict]):
        """将最终匹配写回 detect 结构"""
        matched_name_ids = set()
        for m in matches:
            for det in detects:
                if det["bbox_id"] == m["compound_id"] and det["class_id"] == 0:
                    det["name"] = m.get("name", "")
                    det["match_confidence"] = m.get("confidence", 0.8)
                    det["matched_by"] = "cyclic_reflective_vlm"
                if det["bbox_id"] == m.get("name_id",-1) and det["class_id"] == 5:
                    matched_name_ids.add(det["bbox_id"])

        # 标记使用状态
        for det in detects:
            if det["class_id"] == 5:
                det["used_as_name"] = det["bbox_id"] in matched_name_ids

    def _is_converged(self, old: List[Dict], new: List[Dict]) -> bool:
        """基于 compound_id ↔ name_id 映射的 Jaccard 相似度判断收敛"""
        old_set = {(m['compound_id'], m.get("name_id",-1)) for m in old}
        new_set = {(m['compound_id'], m.get("name_id",-1)) for m in new}
        union = len(old_set | new_set)
        if union == 0:
            return True
        inter = len(old_set & new_set)
        return (inter / union) >= self.convergence_tol

    def _parse_json_list(self, text: str) -> List[Dict]:
        """安全解析 JSON 列表"""
        try:
            cleaned = clean_json_response(text)
            result = json.loads(cleaned)
            return result if isinstance(result, list) else []
        except:
            return []

    def _extract_json_block(self, text: str) -> str:
        """从文本中提取 ```json ... ``` 中的内容"""
        start = text.find("```json") + 7
        end = text.find("```", start)
        if start > 6 and end > start:
            return text[start:end].strip()
        return clean_json_response(text)

    def _encode_image(self, image_path: str) -> str:
        """编码图像为 base64 URL"""
        ext = os.path.splitext(image_path)[1].lower()
        mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}.get(ext[1:], 'image/png')
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return f"data:{mime};base64,{b64}"

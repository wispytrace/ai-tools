# chemistry_extraction/agents/image_agent.py
import base64
import json
import os
import time
from typing import Dict, Any, List, Optional
from ..state import ChemistryExtractionState
from ..agents import BaseAgent
from ..utils.llm_utils import call_llm, clean_json_response
from ..config import Config
import copy

# --- 精简后的主 Prompt ---
INITIAL_PROMPT = """
你是一个专业的化学图像语义解析助手。请根据提供的【化学图像】和【检测框元数据】，判断是否存在化学反应，并提取结构化信息。

### 输入说明
1. 图像：包含化学结构式、箭头、文本标签等。
2. 检测框列表（JSON）：
   [
     { "bbox_id": int, "class_id": 0|5, "bbox": [x1,y1,x2,y2], "text/smiles": str }
   ]
   - class_id: 0=化合物, 5=文本

### 解析逻辑

#### 1. 是否存在反应？
- 查找视觉上的反应箭头（→, ⇒, ⟶ 等）
- 若有箭头连接多个化合物 → 视为反应
- 否则仅作为独立化合物集合处理（reaction_smiles = null）

#### 2. 提取字段
- **reactants/products**: 来自 class_id=0 的 smiles，name 尽量关联最近文本（优先下方/侧方）
- **catalysts**: 箭头附近标注的催化剂名称（如 Pd/C）
- **conditions**: 温度、溶剂、时间等环境参数（如 "80°C", "in DMF"）
- **reaction_smiles**: 构造为 `r1.r2>>p1`；若无反应则为 null
- **symbolic_groups**: 记录 [*], [R], Boc 等通配符含义

> ⚠️ 所有 SMILES 必须保留立体化学（@/@@）和占位符（*）

### 输出格式（必须严格遵守）
{
  "reactants":    [{"name": "", "smiles": ""}],
  "products":     [{"name": "", "smiles": ""}],
  "catalysts":    [""],
  "conditions":   [""],
  "reaction_smiles": "",
  "symbolic_groups": [{"symbol": "", "meaning": ""}],
  "source_image": ""
}
若未发现反应，所有数组留空，reaction_smiles 设为 null。
"""

# --- 反思 Prompt ---
REFLECTION_PROMPT = """
你是该化学图像解析系统的反思模块。以下是某个图像的初始解析结果，请结合以下原则进行审查与修正：

### 审查重点
1. 【命名一致性】  
   - 化合物 name 是否合理？是否错误地将条件文本当作名字？
   - 示例错误："80 °C" 被误标为某化合物名 → 应移入 conditions

2. 【反应方向性】  
   - 箭头是否明确？反应物/产物顺序是否正确（通常左→右）？

3. 【催化剂/条件归属】  
   - 催化剂是否紧邻箭头？避免把试剂当催化剂
   - 条件是否完整？常见遗漏：气氛、光照、时间

4. 【符号解释完整性】  
   - 是否遗漏了 [*], R, Ar, Bn 等常见符号定义？

5. 【SMILES 合法性】  
   - smiles 字段是否包含完整结构？是否有明显截断？

6. 【空值处理】  
   - 无反应时 reaction_smiles 应为 null，不是空字符串

### 任务
- 指出原结果中的问题
- 输出修正后的 JSON（格式完全一致）

> 原始结果：
{original_result}

> 请输出反思报告 + 修正结果（用 ```json ... ``` 包裹）：
"""


class ImageAnalysisAgent(BaseAgent):
    """具有反思能力的图像分析智能体"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.primary_model = Config.get("IMAGE_ANALYSIS_MODEL")
        self.reflection_model = Config.get("REFLECTION_MODEL", self.primary_model)  # 可指定轻量模型
        self.enable_reflection = Config.get("ENABLE_REFLECTION", False)

    def process(self, state: ChemistryExtractionState):
        try:
            metadata = {}
            metadata['image_agent_start'] = time.time()
            current_stage = ["image_processing"]
            image_extractions = []

            for i, section in enumerate(state["yolo_detections"]):
                detections = section["detect"]
                if not detections:
                    self.logger.warning(f"No detection data in section {i+1}, skipping.")
                    continue

                image_path = detections[0]["visualized_image"]
                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}")
                    continue

                # 构建输入数据
                input_data = [
                    {k: det[k] for k in ['bbox_id', 'class_id', 'bbox', 'name', 'smiles']}
                    for det in detections
                ]
                input_data_str = json.dumps(input_data, ensure_ascii=False, indent=2)

                # 阶段1：初始分析
                raw_result = self._analyze_single_image(image_path, input_data_str)

                if not raw_result:
                    continue

                # 阶段2：反思修正（可选）
                if self.enable_reflection:
                    final_result = self._reflect_on_result(image_path, raw_result)
                else:
                    final_result = raw_result

                # 添加来源
                final_result["source_image"] = os.path.basename(image_path)

                # 保存结果
                image_extraction = copy.deepcopy(section)
                image_extraction['ocr_result'] = final_result
                image_extractions.append(image_extraction)

            # 更新状态
            metadata['image_extractions_count'] = len(image_extractions)
            metadata['image_agent_end'] = time.time()

            self.logger.info(f"Image-agent: Processed {len(image_extractions)} images with reflection={self.enable_reflection}")
            return {
                'image_extractions': image_extractions,
                'current_stage': current_stage,
                'metadata': metadata
            }

        except Exception as e:
            return self.handle_error(state, e, "reflective_image_analysis")

    def _analyze_single_image(self, image_path: str, input_data: str) -> Dict:
        """第一阶段：初步图像理解"""
        try:
            self.logger.info(f"[Phase 1] Analyzing image: {image_path}")
            base64_img = self._encode_image(image_path)
            messages = [
                {"type": "text", "text": INITIAL_PROMPT.strip()},
                {"type": "text", "text": f"Detection metadata:\n{input_data}"},
                {"type": "image_url", "image_url": {"url": base64_img}}
            ]

            response = call_llm(
                model=self.primary_model,
                messages=[{"role": "user", "content": messages}],
                max_tokens=4096,
                temperature=0.2
            )

            cleaned = clean_json_response(response)
            result = json.loads(cleaned)
            return result

        except Exception as e:
            self.logger.error(f"Initial analysis failed for {image_path}: {str(e)}")
            return self._empty_result()

    def _reflect_on_result(self, image_path: str, initial_result: Dict) -> Dict:
        """第二阶段：反思与修正"""
        try:
            self.logger.info(f"[Phase 2] Reflecting on initial result for: {image_path}")

            # 序列化原结果用于提示
            original_json = json.dumps(initial_result, ensure_ascii=False, indent=2)
            reflection_prompt = REFLECTION_PROMPT.replace("{original_result}", original_json)

            base64_img = self._encode_image(image_path)
            messages = [
                {"type": "text", "text": reflection_prompt},
                {"type": "image_url", "image_url": {"url": base64_img}}
            ]

            response = call_llm(
                model=self.reflection_model,
                messages=[{"role": "user", "content": messages}],
                max_tokens=2048,
                temperature=0.1  # 更确定
            )

            # 提取最终JSON（可能包裹在文本中）
            start = response.find("```json") + 7
            end = response.find("```", start)
            if start > 6 and end > start:
                repaired = response[start:end].strip()
            else:
                repaired = clean_json_response(response)

            final_result = json.loads(repaired)
            self.logger.debug(f"Reflection applied. Success.")

            return final_result

        except Exception as e:
            self.logger.warning(f"Reflection failed, falling back to initial result: {str(e)}")
            return initial_result  # 失败则保留原结果

    def _encode_image(self, image_path: str) -> str:
        """编码图像为base64 URL"""
        ext = os.path.splitext(image_path)[1].lower()
        mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}.get(ext[1:], 'image/png')
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return f"data:{mime};base64,{b64}"

    def _empty_result(self) -> Dict:
        """返回空但格式正确的默认结果"""
        return {
            "reactants": [],
            "products": [],
            "catalysts": [],
            "conditions": [],
            "reaction_smiles": None,
            "symbolic_groups": [],
            "source_image": ""
        }

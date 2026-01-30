# chemistry_extraction/agents/fusion_agent.py
import json
import re
import time
from typing import Dict, Any, List, Optional
from ..state import ChemistryExtractionState
from ..agents import BaseAgent
from ..utils.llm_utils import call_llm, clean_json_response, robust_json_parse
from ..config import Config
FUSION_PROMPT = """
你是化学反应融合引擎，严格以图像解析结果为主干，文本仅用于补充字段。

# 输入
【图像解析】@image_analysis@
【文本解析】@text_analysis@

# 核心规则
1. 【主干唯一来源：图像】
   - 所有主反应来自 `image_analysis` 中：
     - `diagram_type != null` 的区域
     - `ocr_result` 中识别出的反应式（如 A + B → C）
   - 每个有效 OCR 条目必须进入 `reactions`，即使无法生成 SMILES（设 rxn_smiles=null）

2. 【文本 = 补丁工具】
   - 禁止从文本中主动提取任何新反应路径
   - 仅允许将文本中的以下信息合并到已有的图像反应中：
     - conditions (temp, time)
     - solvent
     - yields
     - reagents / catalysts
     - experiments（实验描述摘要）
   - 匹配依据：反应物+产物 SMILES Tanimoto ≥ 0.9

5. 【完整度评分】
   为每条 reaction 计算 completeness_score (0.0–1.0)：
   - rxn_smiles 有效: +0.2
   - 至少一个 reactant 具名且有 smiles: +0.1
   - 至少一个 product 具名且有 smiles: +0.1
   - solvent 明确: +0.1
   - conditions 完整 (temp+time): +0.2
   - yield 给出: +0.2
   - atom_mapping_conf ≥ 0.6: +0.1

# 输出格式（严格 JSON）
{
  "reactions": [
    {
      "rxn_smiles": string | null,
      "reactants": [ {"name": str, "smiles": str, "evidence": str} ],
      "products": [ {"name": str, "smiles": str, "evidence": str} ],
      "reagents": [ {"name": str, "role": str} ],
      "solvent": string,
      "experiments": string,
      "conditions": { "temp": string, "time": string },
      "yields": [ {"value": number, "unit": "%", "substrate_label": string} ],
      "atom_mapping_conf": number | null,
      "evidence": { "source": "image", "page": int, "boxes": [[x1,y1,x2,y2]] },
      "completeness_score": float
    }
  ],
  "fusion_notes": string  // 如："Added yield 85% and 24h from text to image reaction"
}

# 强制约束
- 所有 ocr_result 反应必须保留在 reactions 中
- 字段缺失写 null / [] / {}
- 输出仅为合法 JSON，无解释、无推理过程
- 不得创建 alternatives
- non_reaction_chemicals 与 reactions 严格去重

现在开始融合：
""".strip()



class FusionAgent(BaseAgent):
    """融合文本和图像提取结果的智能体"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = Config.get("FUSION_MODEL", "gpt-4-turbo")
        self.min_confidence = Config.get("MIN_CONFIDENCE_SCORE", 0.7)
        self.max_retries = 2  # 最多重试2次
    
    def process(self, state: ChemistryExtractionState):
        """融合处理的主要逻辑"""
        try:
            metadata = {}
            metadata['fusion_agent_start'] = time.time()
            current_stage = ["fusion"]
            
            text_data = json.dumps(state["text_extractions"], ensure_ascii=False)

            # 构建完整的prompt
            fusion_results = []
            for item in state["image_extractions"]:
                image_data = json.dumps(item["ocr_result"], ensure_ascii=False)
                full_prompt = FUSION_PROMPT
                full_prompt = full_prompt.replace("@text_analysis@", text_data)
                full_prompt = full_prompt.replace("@image_analysis@", image_data)

            # 尝试获取有效JSON响应
                fusion_result = self._get_valid_fusion_result(full_prompt)
            
                if fusion_result is None:
                    self.logger.error("Failed to get valid fusion result after multiple attempts")
                    return self._create_empty_fusion(state)

                fusion_result['bbox'] = item['bbox']
                fusion_result['page_idx'] = item['page_idx']
                fusion_results.append(fusion_result)
            current_stage.append('completed')
            metadata['fusion_agent_end'] = time.time()
            metadata['fusion_success'] = True
            
            # 记录成功信息
            reaction_count = len(fusion_results)
            self.logger.info(f"Successfully fused text and image data. Found {reaction_count} reactions")
            
            return {
                "fusion_result": fusion_results,
                "current_stage": current_stage,
                "metadata": metadata
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.handle_error(state, e, "fusion_process")
    
    def _get_valid_fusion_result(self, full_prompt: str) -> Optional[Dict[str, Any]]:
        """尝试获取有效的融合结果，支持重试机制"""
        try:
            self.logger.info(f"Calling LLM for data fusion ....")
            response = call_llm(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=8192,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # 清理并验证JSON
            paresed_response = robust_json_parse(response)

            return paresed_response

        except json.JSONDecodeError as je:
            self.logger.warning(f"JSON decode error on attempt : {str(je)}")
        except Exception as e:
            self.logger.warning(f"Error on attempt : {str(e)}")
        
        return None
    

    def _create_empty_fusion(self, state: ChemistryExtractionState):
        """创建空融合结果"""
        fusion_result = {
            "reactions": [],
            "experimental_procedure": "No extractable data found in document",
            "validation_notes": ["No text or image data available for fusion"]
        }
        current_stage = ["complete"]
        metadata = {"fusion_success": False}
        return {
            "current_stage": current_stage,
            "metadata": metadata,
            "fusion_result" : fusion_result
        }


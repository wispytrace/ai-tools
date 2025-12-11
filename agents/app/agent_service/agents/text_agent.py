# chemistry_extraction/agents/text_agent.py
import re
import json
import time
from typing import Dict, Any, List, Optional
from ..state import ChemistryExtractionState
from ..agents import BaseAgent
from ..utils.llm_utils import call_llm, clean_json_response
from ..config import Config
import os
import json
PROMPT_REACTIONS = """
你是一个化学反应信息与物质抽取专家。请从以下的json数组的文本中提取所有与合成、转化、实验操作相关的信息片段以及被提及的化学物质和属性。

化学反应信息重点包括：
- 反应路径（底物→产物）
- 实验条件（温度、时间、溶剂、气氛）
- 试剂/催化剂
- 产率数据
- 实验步骤描述

化学物质属性重点包括：
- 名称、标签（如 cmpd-3a）
- 别名、编号
- 物理性质（颜色、熔点、旋光度）
- 分析数据（NMR, HRMS）
- 功能用途（抑制剂、中间体等）

每个 reaction 条目包含：
reactants, reagents, solvent, conditions, yields, products, experiments, description_snippet, confidence_text

每个 compound 条目包含：
name, label, aliases, smiles, properties, structural_notes, functional_role, evidence

输出为 JSON 列表，字段缺失设为 null 或 []，SMILES 超过 200 字符则留空。
"""

class TextExtractionAgent(BaseAgent):
    """处理Markdown文本并提取化学信息的智能体"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = Config.get("TEXT_EXTRACTION_MODEL")
    
    def process(self, state: ChemistryExtractionState) -> ChemistryExtractionState:
        """处理文本提取的主要逻辑"""
        try:
            metadata = {}
            metadata['text_agent_start'] = time.time()
            current_stage = ["text_processing"]

            json_sections = state["text_jsons"]
            self.logger.info(f"Found {len(json_sections)} sections to process")

            
            # 处理每个相关部分
            extractions = []
            section_str = json.dumps([section.get('text', '') for section in json_sections], ensure_ascii=False)
            extractions = self._extract_from_section(section_str)
            # for i, section in enumerate(json_sections):
            #     self.logger.info(f"Processing section {i+1}/{len(json_sections)}")
            #     extraction = {}
            #     # 调用LLM提取信息
            #     section_str = json.dumps(section.get('text', ''), ensure_ascii=False)
            #     extraction['result'] = self._extract_from_section(section_str)
            #     extraction['source_text_index'] = i
            #     if extraction:
            #         extractions.append(extraction)
            
            # 更新状态
            text_extractions = extractions
            metadata['text_extractions_count'] = len(extractions)
            metadata['text_agent_end'] = time.time()

            self.logger.info(f"Successfully extracted {len(extractions)} text sections")
            return {
                'text_extractions': text_extractions,
                'current_stage': current_stage,
                'metadata': metadata
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.handle_error(state, e, "text_extraction_process")

    def _extract_from_section(self, section: Dict[str, str]):
        """从单个部分提取化学信息"""
        try:
            full_prompt = PROMPT_REACTIONS.strip() + section + "\n\n"
            
            response = call_llm(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=8192,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            cleaned = clean_json_response(response)
            result = json.loads(cleaned)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract from section {section}: {str(e)}")
            print("llm response was:", response)
            return None

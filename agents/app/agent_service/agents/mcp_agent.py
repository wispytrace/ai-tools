import json
import time
from typing import Dict, Any, List, Optional
from ..state import ChemistryExtractionState
from ..agents import BaseAgent
from ..tools.base_tool import BaseTool
import os
from ..utils.result_export import extract_fusion_result
import copy

class MCPAgent(BaseAgent):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mcp_tool = config.get("mcp_tool", "unknown")
    
    def process(self, state: ChemistryExtractionState):
        try:
            op = self.mcp_tool
            self.logger.info(f"MCP Agent processing operation: {self.mcp_tool}")
            current_stage = [f"mcp_tool_processing_{op}"]
            metadata = {}
            metadata[f"mcp_agent_start_{op}"]  = time.time()
            tool = BaseTool.get_tool(op, {})
            if op == "mineru_pdf_extraction":
                file_name = os.path.basename(state["pdf_path"])
                pdf_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../extracted_results/{file_name.split('.')[0]}_mineru_output")
                if not os.path.exists(pdf_output_dir):
                    print("该文件首次解析，上传mineru开始解析")
                    result = tool.run({
                        "pdf_path": state["pdf_path"],
                    })
                    pdf_output_dir = result.data.get("pdf_output_dir", "")
                conetnt_json = ""
                for file in os.listdir(pdf_output_dir):
                    if "content_list.json" in file:
                        conetnt_json = os.path.join(pdf_output_dir, file)
                with open(conetnt_json, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                json_sections = json.loads(raw_text)
                text_jsons = [section for section in json_sections if section['type'] != 'image']
                image_jsons = [section for section in json_sections if section['type'] == 'image']
                metadata['f"mcp_agent_end_{op}"'] = time.time()
                return{
                    'current_stage': current_stage,
                    'pdf_output_dir': pdf_output_dir,
                    'raw_text': raw_text,
                    'text_jsons': text_jsons,
                    'image_jsons': image_jsons,
                    'metadata': metadata
                }
            elif op == "reaction_checker":
                extract_result = extract_fusion_result(state['fusion_result'])
                reaction_smiles = []
                for i in range(len(extract_result)):
                    reaction_smiles.append(extract_result[i]['reactants'])
                result = tool.run({
                    'reaction_smiles': reaction_smiles
                })
                rxn_results = result.data
                for i in range(len(extract_result)):
                    if rxn_results[i] != {}:
                        extract_result[i]['mapped_rxn'] = rxn_results[i]['mapped_rxn']
                        extract_result[i]['confidence'] = rxn_results[i]['confidence']
                    else:
                        extract_result[i]['mapped_rxn'] = ''
                        extract_result[i]['confidence'] = 0.0
                metadata['f"mcp_agent_end_{op}"'] = time.time()
                return{
                    'current_stage': current_stage,
                    'metadata': metadata,
                    'extported_result': extract_result
                }
            elif op == "yolo_detector":
                compound_detections = []
                for section in state["image_jsons"]:
                    image_path = os.path.join(state["pdf_output_dir"], section.get('img_path', ''))
                    if not os.path.exists(image_path):
                        self.logger.warning(f"Image path does not exist: {image_path}")
                        continue
                    result = tool.run({
                        "image_path": image_path,
                    })
                    compound_detection = copy.deepcopy(section)
                    compound_detection['detect'] = result.data
                    compound_detections.append(compound_detection)
                print(f"MCP Agent yolo_detector found {len(compound_detections)} image detections.")
                metadata['f"mcp_agent_end_{op}"'] = time.time()
                return{
                    'yolo_detections': compound_detections,
                    'current_stage': current_stage,
                    'metadata': metadata,
                }
                            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.handle_error(state, e, "mcp_tool_process")
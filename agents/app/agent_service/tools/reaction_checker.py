
# chemistry_extraction/tools/paddle_ocr_tool.py

import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from rxnmapper import RXNMapper, BatchedMapper
from .base_tool import BaseTool, ToolResult
import requests
import mimetypes

class ReactionChecker(BaseTool):
    """
    MCP 工具：对识别出的smiles码进行校验
    """
    name: str = "reaction_checker"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 可配置参数
        # 初始化 PaddleOCR 实例（仅初始化一次）
        self.logger.info("ReactionChecker instance initialized.")
        self.rxn_mapper =  BatchedMapper(batch_size=32)


    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        try:
            # 解析输入
            self.logger.info(f"Running reaction checker")
            rxn_smiles = input_data["reaction_smiles"]
            results = list(self.rxn_mapper.map_reactions_with_info(rxn_smiles))
            # 执行 OCR 推理
            exec_time = time.time() - start_time
            return ToolResult(
                success=True,
                data=results,
                tool_name=self.name,
                execution_time=exec_time
            )

        except Exception as e:
            exec_time = time.time() - start_time
            self.logger.error(f"OCR execution failed: {str(e)}")
            return ToolResult(
                success=False,
                data={},
                tool_name=self.name,
                execution_time=exec_time,
                error=str(e)
            )


def run_reaction_checker(smiles: str, config: Dict[str, Any] = None) -> ToolResult:
    """
    快捷函数：直接运行 OCR 工具
    """
    tool = ReactionChecker(config=config)
    return tool.execute({"reaction_smiles": smiles})

if __name__ == "__main__":
    # 测试代码

    rxns = ['CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F', 'C1COCCO1.CC(C)(C)OC(=O)CONC(=O)NCc1cccc2ccccc12.Cl>>O=C(O)CONC(=O)NCc1cccc2ccccc12']
    results = run_reaction_checker(rxns)
    print(results)
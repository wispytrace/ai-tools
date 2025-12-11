# chemistry_extraction/tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import time
import logging
import json

class ToolResult:
    """工具执行结果封装"""
    def __init__(
        self,
        success: bool,
        data: Dict[str, Any],
        tool_name: str,
        execution_time: float,
        error: Optional[str] = None
    ):
        self.success = success
        self.data = data
        self.tool_name = tool_name
        self.execution_time = execution_time
        self.error = error
        self.timestamp = time.time()
    
    def as_dict(self) -> Dict[str, Any]:
        """将结果转换为字典形式"""
        return {
            "success": self.success,
            "data": self.data,
            "tool_name": self.tool_name,
            "execution_time": self.execution_time,
            "error": self.error,
            "timestamp": self.timestamp
        }

class BaseTool(ABC):
    """所有外部工具的基类"""
    
    name: str = "base_tool"
    description: str = "Base tool for chemical information processing"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        self.timeout = self.config.get("timeout", 10.0)  # 默认超时10秒
    
    def _setup_logger(self):
        """设置工具专用日志器"""
        logger = logging.getLogger(f"tool.{self.name}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - MCP-{self.name.upper()} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据是否适合本工具"""
        return True  # 默认总是有效，子类可重写
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """执行工具核心逻辑"""
        pass
    
    def run(self, input_data: Dict[str, Any]) -> ToolResult:
        """统一执行入口（带超时和错误处理）"""

        if not self.validate_input(input_data):
            return ToolResult(
                success=False,
                data={},
                tool_name=self.name,
                execution_time=0,
                error="Invalid input for this tool"
            )
        
        start_time = time.time()
        try:
            # 实际执行（子类实现）
            result = self.execute(input_data)
            exec_time = time.time() - start_time
            
            # 确保返回的是 ToolResult
            if isinstance(result, dict):
                return ToolResult(
                    success=True,
                    data=result,
                    tool_name=self.name,
                    execution_time=exec_time
                )
            return result
        except Exception as e:
            exec_time = time.time() - start_time
            error_msg = f"{str(e)}\nInput: {json.dumps(input_data, indent=2)[:500]}"
            self.logger.error(f"Tool execution failed: {error_msg}")
            return ToolResult(
                success=False,
                data={},
                tool_name=self.name,
                execution_time=exec_time,
                error=str(e)
            )
    
    @classmethod
    def get_tool(cls, tool_name: str, config: Dict[str, Any] = None) -> 'BaseTool':
        """工具工厂方法"""
        from .mineru_tool import MinerUExtractTool
        from .reaction_checker import ReactionChecker
        from .yolo_detect import YoloDetector
        # from .paddle_ocr_tool import PaddleOCRPredictTool
        
        print("try to get tool:", tool_name)
        TOOL_MAP: Dict[str, Type[BaseTool]] = {
            "mineru_pdf_extraction": MinerUExtractTool,
            "reaction_checker": ReactionChecker,
            "yolo_detector": YoloDetector,
            # "paddle_ocr_prediction": PaddleOCRPredictTool,
        }
        
        if tool_name not in TOOL_MAP:
            raise ValueError(f"Unknown MCP tool: {tool_name}. Available: {list(TOOL_MAP.keys())}")
        
        return TOOL_MAP[tool_name](config)

    @classmethod
    def list_available_tools(cls) -> List[str]:
        """列出所有可用的MCP工具名称"""
        return [
            "mineru_pdf_extraction",
            "reaction_checker",
            "yolo_detector",
            # "paddle_ocr_prediction",
            # "smiles_normalizer",
            # "reaction_checker",
        ]
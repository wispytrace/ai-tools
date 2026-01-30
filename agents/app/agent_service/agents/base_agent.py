# chemistry_extraction/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from ..state import ChemistryExtractionState
import time

class BaseAgent(ABC):
    """所有智能体的基类 - 定义统一接口"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__.replace("Agent", "").lower()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """设置智能体专用日志器"""
        import logging
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - {self.name.upper()} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    @abstractmethod
    def process(self, state: ChemistryExtractionState):
        """处理状态的核心方法 - 所有子类必须实现"""
        pass
    
    def handle_error(self, state: ChemistryExtractionState, error: Exception, context: str):
        """统一错误处理"""
        error_msg = f"{time.time()}--Error in {self.name} agent: {str(error)} | Context: {context}"
        self.logger.error(error_msg)
        return {
            'errors': [error_msg]
        }
    
    @classmethod
    def get_agent(cls, agent_type: str, config: Optional[Dict[str, Any]] = None) -> 'BaseAgent':
        """智能体工厂方法"""
        from .text_agent import TextExtractionAgent
        from .image_agent import ImageAnalysisAgent
        from .fusion_agent import FusionAgent
        from .mcp_agent import MCPAgent
        from .compound_agent import CompoundNameAgent
        
        AGENT_MAP: Dict[str, Type[BaseAgent]] = {
            "text": TextExtractionAgent,
            "image": ImageAnalysisAgent,
            "fusion": FusionAgent,
            "mcp": MCPAgent,
            "compound": CompoundNameAgent
        }
        
        if agent_type not in AGENT_MAP:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENT_MAP.keys())}")
        
        return AGENT_MAP[agent_type](config)

# chemistry_extraction/workflow.py
from typing import Dict, Any, Callable, TypedDict, Optional
from langgraph.graph import StateGraph, END
from .state import ChemistryExtractionState
from .agents import BaseAgent
from .config import Config
from .tools.base_tool import BaseTool

class NodeFunction(TypedDict):
    name: str
    function: Callable[[ChemistryExtractionState], ChemistryExtractionState]

class ChemistryWorkflow:
    """化学信息提取工作流控制器（增强版，支持MCP）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.graph = StateGraph(ChemistryExtractionState)
        self.nodes: Dict[str, NodeFunction] = {}
        
        # 注册节点
        self._setup_nodes()
        # self._setup_edges()
    
    def set_workflow(self, workflow_type: str):
        """设置工作流类型"""
        if workflow_type == "compound_extraction":
            self._setup_compound_extract_edges()
        elif workflow_type == "reaction_extraction":
            self._setup_reaction_edges()

    def _setup_nodes(self):
        """注册所有工作流节点"""
        # 任务路由节点
        mineru_mcp_agent = BaseAgent.get_agent("mcp", {"mcp_tool": "mineru_pdf_extraction"})
        self.graph.add_node("mineru_processing", mineru_mcp_agent.process)
        self.nodes["mineru_processing"] = {
            "name": "Mineru PDF Extraction",
            "function": mineru_mcp_agent.process
        }

        # 文本处理节点
        text_agent = BaseAgent.get_agent("text", self.config.get("text_agent", {}))
        self.graph.add_node("text_processing", text_agent.process)
        self.nodes["text_processing"] = {
            "name": "Text Extraction",
            "function": text_agent.process
        }

        # 图像切割节点
        yolo_mcp_agent = BaseAgent.get_agent("mcp", {"mcp_tool": "yolo_detector"})  
        self.graph.add_node("yolo_processing", yolo_mcp_agent.process)
        self.nodes["yolo_processing"] = {
            "name": "Yolo Detection",
            "function": yolo_mcp_agent.process
        }
        # 化合物提取节点
        compound_agent = BaseAgent.get_agent("compound", self.config.get("compound_agent", {}))
        self.graph.add_node("compound_processing", compound_agent.process)
        self.nodes["compound_processing"] = {
            "name": "Compound Extraction",
            "function": compound_agent.process
        }

        # 图像处理节点
        image_agent = BaseAgent.get_agent("image", self.config.get("image_agent", {}))
        self.graph.add_node("image_processing", image_agent.process)
        self.nodes["image_processing"] = {
            "name": "Image Analysis",
            "function": image_agent.process
        }
        
        # 融合节点
        fusion_agent = BaseAgent.get_agent("fusion", self.config.get("fusion_agent", {}))
        self.graph.add_node("fusion", fusion_agent.process)
        self.nodes["fusion"] = {
            "name": "Fusion",
            "function": fusion_agent.process
        }

        #验证节点
        reaction_checker_mcp_agent = BaseAgent.get_agent("mcp", {"mcp_tool": "reaction_checker"})
        self.graph.add_node("reaction_checker", reaction_checker_mcp_agent.process)
        self.nodes["reaction_checker"] = {
            "name": "Reaction Checker",
            "function": reaction_checker_mcp_agent.process
        }


    def _setup_compound_extract_edges(self):
        """设置Yolo工作流边和条件路由"""
        # 从开始节点路由
        self.graph.set_entry_point("mineru_processing")        
        self.graph.add_edge("mineru_processing", "yolo_processing")
        self.graph.add_edge("yolo_processing", "compound_processing")
        self.graph.add_edge("compound_processing", END)

    def _setup_reaction_edges(self):
        """设置Reaction工作流边和条件路由"""
        # 从开始节点路由
        self.graph.set_entry_point("mineru_processing")

        self.graph.add_edge("mineru_processing", "yolo_processing")

        self.graph.add_edge("yolo_processing", "text_processing")
        
        self.graph.add_edge("yolo_processing", "image_processing")

        self.graph.add_edge(["image_processing", "text_processing"], "fusion")

        self.graph.add_edge("fusion", "reaction_checker")

        self.graph.add_edge("reaction_checker", END)

    def _setup_edges(self):
        """设置工作流边和条件路由"""
        # 从开始节点路由
        # self._setup_compound_extract_edges()
        self._setup_reaction_edges()

        # 融合后结束
    
    def compile(self) -> StateGraph:
        """编译工作流图"""

        comiled_graph = self.graph.compile()
        print(comiled_graph.get_graph().draw_ascii())
        return comiled_graph
    
    def get_node_info(self) -> Dict[str, Dict[str, str]]:
        """获取节点信息用于调试"""
        return {k: {"name": v["name"]} for k, v in self.nodes.items()}

    

# chemistry_extraction/config.py
import os
from typing import Dict, List, Optional

class Config:
    """全局配置类 - 所有参数集中管理"""
    
    # LLM 配置
    LLM_PROVIDER: str = "dashscope"  # 或 "dashscope"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-3aa117e4db4b471ebe20215f1bbc3b06")
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "sk-3aa117e4db4b471ebe20215f1bbc3b06")
    
    # 模型配置
    TEXT_EXTRACTION_MODEL: str = "qwen-plus"
    IMAGE_ANALYSIS_MODEL: str = "qwen3-vl-plus-2025-09-23"
    FUSION_MODEL: str = "qwen3-max"
    TASK_ROUTING_MODEL: str = "qwen-plus"
    IMAGE_COMPOUND_DETECTION_MODEL: str = "qwen-vl-max"
    ARROW_DETECTION_MODEL: str = "qwen-vl-max"
    # 路径配置
    DEFAULT_IMAGE_DIR: str = "./images"
    OUTPUT_DIR: str = "./output"
    
    # 处理参数
    MAX_CONCURRENT_TASKS: int = 3
    TEXT_CHUNK_SIZE: int = 2000
    MIN_CONFIDENCE_SCORE: float = 0.7
    
    # 扩展配置 (支持动态添加)
    _extra_configs: Dict[str, any] = {}
    
    @classmethod
    def set(cls, key: str, value: any) -> None:
        """动态设置配置项"""
        if hasattr(cls, key):
            setattr(cls, key, value)
        else:
            cls._extra_configs[key] = value
    
    @classmethod
    def get(cls, key: str, default: any = None) -> any:
        """安全获取配置项"""
        if hasattr(cls, key):
            return getattr(cls, key)
        return cls._extra_configs.get(key, default)
    
    @classmethod
    def validate(cls) -> None:
        """验证关键配置"""
        if cls.LLM_PROVIDER == "dashscope" and not cls.DASHSCOPE_API_KEY:
            raise ValueError("DashScope API key is required when using DashScope provider")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required when using OpenAI provider")

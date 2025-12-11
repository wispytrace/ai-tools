# chemistry_extraction/utils/llm_utils.py
import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..config import Config

def get_client() -> OpenAI:
    """获取LLM客户端"""
    if Config.LLM_PROVIDER == "dashscope":
        return OpenAI(
            api_key=Config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:  # OpenAI
        return OpenAI(
            api_key="sk-3aa117e4db4b471ebe20215f1bbc3b06",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus"
        )

def call_llm(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 8192,
    temperature: float = 0.2,
    response_format: Optional[Dict[str, str]] = None
) -> str:
    """统一调用LLM接口"""
    client = get_client()
    
    # 准备请求参数
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if response_format:
        params["response_format"] = response_format
    
    # 调用API
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content

def clean_json_response(response: str) -> str:
    """清理LLM返回的JSON响应"""
    # 移除可能的Markdown代码块
    # response = re.sub(r'^```(?:json)?\n?', '', response)
    # response = re.sub(r'\n?```$', '', response)
    
    # 确保是有效的JSON
    try:
        # 尝试解析以验证
        json.loads(response)
        return response
    except json.JSONDecodeError:
        # 尝试提取第一个JSON对象
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json_match.group(0)
        raise ValueError("Could not extract valid JSON from response")

def robust_json_parse(text):
    """尽力从不规则文本中提取并解析 JSON"""
    import json
    import json5

    # 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass

    # 尝试 json5
    try:
        return json5.loads(text)
    except:
        pass

    # 提取可能的 JSON 块
    brackets = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{' or c == '[':
            if start is None:
                start = i
            depth += 1
        elif c == '}' or c == ']':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i+1]
                try:
                    return json5.loads(candidate)
                except:
                    start = None
                start = None  # 继续找下一个？
    
    return {"content": text}  # 返回原始文本作为最后手段

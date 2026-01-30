import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# LangChain 相关库
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

# ================= 配置区域 =================
# 建议将 KEY 放入环境变量，或者在此处临时替换
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY", "sk-3aa117e4db4b471ebe20215f1bbc3b06")

# 初始化 FastAPI
app = FastAPI(title="ChemLLM Retro Prediction Service")

# ================= Pydantic 数据模型 =================

# 1. 接口输入模型
class RetroRequest(BaseModel):
    target_product: str = Field(..., description="Target product SMILES")
    ref_reaction: str = Field(default="", description="Reference reaction hint (optional)")

# 2. LLM 输出结构定义 (用于指导 LLM 生成 JSON)
class ReactionExtraction(BaseModel):
    reaction_smiles: str = Field(description="The complete reaction SMILES string in format 'Reactants>>Product'. Example: 'C=C.O>>CCO'")
    reactants: str = Field(description="The SMILES of reactants only. Example: 'C=C.O'")
    yield_val: str = Field(description="The yield percentage if explicitly mentioned or highly certain (e.g., '85%'). If unknown, return empty string ''.")
    conditions: str = Field(description="Reaction conditions (solvent, temperature, catalyst) if known. If unknown, return empty string ''.")
    doi: str = Field(description="DOI or literature reference if specifically known for this reaction. If unknown, return empty string ''. DO NOT make up fake DOIs.")

# 3. 接口统一响应模型
class APIResponse(BaseModel):
    status: str
    inference_time: float
    data: Dict[str, Any]
    error: Optional[str] = None

# ================= LLM 初始化 =================

# 初始化解析器
parser = JsonOutputParser(pydantic_object=ReactionExtraction)

# 初始化模型
# temperature=0.01 极低温度，为了保证化学式的准确性和防止幻觉
llm = ChatOpenAI(
    model="qwen3-max",
    openai_api_key=ALIYUN_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.01,
    max_tokens=1024
)

# ================= 核心接口逻辑 =================

@app.post("/predict_retro", response_model=APIResponse)
async def predict_retro(data: RetroRequest):
    start_time = time.time()
    
    # --- 1. 构建 System Prompt (防幻觉核心) ---
    system_content = """You are a Senior Organic Synthesis Strategist and Computational Chemist. 
    Your goal is to design highly feasible retrosynthetic routes based on mechanistic rigor.

    **CORE RESPONSIBILITIES**:
    1. **SMILES Interpretation**: Treat input SMILES strings as molecular graphs representing 3D structures. You must analyze:
    - Electronic effects (Inductive/Resonance effects of substituents).
    - Steric hindrance (how bulkiness affects reaction sites).
    - pKa and acidity/basicity of functional groups.
    2. **Chemoselectivity & Compatibility**: 
    - Ensure the proposed reagents do NOT react with other functional groups in the molecule (or suggest protecting groups if necessary).
    - Verify that the reaction conditions (pH, temperature, oxidants/reductants) are compatible with the product's stability.
    3. **Mechanistic Validity**: The disconnection must follow established organic reaction mechanisms (e.g., SN2, Aldol, Suzuki-Miyaura, Amide coupling).

    **OUTPUT RULES**:
    1. You must respond in valid **JSON** format only.
    2. **ANTI-HALLUCINATION**: If specific details (Yield, Conditions, DOI) are not strictly inferable or known, set them to an **empty string ""**. 
    3. The `reaction_smiles` must be atom-mapped if possible, and strictly balanced.
    """
    # --- 2. 构建 User Prompt ---
    # 动态插入 Target 和 Hint，并注入 format_instructions
    user_content = f"""
    [Task]
    Perform a retrosynthetic analysis for the Target Product below. 

    [Input Data]
    Target Product (SMILES): {data.target_product}
    Reference Hint (Optional): {data.ref_reaction if data.ref_reaction else "No reference provided - rely on general chemical principles."}

    [Analysis Requirements]
    Before generating the JSON, internally process the following steps:
    1. **Identify the Strategic Bond**: Find the bond most likely to be formed last (the "disconnection site") based on standard retrosynthetic rules (transform-based).
    2. **Functional Group Scan**: List all functional groups in the target. Check if the proposed reaction conditions would destroy any existing groups (e.g., using LiAlH4 in the presence of an ester when you only want to reduce an amide).
    3. **Synthons to Reagents**: Convert the theoretical synthons into commercially available or stable reactant SMILES.

    [Output Format]
    {parser.get_format_instructions()}
    """

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ]

    try:
        # --- 3. 调用 LLM ---
        # invoke 会自动处理网络请求
        response = llm.invoke(messages)
        
        # --- 4. 解析结果 ---
        # parser.parse 会自动提取 Markdown JSON 代码块并转为 Dict
        parsed_result = parser.parse(response.content)
        
        # --- 5. 后处理清洗 (Double Check) ---
        # 确保 SMILES 格式基本正确 (去除多余空格)
        if "reaction_smiles" in parsed_result:
            parsed_result["reaction_smiles"] = parsed_result["reaction_smiles"].strip().replace(" . ", ".").replace(" >> ", ">>")
        if "reactants" in parsed_result:
             parsed_result["reactants"] = parsed_result["reactants"].strip().replace(" . ", ".")

        end_time = time.time()
        
        return APIResponse(
            status="success",
            inference_time=round(end_time - start_time, 2),
            data=parsed_result
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 优雅降级：报错时不崩溃，返回空数据结构
        return APIResponse(
            status="error",
            inference_time=round(time.time() - start_time, 2),
            error=str(e),
            data={
                "reaction_smiles": "",
                "reactants": "",
                "yield_val": "",
                "conditions": "",
                "doi": ""
            }
        )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting ChemLLM Service (Qwen-Plus Powered)...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
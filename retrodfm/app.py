import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import asynccontextmanager
import time

# ================= 配置区域 =================
MODEL_PATH = "/root/binghao/retrodfm/chellm/ChemLLM-20B-Chat-DPO"

model_resource = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"📂 Loading model from: {MODEL_PATH}")
    try:
        model_resource["tokenizer"] = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True
        )
        model_resource["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise e
    yield
    model_resource.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# ================= 数据模型 =================
class RetroRequest(BaseModel):
    target_product: str
    # 改为单一字段：参考反应
    # 您可以传入 SMILES 反应式 (A.B>>C) 或任何描述性反应字符串
    ref_reaction: str = "" 

class RetroResponse(BaseModel):
    reactants: str
    inference_time: float

# ================= 核心逻辑 =================
@app.post("/predict_retro", response_model=RetroResponse)
async def predict_retro(data: RetroRequest):
    if "model" not in model_resource:
        raise HTTPException(status_code=503, detail="Model not initialized")

    tokenizer = model_resource["tokenizer"]
    model = model_resource["model"]

    # 动态构建 Prompt
    # 如果用户没传参考反应，Prompt 中这部分会显示为空，不影响模型
    prompt = f"""<|im_start|>system
You are an expert organic chemist.
Task: Predict the reactants for the Target Product.



You are provided with a [Reference Example] retrieved from a database.
**CRITICAL INSTRUCTION**:
- The Reference is a HINT.
- **IF** the Reference logic applies to the Target, adapt it.
- **IF** the Reference is irrelevant or you know a better route, **IGNORE** it.

<|im_start|>user
[Reference Example (Hint)]
Reaction: {data.ref_reaction}

[Target Product]
Product: {data.target_product}

**FORMATTING RULES**:
1. Output the **COMPLETE REACTION** SMILES string.
2. Strict Format: `Reactants>>Product` (e.g., `A.B>>C`).
3. NO explanations, NO extra tex

**FORMATTING RULES**:
1. Output the **COMPLETE REACTION** SMILES string.
2. Strict Format: `Reactants>>Product` (e.g., `A.B>>C`).
3. NO explanations, NO extra tex

**FORMATTING RULES**:
1. Output the **COMPLETE REACTION** SMILES string.
2. Strict Format: `Reactants>>Product` (e.g., `A.B>>C`).
3. NO explanations, NO extra tex

<|im_end|>
Best Reactants:
<|im_end|>
<|im_start|>assistant
"""

    start_time = time.time()
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        
        full_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 清洗结果：确保是用 "." 连接，且无多余空格
        clean_response = full_response.strip().replace(" . ", ".")

        end_time = time.time()
        
        return RetroResponse(
            reactants=clean_response,
            inference_time=round(end_time - start_time, 2)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
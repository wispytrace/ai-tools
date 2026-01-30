import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# =================配置区域=================
# 1. 这里填你解压后的模型文件夹绝对路径
# 例如: "/mnt/models/ChemLLM-7B-Chat-1.5-SFT"
MODEL_PATH = "/root/binghao/retrodfm/chellm/ChemLLM-20B-Chat-DPO" 

# =================加载模型=================
print(f"📂 Loading model from: {MODEL_PATH}")
print("   (This might take a minute...)")

# 加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True
    )
except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    exit()

# 加载模型
# device_map="auto": 自动把模型切分到你的两张 4090 上（如果一张放不下）
# torch_dtype=torch.bfloat16: 4090 专属优化精度，比 fp16 更稳，比 fp32 省一半显存
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("💡 Tip: Did you install 'accelerate'? (pip install accelerate)")
    exit()

print(f"✅ Model loaded successfully on: {model.device}")
print("-" * 50)

# ================= 灵活版 Prompt =================
target_product = "CCO"  # 布洛芬

ref_product = ""
ref_reactants = ""

prompt = f"""<|im_start|>system
You are an expert organic chemist.
Task: Predict the reactants for the Target Product.

You are provided with a [Reference Example] retrieved from a database based on structural similarity.
**CRITICAL INSTRUCTION**:
- The Reference is a HINT, not a rule.
- **IF** the Reference logic applies well to the Target, adapt it.
- **IF** you know a standard/better synthesis route for the Target (e.g., a well-known named reaction), or if the Reference is irrelevant, **IGNORE the Reference** and use your own knowledge.

**FORMATTING RULES (STRICT)**:
1. Output **ONLY** the SMILES strings of the reactants.
2. Join multiple reactants with a dot `.` (e.g., `ReactantA.ReactantB`).
3. **FORBIDDEN**: 
   - Do NOT output the Target Product SMILES.
   - Do NOT use reaction arrows (`>>` or `->`).
   - Do NOT explain.
<|im_end|>

<|im_start|>user
[Reference Example (Hint)]
Product: {ref_product}
Reactants: {ref_reactants}

[Target Product]
Product: {target_product}

Best Reactants:
<|im_end|>
<|im_start|>assistant
"""

# =================开始推理=================
print(f"🧪 Question: {prompt}")
print("Thinking...")

start_time = time.time()

# 1. 编码
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 2. 生成
# max_new_tokens: 生成的最大长度
# temperature: 0.2 (由于是科学问题，建议设低一点，让它更严谨，不要乱发挥)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False  # <--- 添加这一行！
    )
# 3. 解码
# 只截取生成的回答部分（去掉原来的 Prompt）
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

end_time = time.time()

# =================输出结果=================
print("-" * 50)
print("🤖 ChemLLM Answer:")
print(response)
print("-" * 50)
print(f"⏱️ Time Taken: {end_time - start_time:.2f}s")
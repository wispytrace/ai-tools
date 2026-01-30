import re
import os
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers

def generate_ultimate_selfies_tokenizer(input_file, output_path, min_freq=500):
    """
    1. 提取所有 [xxx] 语义块
    2. 提取所有基础字符 (0-9, a-z, A-Z, +, -, etc.)
    3. 实现高频整体化，低频零件化的混合分词
    """
    token_pattern = re.compile(r"(\[[^\]]+\])")
    unique_elements = Counter()
    all_raw_chars = set()

    print(f"📖 正在扫描 50w 数据文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到输入文件 {input_file}")
        return

    # 第一遍扫描：收集语义块和基础字符
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 收集 [C], [Branch1] 等语义块
            elements = token_pattern.findall(line)
            unique_elements.update(elements)
            
            # 收集所有基础字符（零件）
            all_raw_chars.update(list(line))

    # 构建词汇表
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    # 1. 优先加入所有基础单字符零件 (a-z, 0-9, +, -, =, #, [, ], etc.)
    sorted_chars = sorted(list(all_raw_chars))
    for char in sorted_chars:
        if char not in vocab:
            vocab[char] = len(vocab)
            
    # 2. 加入高频出现的语义块 (根据 min_freq 过滤)
    high_freq_elements = [el for el, count in unique_elements.items() if count >= min_freq]
    for el in sorted(high_freq_elements):
        if el not in vocab:
            vocab[el] = len(vocab)

    # 创建 WordLevel Tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))

    # 设置预分词器：核心正则
    # 优先匹配 [xxx] 结构，匹配不上的字符按 (.) 拆分
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"(\[[^\]]+\])|(.)", 
        behavior="isolated"
    )

    # 保存
    tokenizer.save(output_path)
    
    print("-" * 30)
    print(f"✅ Tokenizer 生成成功！")
    print(f"📦 词表总规模: {len(vocab)}")
    print(f"🧱 基础字符零件: {len(sorted_chars)}")
    print(f"🧪 高频语义块: {len(high_freq_elements)}")
    print("-" * 30)

    # --- 测试编码环节 ---
    print("\n🔍 运行测试编码：")

import re

def hybrid_encode(text, vocab):
    """
    两级路由编码函数：
    1. 优先匹配词表中的 [语义块] 或 基础字符
    2. 词表中没有的 [语义块] 自动拆解为单字符零件
    """
    # 获取基础 ID
    unk_id = vocab.get("<unk>", 3)

    # 1. 一级切分：利用正则提取出所有的 [xxx] 和 每一个单独的字符
    # 结果示例: "[C][Rare]" -> ['[C]', '[Rare]']
    # 结果示例: "C=O[C]" -> ['C', '=', 'O', '[C]']
    segments = [s[0] if s[0] else s[1] for s in re.findall(r"(\[[^\]]+\])|(.)", text)]

    final_ids = []
    
    for seg in segments:
        if seg in vocab:
            # 命中：高频语义块 或 基础零件字符
            final_ids.append(vocab[seg])
        elif seg.startswith("[") and seg.endswith("]"):
            # 未命中整体：将稀有中括号块拆解为 [ , R, a, r, e, ]
            for char in seg:
                final_ids.append(vocab.get(char, unk_id))
        else:
            # 其他无法识别的情况
            final_ids.append(unk_id)

    return final_ids

if __name__ == "__main__":
    # --- 请在此修改路径 ---
    INPUT_TXT = "/root/binghao/smiles/selfies.txt"
    OUTPUT_JSON = "tokenizer-selfies-ultimate.json"
    
    if os.path.exists(OUTPUT_JSON) is False:
        generate_ultimate_selfies_tokenizer(INPUT_TXT, OUTPUT_JSON, min_freq=500)

    test_cases = [
        "[C][O][Branch1]",              # 标准高频组合
        "[C][=O][NH1+]",                # 包含符号和数字的块
        "[C][RareElement999][C]",       # 模拟一个未在词表中出现的罕见块
        "C=O[C]"                        # 混合了非括号字符的情况
    ]

    tokenizer = Tokenizer.from_file(OUTPUT_JSON)
    for text in test_cases:
        token_ids = hybrid_encode(text, tokenizer.get_vocab())
        print(f"文本: {text} -> Token IDs: {token_ids}")

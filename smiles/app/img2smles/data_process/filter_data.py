import os
import re
import csv
from collections import Counter
from rdkit import Chem
try:
    import selfies as sf
except ImportError:
    print("❌ 未安装 selfies，请运行：pip install selfies")
    exit(1)

def smiles_to_selfies(smi):
    """将 SMILES 转换为标准化的 SELFIES"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        return sf.encoder(canonical_smi)
    except Exception:
        return None

def read_selfies_from_csv(file_path: str, output_path: str):
    """从 CSV 读取 SMILES 并保存为 SELFIES 文本文件"""
    print(f"📖 正在从 CSV 提取数据: {file_path}")
    selfies_list = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and len(row) > 1:
                line = row[1].strip()
                selfi = smiles_to_selfies(line)
                if selfi: selfies_list.append(selfi)

    with open(output_path, "w") as fout:
        for line in selfies_list:
            fout.write(line + "\n")
    print(f"✅ 提取完成，保存至: {output_path}")

def get_top_selfies_tokens(input_file: str, top_k: int = 30, save_freq_path: str = None):
    """
    提取高频 Tokens，并支持保存频数统计文件
    """
    token_counter = Counter()
    token_pattern = re.compile(r'\[[^\]]+\]')
    
    print(f"🔍 正在统计 Token 频率...")
    with open(input_file, "r") as f:
        for line in f:
            tokens = token_pattern.findall(line)
            for token in tokens:
                token_counter[token] += 1

    # === 新增功能：保存频数统计 ===
    if save_freq_path:
        print(f"💾 正在保存 Token 频数统计至: {save_freq_path}")
        with open(save_freq_path, "w") as f:
            f.write("Token\tCount\n") # 写个表头
            # most_common() 不带参数会返回所有元素，且按频数从高到低排序
            for token, count in token_counter.most_common():
                f.write(f"{token}\t{count}\n")
    # ============================

    top_tokens = set(token for token, _ in token_counter.most_common(top_k))
    print(f"✅ 已提取前 {top_k} 个高频 Tokens")
    return top_tokens

def filter_selfies(input_file: str, output_file: str, excpt_file: str, top_tokens: set):
    """执行过滤逻辑"""
    kept, excluded, total = 0, 0, 0
    token_pattern = re.compile(r'\[[^\]]+\]')

    print(f"🧹 开始过滤数据...")
    with open(input_file, "r") as fin, open(output_file, "w") as fout, open(excpt_file, "w") as fexc:
        for line in fin:
            total += 1
            original_line = line.strip()
            if not original_line: continue

            # 过滤多组分（含'.'）或不在 top_tokens 内的行
            tokens = token_pattern.findall(original_line)
            
            # 判断逻辑：不含圆点 AND 有Token AND 所有Token都在高频集合里
            if '.' not in original_line and len(tokens) > 0 and all(t in top_tokens for t in tokens):
                fout.write(original_line + "\n")
                kept += 1
            else:
                fexc.write(original_line + "\n")
                excluded += 1

    print(f"📊 过滤报告: 总计 {total}, 保留 {kept}, 排除 {excluded}")

if __name__ == "__main__":
    # 配置路径
    RAW_CSV = "/root/binghao/smiles/cas_smiles_202512191731.csv"
    SELFIES_TXT = "/root/binghao/smiles/selfies.txt"
    
    TOP_FILE = "/root/binghao/smiles/app/selfies_top30.txt"
    EXCEPT_FILE = "/root/binghao/smiles/app/except_top30.txt"
    
    # 新增：频数统计文件路径
    FREQ_FILE = "/root/binghao/smiles/app/token_frequencies.txt"

    # 1. 提取/生成 selfies.txt
    # if not os.path.exists(SELFIES_TXT):
    read_selfies_from_csv(RAW_CSV, SELFIES_TXT)
    
    # 2. 统计频率并获取 Top K (传入 save_freq_path 参数)
    top_tokens = get_top_selfies_tokens(
        SELFIES_TXT, 
        top_k=24, 
        save_freq_path=FREQ_FILE  # 这里传入路径就会自动保存
    )
    
    # 3. 过滤数据
    filter_selfies(SELFIES_TXT, TOP_FILE, EXCEPT_FILE, top_tokens)
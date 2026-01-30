# utils/bpe_tokenizer.py
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFD, StripAccents
import os
import csv

def train_bpe_tokenizer(smiles_file: str, vocab_size: int = 1000):
    # 特殊符号
    special_tokens = ["<sos>", "<eos>", "<pad>", "<unk>"]
    
    # 初始化 BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # SMILES 无空格，可省略
    tokenizer.normalizer = None  # 不要 normalize 化学符号！

    smiles_list = []
    with open(smiles_file, "r", encoding="utf-8") as f:
        reader = f.readlines()
        for row in reader:
            if not row:
                continue
            raw = row.strip()
            if raw is not None:
                smiles_list.append(raw)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )
    tokenizer.train_from_iterator(smiles_list, trainer=trainer)
    
    return tokenizer

# 使用示例
if __name__ == "__main__":
    tokenizer = train_bpe_tokenizer("/root/binghao/smiles/selfies.txt", vocab_size=200)
    tokenizer.save("tokenizer-bpe.json")

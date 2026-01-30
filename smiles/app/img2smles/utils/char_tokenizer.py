# utils/char_tokenizer.py
from tokenizers import Tokenizer, models, pre_tokenizers
import json

def create_selfies_char_tokenizer(selfies_file: str):
    # 1. 收集所有字符
    chars = set()
    with open(selfies_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                chars.update(list(line))  # 拆成字符！

    # 2. 构建词汇表
    special_tokens = ["<sos>", "<eos>", "<pad>", "<unk>"]
    vocab = {token: i for i, token in enumerate(special_tokens)}
    for char in sorted(chars):
        if char not in vocab:
            vocab[char] = len(vocab)

    # 3. 创建 tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    # 关键：不要任何 pre_tokenizer（默认按字符）
    # tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # 实际不用，因为输入无空格
    return tokenizer

# 保存
if __name__ == "__main__":
    tok = create_selfies_char_tokenizer("/root/binghao/smiles/selfies.txt")
    tok.save("tokenizer-char.json")
    
    # 测试时
    test = "[C][O][Ring1]"
    char_list = list(test)  # ['[', 'C', ']', '[', 'O', ']', '[', 'R', ...]

    # 手动查 ID
    ids = []
    for char in char_list:
        id_ = tok.token_to_id(char)
        if id_ is None:
            id_ = tok.token_to_id("<unk>")
        ids.append(id_)

    print("IDs:", ids)  # 应该是正常数字列表，不是 [3]

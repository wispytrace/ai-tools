# utils/tokenizer.py
class SMILESTokenizer:
    def __init__(self):
        # 常见 SMILES 字符 + 特殊符号
        chars = [
            ' ', '<sos>', '<eos>', '<pad>',
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'F', 'I', 'B', 'r', 'l', 'a', 'e',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '(', ')', '[', ']', '=', '#', '-', '+', '\\', '/', '%', '.'
        ]
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, smiles: str) -> list[int]:
        tokens = ['<sos>'] + list(smiles) + ['<eos>']
        return [self.stoi.get(ch, self.stoi[' ']) for ch in tokens]

    def decode(self, tokens: list[int]) -> str:
        chars = [self.itos[t] for t in tokens if t not in (self.stoi['<sos>'], self.stoi['<eos>'], self.stoi['<pad>'])]
        return ''.join(chars)

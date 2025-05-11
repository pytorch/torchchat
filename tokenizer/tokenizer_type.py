from enum import Enum

class TokenizerType(Enum):
    NONE = 0
    TIKTOKEN = 1
    SENTENCEPIECE = 2
    HF_TOKENIZER = 3

    def is_tiktoken(self):
        return self == TokenizerType.TIKTOKEN
    def is_sentencepiece(self):
        return self == TokenizerType.SENTENCEPIECE
    def is_hf_tokenizer(self):
        return self == TokenizerType.HF_TOKENIZER
    def is_none(self):
        return self == TokenizerType.NONE
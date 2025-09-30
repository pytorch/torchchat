from enum import Enum

class TokenizerType(Enum):
    NONE = 0
    TIKTOKEN = 1
    SENTENCEPIECE = 2
    HF_TOKENIZER = 3
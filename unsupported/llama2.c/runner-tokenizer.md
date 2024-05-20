The SentencePiece tokenizer implementations for Python (developed by
Google) and the C/C++ implementation (developed by Andrej Karpathy)
use different input formats. The Python implementation reads a
tokenizer specification in tokenizer.model format. The C/C++ tokenizer
that reads the tokenizer instructions from a file in tokenizer.bin
format. We include Andrej's SentencePiece converter which translates a
SentencePiece tokenizer in tokenizer.model format to tokenizer.bin in
the XXXutilsXXX subdirectory:

```
python3 XXXutilsXXX/tokenizer.py --tokenizer-model=${MODEL_DIR}/tokenizer.model
```
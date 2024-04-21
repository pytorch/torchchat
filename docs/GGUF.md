# Using GGUF Models
We currently support the following models
- F16
- F32
- Q4_0
- Q6_K


### Download
First download a GGUF model and tokenizer.  In this example, we use GGUF Q4_0 format.

```
mkdir -p ggufs/open_orca
cd ggufs/open_orca
wget -O open_orca.Q4_0.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true"

wget -O tokenizer.model "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model"
cd ../..

export GGUF_MODEL_PATH=ggufs/open_orca/open_orca.Q4_0.gguf
export GGUF_TOKENIZER_PATH=ggufs/open_orca/tokenizer.model
export GGUF_PTE_PATH=/tmp/gguf_model.pte
```

### Generate eager
```
python torchchat.py generate --gguf-path ${GGUF_MODEL_PATH} --tokenizer-path ${GGUF_TOKENIZER_PATH} --temperature 0 --prompt "In a faraway land" --max-new-tokens 20
```

### ExecuTorch export + generate
```
# Convert the model for use
python torchchat.py export --gguf-path ${GGUF_MODEL_PATH} --output-pte-path ${GGUF_PTE_PATH}

# Generate using the PTE model that was created by the export command
python torchchat.py generate --gguf-path ${GGUF_MODEL_PATH} --pte-path ${GGUF_PTE_PATH} --tokenizer-path ${GGUF_TOKENIZER_PATH} --temperature 0 --prompt "In a faraway land" --max-new-tokens 20

```

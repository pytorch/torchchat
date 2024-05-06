# Using GGUF Models
We support parsing [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) files with
the following tensor types:
- F16
- F32
- Q4_0
- Q6_K

If an unsupported type is encountered while parsing a GGUF file, an
exception is raised.

We now go over an example of using GGUF files in the torchchat flow.

### Download resources

First download a GGUF model and tokenizer.  In this example, we use a
Q4_0 GGUF file.  (Note that Q4_0 is only the dominant tensor type in
the file, but the file also contains GGUF tensors of types Q6_K, F16,
and F32.)

```
# Download resources
mkdir -p ggufs/open_orca
cd ggufs/open_orca
wget -O open_orca.Q4_0.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true"

wget -O tokenizer.model "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model"
cd ../..

export GGUF_MODEL_PATH=ggufs/open_orca/open_orca.Q4_0.gguf
export GGUF_TOKENIZER_PATH=ggufs/open_orca/tokenizer.model

# Define export paths for examples below
export GGUF_SO_PATH=/tmp/gguf_model.so
export GGUF_PTE_PATH=/tmp/gguf_model.pte
```

### Eager generate
We can generate text in eager mode as we did before, but we now pass a GGUF file path.
```
python3 torchchat.py generate --gguf-path ${GGUF_MODEL_PATH} --tokenizer-path ${GGUF_TOKENIZER_PATH} --temperature 0 --prompt "Once upon a time" --max-new-tokens 15
```

### AOTI export + generate
```
# Convert the model for use
python3 torchchat.py export --gguf-path ${GGUF_MODEL_PATH} --output-dso-path ${GGUF_SO_PATH}

# Generate using the PTE model that was created by the export command
python3 torchchat.py generate --gguf-path ${GGUF_MODEL_PATH} --dso-path ${GGUF_SO_PATH} --tokenizer-path ${GGUF_TOKENIZER_PATH} --temperature 0 --prompt "Once upon a time" --max-new-tokens 15

```


### ExecuTorch export + generate
Before running this example, you must first [Set-up ExecuTorch](executorch_setup.md).
```
# Convert the model for use
python3 torchchat.py export --gguf-path ${GGUF_MODEL_PATH} --output-pte-path ${GGUF_PTE_PATH}

# Generate using the PTE model that was created by the export command
python3 torchchat.py generate --gguf-path ${GGUF_MODEL_PATH} --pte-path ${GGUF_PTE_PATH} --tokenizer-path ${GGUF_TOKENIZER_PATH} --temperature 0 --prompt "Once upon a time" --max-new-tokens 15
```

[end default]: end

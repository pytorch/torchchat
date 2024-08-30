> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.

# Using GGUF Models

<!--
[shell default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}" huggingface-cli login

[shell default]: ./install/install_requirements.sh

[shell default]: TORCHCHAT_ROOT=${PWD} ./torchchat/utils/scripts/install_et.sh
-->

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
pushd ggufs/open_orca

curl -o open_orca.Q4_0.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true"
curl -o ./tokenizer.model https://github.com/karpathy/llama2.c/raw/master/tokenizer.model

popd

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
Before running this example, you must first [Set-up ExecuTorch](torchchat/edge/docs/executorch_setup.md).
```
# Convert the model for use
python3 torchchat.py export --gguf-path ${GGUF_MODEL_PATH} --output-pte-path ${GGUF_PTE_PATH}

# Generate using the PTE model that was created by the export command
python3 torchchat.py generate --gguf-path ${GGUF_MODEL_PATH} --pte-path ${GGUF_PTE_PATH} --tokenizer-path ${GGUF_TOKENIZER_PATH} --temperature 0 --prompt "Once upon a time" --max-new-tokens 15
```

### Advanced: loading unsupported GGUF formats in torchchat
GGUF formats not presently supported natively in torchchat can be
converted to one of the supported formats with GGUF's
[quantize](https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize) utility.
If you convert to the FP16 or FP32 formats with GGUF's quantize utility, you can
then requantize these models with torchchat's native quantization workflow.

**Please note that quantizing and dequantizing is a lossy process, and
you will get the best results by starting with the original
unquantized model, not a previously quantized and then
dequantized model.**

As an example, suppose you have [llama.cpp cloned and installed](https://github.com/ggerganov/llama.cpp) at ${GGUF}.
You can then convert a model to FP16 with the following command:

<!--
[shell command]: export GGUF=`pwd`/llama.cpp; git clone https://github.com/ggerganov/llama.cpp.git
[shell command]: cd llama.cpp ; make
-->

[skip default]: begin
```
${GGUF}/quantize --allow-requantize path_of_model_you_are_converting_from.gguf path_for_model_you_are_converting_to.gguf fp16
```
[skip default]: end

For example, to convert the quantized model you downloaded above to an FP16 model, you would execute:
```
${GGUF}/quantize --allow-requantize ${GGUF_MODEL_PATH} ./open_orca_fp16.gguf fp16
```

After the model is converted to a supported format like FP16, you can proceed using the instructions above.

[end default]: end

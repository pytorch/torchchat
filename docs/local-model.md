# Using Local Models in Torchcha/
Torchchat provides powerful capabilities for running large language models (LLMs) locally. This guide focuses on utilizing local copies of 
model checkpoints or models in GGUF format to create a chat application. It also highlights relevant options for advanced users.

## Prerequisites
To work with local models, you need:
1. **Model Weights**: A checkpoint file (e.g., `.pth`, `.pt`) or a GGUF file (e.g., `.gguf`).
2. **Tokenizer**: A tokenizer model file.This can either be in SentencePiece or TikToken format, depending on the tokenizer used with the model.
3. **Parameter File**: (a) A custom parameter file in JSON format, or (b) a pre-existing parameter file with `--params-path`
   or `--params-table`, or (c) a pathname that’s matched against known models by longest substring in configuration name, using the same algorithm as GPT-fast.

Ensure the tokenizer and parameter files are in the same directory as the checkpoint or GGUF file for automatic detection.
Let’s use a local download of the stories15M tinyllama model as an example:

```
mkdir stories15M
cd stories15M
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt
wget https://github.com/karpathy/llama2.c/raw/refs/heads/master/tokenizer.model
cp ../torchchat/model_params/stories15M.json model.json
cd ..
``` 


## Using Local Checkpoints
Torchchat provides the CLI flag `--checkpoint-path` for specifying local model weights. The tokenizer is 
loaded from the same directory as the checkpoint with the name ‘tokenizer.model’ unless separately specified.  
This example obtains the model parameters by name matching to known models because ‘stories15M’ is one of the 
models known to torchchat with a configuration stories in ‘torchchat/model_params’:


### Example 1: Basic Text Generation


```
python3 torchchat.py generate \
 --checkpoint-path stories15M/stories15M.pt \
 --prompt "Hello, my name is"
```


### Example 2: Providing Additional Artifacts
The following is an example of how to specify a local model checkpoint, the model architecture, and a tokenizer file:
```
python3 torchchat.py generate \
 --prompt "Once upon a time" \
 --checkpoint-path stories15M/stories15M.pt \
 --params-path stories15M/model.json \
 --tokenizer-path stories15M/tokenizer.model
```


Alternatively, we can specify the known architecture configuration for known models using ‘--params-table’ 
to specify a p[particular architecture in the ‘torchchat/model_params’:

```
python3 torchchat.py generate \
 --prompt "Once upon a time" \
 --checkpoint-path stories15M/stories15M.pt \
 --params-table stories15M \
 --tokenizer-path stories15M//tokenizer.model
```


## Using GGUF Models
Torchchat supports loading models in GGUF format using the `--gguf-file`. Refer to GGUF.md for additional 
documentation about using GGUF files in torchchat.

The GGUF format is compatible with several quantization levels such as F16, F32, Q4_0, and Q6_K. Model 
configuration information is obtained directly from the GGUF file, simplifying setup and obviating the 
need for a separate `model.json` model architecture specification.


## Using local models
Torchchat supports all commands such as chat, browser, server and export using local models. (In fact, 
known models simply download and populate the parameters specified for local models.) 
Here is an example setup for running a server with a local model:


[skip default]: begin
```
python3 torchchat.py server --checkpoint-path stories15M/stories15M.pt
```
[skip default]: end


[shell default]: python3 torchchat.py server --checkpoint-path stories15M/stories15M.pt & server_pid=$! ; sleep 90 # wait for server to be ready to accept requests


In another terminal, query the server using `curl`. Depending on the model configuration, this query might take a few minutes to respond.


> [!NOTE]
> Since this feature is under active development, not every parameter is consumed. See `#api/api.pyi` for details on
> which request parameters are implemented. If you encounter any issues, please comment on the [tracking Github issue](https://github.com/pytorch/torchchat/issues/973).


<details>


<summary>Example Query</summary>
Setting `stream` to "true" in the request emits a response in chunks. If `stream` is unset or not "true", then the client will 
await the full response from the server.


**Example: using the server**
A model server used witha local model works like any other torchchat server.  You can test it by sending a request with ‘curl’:
```
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1",
    "stream": "true",
    "max_tokens": 200,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```


[shell default]: kill ${server_pid}


</details>


For more information about using different commands, see the root README.md and refer to the Advanced Users Guide for further details on advanced configurations and parameter tuning.


[end default]: end

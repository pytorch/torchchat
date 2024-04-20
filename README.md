# Easy to Use LLM Runner for PC, Mobile and Embedded Devices
Torchchat is an easy to use library for leveraging LLMs on edge devices including mobile phones and desktops.

## Highlights
- Command line interaction with popular LLMs such as Llama2, Llama3, Stories, Mistral and more
  - Supporting both GGUF fp32/16 and the HF Format
- Supports popular hardware/OS
  - Linux (x86)
  - Mac OS (M1/M2/M3)
  - Android (devices that handle XNNPACK)
  - iOS 17+ (iPhone 13 pro+)
- Multiple Data Types including: float32, float16, bfloat16
- Multiple Quantization Schemes
- Multiple Execution Modes including: Eager, AOTI and ExecuTorch

## Quick Start
### Initialize the Environment
The following steps requires you have [Python 3.10](https://www.python.org/downloads/release/python-3100/) and [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) installed

```
# set up a virtual environment
python3 -m virtualenv .venv/torchchat
source .venv/torchchat/bin/activate

# get the code and dependencies
git clone https://github.com/pytorch/torchchat.git
cd torchchat
pip install -r requirements.txt

# ensure everything installed correctly. If this command works you'll see a welcome message and some details
python torchchat.py --help

```

### Generating Text

```
python torchchat.py generate stories15M
```
That’s all there is to it!
Read on to learn how to use the full power of torchchat.

## Customization
For the full details on all commands and parameters run `python torchchat.py --help`

### Download
For supported models, torchchat can download model weights. Most models use HuggingFace as the distribution channel, so you will need to create a HuggingFace
account and install `huggingface-cli`.

To install `huggingface-cli`, run `pip install huggingface-cli`. After installing, create a user access token [as documented here](https://huggingface.co/docs/hub/en/security-tokens). Run `huggingface-cli login`, which will prompt for the newly created token. Once this is done, torchchat will be able to download model artifacts from
HuggingFace.

```
python torchchat.py download llama2
```

### Chat
Designed for interactive and conversational use.
In chat mode, the LLM engages in a back-and-forth dialogue with the user. It responds to queries, participates in discussions, provides explanations, and can adapt to the flow of conversation. This mode is typically what you see in applications aimed at simulating conversational partners or providing customer support.

For more information run `python torchchat.py chat --help`

**Examples**
```
# Chat with some parameters
```

### Generate
Aimed at producing content based on specific prompts or instructions.
In generate mode, the LLM focuses on creating text based on a detailed prompt or instruction. This mode is often used for generating written content like articles, stories, reports, or even creative writing like poetry.

For more information run `python torchchat.py generate --help`

**Examples**
```
python torchchat.py generate llama2 --device=cpu --dtype=fp16
```

### Export
Compiles a model for different use cases

For more information run `python torchchat.py export --help`

**Examples**

```
python torchchat.py export stories15M --output-pte-path=stories15m.pte
```

### Browser
Run a chatbot in your browser that’s supported by the model you specify in the command

**Examples**

```
python torchchat.py browser --device cpu --checkpoint-path ${MODEL_PATH} --temperature 0 --num-samples 10
```

*Running on http://127.0.0.1:5000* should be printed out on the terminal. Click the link or go to [http://127.0.0.1:5000](http://127.0.0.1:5000) on your browser to start interacting with it.

Enter some text in the input box, then hit the enter key or click the “SEND” button. After 1 second or 2, the text you entered together with the generated text will be displayed. Repeat to have a conversation.

### Eval
Uses lm_eval library to evaluate model accuracy on a variety of tasks. Defaults to wikitext and can be manually controlled using the tasks and limit args.l

For more information run `python torchchat.py eval --help`

**Examples**
Eager mode:
```
# Eval example for Mac with some parameters
python -m torchchat.py eval --device cuda --checkpoint-path ${MODEL_PATH} -d fp32 --limit 5
```

To test the perplexity for lowered or quantized model, pass it in the same way you would to generate.py:

```
python3 -m torchchat.py eval --pte <pte> -p <params.json> -t <tokenizer.model> --limit 5
```
## Models
These are the supported models
| Model | Mobile Friendly | Notes |
|------------------|---|---------------------|
|[tinyllamas/stories15M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||
|[tinyllamas/stories42M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||
|[tinyllamas/stories110M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||
|[openlm-research/open_llama_7b](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||
|[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)|✅||
|[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)|||
|[meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)|||
|[meta-llama/CodeLlama-7b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf)|✅||
|[meta-llama/CodeLlama-34b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-34b-Python-hf)|✅||
|[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)|✅||
|[mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)|✅||
|[mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)|✅||
|[meta-llama/Llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)|✅||

See the [documentation on GGUF](docs/GGUF.md) to learn how to use GGUF files.

**Examples**
```
#Llama3
```

```
#Stories
```

```
#CodeLama
```

## Desktop Execution

### AOTI (AOT Inductor ) - PC Specific
AOT compiles models into machine code before execution, enhancing performance and predictability. It's particularly beneficial for frequently used models or those requiring quick start times. AOTI also increases security by not exposing the model at runtime. However, it may lead to larger binary sizes and lacks the runtime optimization flexibility

**Examples**
The following example uses the Stories15M model.
```
TODO: Update after the CLI gets fixed. Use real paths so user can copy paste

# Compile
python torchchat export --checkpoint-path ${MODEL_PATH} --device {cuda,cpu} --output-dso-path ${MODEL_OUT}/${MODEL_NAME}.so

# Execute
python torchchat generate --device {cuda,cpu} --dso-path ${MODEL_OUT}/${MODEL_NAME}.so --prompt "Hello my name is"
```

NOTE: The exported model will be large. We suggest you quantize the model, explained further down, before deploying the model for use.

### ExecuTorch
ExecuTorch enables you to optimize your model for execution on a mobile or embedded device

If you want to deploy and execute a model within your iOS app <do this>
If you want to deploy and execute a model within your Android app <do this>
If you want to deploy and execute a model within your edge device <do this>
If you want to experiment with our sample apps. Check out our iOS and Android sample apps.

## Quantization
Quantization focuses on reducing the precision of model parameters and computations from floating-point to lower-bit integers, such as 8-bit integers. This approach aims to minimize memory requirements, accelerate inference speeds, and decrease power consumption, making models more feasible for deployment on edge devices with limited computational resources. While quantization can potentially degrade the model's performance, the methods supported by torchchat are designed to mitigate this effect, maintaining a balance between efficiency and accuracy.

TODO:
- Brief rundown on supported quant modes and torchchat.py flags (emphasis on brief).
- Recommendations for quantization modes for 7b local chat, 7b on mobile, etc.
- One line that shows the performance difference between the base model and the 4bit
- Link to Quantization.md.

Read the [Quantization documention](docs/quantization.md) for more details.

## Mobile Execution
**Prerequisites**

Install [ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup.html)

[iOS Details](docs/iOS.md)

[Android Details](docs/Android.md)

# Chat with LLMs Everywhere
Torchchat is an easy-to-use library for running large language models (LLMs) on edge devices including mobile phones and desktops.

## Highlights
- Command line interaction with popular LLMs such as Llama 3, Llama 2, Stories, Mistral and more
  - Supporting both GGUF fp32/16 and the Hugging Face checkpoint format
- PyTorch-native execution with performance
- Supports popular hardware and OS
  - Linux (x86)
  - Mac OS (M1/M2/M3)
  - Android (Devices that support XNNPACK)
  - iOS 17+ (iPhone 13 Pro+)
- Multiple data types including: float32, float16, bfloat16
- Multiple quantization schemes
- Multiple execution modes including: Eager, Compile, AOT Inductor (AOTI) and ExecuTorch

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
bash ./install_requirements.sh

# ensure everything installed correctly
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
python torchchat.py download llama3
```

### Chat
Designed for interactive and conversational use.
In chat mode, the LLM engages in a back-and-forth dialogue with the user. It responds to queries, participates in discussions, provides explanations, and can adapt to the flow of conversation.

For more information run `python torchchat.py chat --help`

**Examples**
```
python torchchat.py chat llama3 --tiktoken
```

### Generate
Aimed at producing content based on specific prompts or instructions.
In generate mode, the LLM focuses on creating text based on a detailed prompt or instruction. This mode is often used for generating written content like articles, stories, reports, or even creative writing like poetry.

For more information run `python torchchat.py generate --help`

**Examples**
```
python torchchat.py generate llama3 --dtype=fp16 --tiktoken
```

### Export
Compiles a model and saves it to run later.

For more information run `python torchchat.py export --help`

**Examples**

AOT Inductor:
```
python torchchat.py export stories15M --output-dso-path stories15M.so
```

ExecuTorch:
```
python torchchat.py export stories15M --output-pte-path stories15M.pte
```

### Browser
Run a chatbot in your browser that’s supported by the model you specify in the command

**Examples**

```
python torchchat.py browser stories15M --temperature 0 --num-samples 10
```

*Running on http://127.0.0.1:5000* should be printed out on the terminal. Click the link or go to [http://127.0.0.1:5000](http://127.0.0.1:5000) on your browser to start interacting with it.

Enter some text in the input box, then hit the enter key or click the “SEND” button. After 1 second or 2, the text you entered together with the generated text will be displayed. Repeat to have a conversation.

### Eval
Uses lm_eval library to evaluate model accuracy on a variety of tasks. Defaults to wikitext and can be manually controlled using the tasks and limit args.

For more information run `python torchchat.py eval --help`

**Examples**

Eager mode:
```
python torchchat.py eval stories15M -d fp32 --limit 5
```

To test the perplexity for lowered or quantized model, pass it in the same way you would to generate:

```
python torchchat.py eval stories15M --pte-path stories15M.pte --limit 5
```

## Models
These are the supported models
| Model | Mobile Friendly | Notes |
|------------------|---|---------------------|
|[meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)|✅||
|[meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)|✅||
|[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)|✅||
|[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)|||
|[meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)|||
|[meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)|✅||
|[meta-llama/CodeLlama-7b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf)|✅||
|[meta-llama/CodeLlama-34b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-34b-Python-hf)|✅||
|[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)|✅||
|[mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)|✅||
|[mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)|✅||
|[tinyllamas/stories15M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||
|[tinyllamas/stories42M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||
|[tinyllamas/stories110M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||
|[openlm-research/open_llama_7b](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅||

See the [documentation on GGUF](docs/GGUF.md) to learn how to use GGUF files.

**Examples**

```
# Llama 3 8B Instruct
python torchchat.py chat llama3 --tiktoken
```

```
# Stories 15M
python torchchat.py chat stories15M
```

```
# CodeLama 7B for Python
python torchchat.py chat codellama
```

## Desktop Execution

### AOTI (AOT Inductor)
AOT compiles models into machine code before execution, enhancing performance and predictability. It's particularly beneficial for frequently used models or those requiring quick start times. However, it may lead to larger binary sizes and lacks the runtime flexibility of eager mode.

**Examples**
The following example uses the Stories15M model.
```
# Compile
python torchchat.py export stories15M --output-dso-path stories15M.so

# Execute
python torchchat.py generate --dso-path stories15M.so --prompt "Hello my name is"
```

NOTE: The exported model will be large. We suggest you quantize the model, explained further down, before deploying the model on device.

### ExecuTorch
ExecuTorch enables you to optimize your model for execution on a mobile or embedded device, but can also be used on desktop for testing.

**Examples**
The following example uses the Stories15M model.
```
# Compile
python torchchat.py export stories15M --output-pte-path stories15M.pte

# Execute
python torchchat.py generate --device cpu --pte-path stories15M.pte --prompt "Hello my name is"
```

See below under Mobile Execution if you want to deploy and execute a model in your iOS or Android app.

## Quantization
Quantization focuses on reducing the precision of model parameters and computations from floating-point to lower-bit integers, such as 8-bit and 4-bit integers. This approach aims to minimize memory requirements, accelerate inference speeds, and decrease power consumption, making models more feasible for deployment on edge devices with limited computational resources. While quantization can potentially degrade the model's performance, the methods supported by torchchat are designed to mitigate this effect, maintaining a balance between efficiency and accuracy.

TODO:
- Brief rundown on supported quant modes and torchchat.py flags (emphasis on brief).
- Recommendations for quantization modes for 7b local chat, 7b on mobile, etc.
- One line that shows the performance difference between the base model and the 4bit
- Link to Quantization.md.

Read the [quantization documention](docs/quantization.md) for more details.

## Mobile Execution
**Prerequisites**

ExecuTorch lets you run your model on a mobile or embedded device. The exported ExecuTorch .pte model file plus runtime is all you need.

Install [ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup.html) to get started.

Read the [iOS documentation](docs/iOS.md) for more details on iOS.

Read the [Android documentation](docs/Android.md) for more details on Android.

## Acknowledgements
Thank you to the [community](docs/ACKNOWLEDGEMENTS.md) for all the awesome libraries and tools
you've built around local LLM inference.

## License
Torchchat is released under the [BSD 3 license](LICENSE). However you may have other legal obligations
that govern your use of content, such as the terms of service for third-party models.

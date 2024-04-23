# Chat with LLMs Everywhere
Torchchat is a small codebase to showcase running large language models (LLMs) within Python OR within your own (C/C++) application on mobile (iOS/Android), desktop and servers.

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
- Multiple execution modes including: Python (Eager, Compile) or Native (AOT Inductor (AOTI), ExecuTorch)


## Installation


The following steps require that you have [Python 3.10](https://www.python.org/downloads/release/python-3100/) installed.

```
# get the code
git clone https://github.com/pytorch/torchchat.git
cd torchchat

# set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
./install_requirements.sh

# ensure everything installed correctly
python3 torchchat.py --help
```

### Download Weights
Most models use HuggingFace as the distribution channel, so you will need to create a HuggingFace
account.

Create a HuggingFace user access token [as documented here](https://huggingface.co/docs/hub/en/security-tokens).
Run `huggingface-cli login`, which will prompt for the newly created token.  

Once this is done, torchchat will be able to download model artifacts from
HuggingFace.

```
python3 torchchat.py download llama3
```

## What can you do with torchchat?

* Run models via PyTorch / Python:
  * [Chat](#chat)
  * [Generate](#generate)
  * [Run via Browser](#browser)
* [Quantizing your model (suggested for mobile)](#quantization)
* Export and run models in native environments (C++, your own app, mobile, etc.)
  * [Exporting for desktop/servers via AOTInductor](#export-server)
  * [Running exported .so file via your own C++ application](#run-server)
     * in Chat mode
     * in Generate mode
  * [Exporting for mobile via ExecuTorch](#export-executorch)
     * in Chat mode
     * in Generate mode
  * [Running exported executorch file on iOS or Android](#run-mobile)

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


## Running via PyTorch / Python

### Chat
Designed for interactive and conversational use.
In chat mode, the LLM engages in a back-and-forth dialogue with the user. It responds to queries, participates in discussions, provides explanations, and can adapt to the flow of conversation.

For more information run `python3 torchchat.py chat --help`

**Examples**
```
python3 torchchat.py chat llama3 --tiktoken
```

### Generate
Aimed at producing content based on specific prompts or instructions.
In generate mode, the LLM focuses on creating text based on a detailed prompt or instruction. This mode is often used for generating written content like articles, stories, reports, or even creative writing like poetry.

For more information run `python3 torchchat.py generate --help`

**Examples**
```
python3 torchchat.py generate llama3 --dtype=fp16 --tiktoken
```

## Exporting your model
Compiles a model and saves it to run later.

For more information run `python3 torchchat.py export --help`

### Exporting for Desktop / Server-side via AOT Inductor

```
python3 torchchat.py export stories15M --output-dso-path stories15M.so
```

This produces a `.so` file, also called a Dynamic Shared Object. This `.so` can be linked into your own C++ program.

### Running the exported `.so` via your own C++ application

[TBF]

### Exporting for Mobile via ExecuTorch

```
python3 torchchat.py export stories15M --output-pte-path stories15M.pte
```

### Browser
Run a chatbot in your browser that’s supported by the model you specify in the command.

**Examples**

```
python3 torchchat.py browser stories15M --temperature 0 --num-samples 10
```

*Running on http://127.0.0.1:5000* should be printed out on the terminal. Click the link or go to [http://127.0.0.1:5000](http://127.0.0.1:5000) on your browser to start interacting with it.

Enter some text in the input box, then hit the enter key or click the “SEND” button. After a second or two, the text you entered together with the generated text will be displayed. Repeat to have a conversation.

### Eval
Uses lm_eval library to evaluate model accuracy on a variety of tasks. Defaults to wikitext and can be manually controlled using the tasks and limit args.

For more information run `python3 torchchat.py eval --help`

**Examples**

Eager mode:
```
python3 torchchat.py eval stories15M -d fp32 --limit 5
```

To test the perplexity for a lowered or quantized model, pass it in the same way you would to generate:

```
python3 torchchat.py eval stories15M --pte-path stories15M.pte --limit 5
```

## Models
The following models are the supported by torchchat:
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
python3 torchchat.py chat llama3 --tiktoken
```

```
# Stories 15M
python3 torchchat.py chat stories15M
```

```
# CodeLama 7B for Python
python3 torchchat.py chat codellama
```

## Desktop Execution

### AOTI (AOT Inductor)
AOT compiles models into machine code before execution, enhancing performance and predictability. It's particularly beneficial for frequently used models or those requiring quick start times. However, it may lead to larger binary sizes and lacks the runtime flexibility of eager mode.

**Examples**
The following example uses the Stories15M model.
```
# Compile
python3 torchchat.py export stories15M --output-dso-path stories15M.so

# Execute
python3 torchchat.py generate --dso-path stories15M.so --prompt "Hello my name is"
```

NOTE: The exported model will be large. We suggest you quantize the model, explained further down, before deploying the model on device.

### ExecuTorch

ExecuTorch enables you to optimize your model for execution on a mobile or embedded device, but can also be used on desktop for testing.
Before running ExecuTorch commands, you must first set-up ExecuTorch in torchchat, see [Set-up Executorch](docs/executorch_setup.md).

**Examples**
The following example uses the Stories15M model.
```
# Compile
python3 torchchat.py export stories15M --output-pte-path stories15M.pte

# Execute
python3 torchchat.py generate --device cpu --pte-path stories15M.pte --prompt "Hello my name is"
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

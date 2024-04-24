# Chat with LLMs Everywhere
Torchchat is a compact codebase to showcase the capability of running large language models (LLMs) seamlessly across diverse platforms. With Torchchat, you could run LLMs from with Python, your own (C/C++) application on mobile (iOS/Android), desktop or servers.

## Highlights
- Command line interaction with popular LLMs such as Llama 3, Llama 2, Stories, Mistral and more
  - Supports [common GGUF formats](docs/GGUF.md) and the Hugging Face checkpoint format
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

```bash
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
Most models use HuggingFace as the distribution channel, so you will need to create a HuggingFace account.

Create a HuggingFace user access token [as documented here](https://huggingface.co/docs/hub/en/security-tokens).
Run `huggingface-cli login`, which will prompt for the newly created token.

Once this is done, torchchat will be able to download model artifacts from
HuggingFace.

```
python3 torchchat.py download llama3
```

View available models with `python3 torchchat.py list`. You can also remove downloaded models
with `python3 torchchat.py remove llama3`.

## What can you do with torchchat?

* Run models via PyTorch / Python:
  * [Chat](#chat)
  * [Generate](#generate)
  * [Run via Browser](#browser)
* [Quantizing your model (suggested for mobile)](#quantizing-your-model-suggested-for-mobile)
* Export and run models in native environments (C++, your own app, mobile, etc.)
  * [Export for desktop/servers via AOTInductor](#export-server)
  * [Run exported .so file via your own C++ application](#run-server)
     * in Chat mode
     * in Generate mode
  * [Export for mobile via ExecuTorch](#export-executorch)
     * in Chat mode
     * in Generate mode
  * [Run exported ExecuTorch file on iOS or Android](#run-mobile)


## Running via PyTorch / Python

### Chat
Designed for interactive and conversational use.
In chat mode, the LLM engages in a back-and-forth dialogue with the user. It responds to queries, participates in discussions, provides explanations, and can adapt to the flow of conversation.

**Examples**
```bash
python3 torchchat.py chat llama3
```

For more information run `python3 torchchat.py chat --help`

### Generate
Aimed at producing content based on specific prompts or instructions.
In generate mode, the LLM focuses on creating text based on a detailed prompt or instruction. This mode is often used for generating written content like articles, stories, reports, or even creative writing like poetry.


**Examples**
```bash
python3 torchchat.py generate llama3
```

For more information run `python3 torchchat.py generate --help`

### Browser

Designed for interactive graphical conversations using the familiar web browser GUI.  The browser command provides a GUI-based experience to engage with the LLM in a back-and-forth dialogue with the user. It responds to queries, participates in discussions, provides explanations, and can adapt to the flow of conversation.

## Quantizing your model (suggested for mobile)

Quantization is the process of converting a model into a more memory-efficient representation.  Quantization is particularly important for accelerators -- to take advantage of the available memory bandwidth, and fit in the often limited high-speed memory in accelerators – and mobile devices – to fit in the typically very limited memory of mobile devices.


Depending on the model and the target device, different quantization recipes may be applied.  Torchchat contains two example configurations to optimize performance for GPU-based systems `config/data/cuda.json` , and mobile systems `config/data/mobile.json`.  The GPU configuration is targeted towards optimizing for memory bandwidth which is a scarce resource in powerful GPUs (and to a less degree, memory footprint to fit large models into a device's memory).  The mobile configuration is targeted towards optimizing for memory fotoprint because in many devices, a single application is limited to as little as GB or less of memory.

You can use the quantization recipes in conjunction with any of the `chat`, `generate` and `browser` commands to test their impact and accelerate model execution. You will apply these recipes to the export comamnds below, to optimize the exported models.  To adapt these recipes or wrote your own, please refer to the [quantization overview](docs/quantization.md). 

---
*TO BE REPLACED BY SUITABLE ORDING PROVIDED BY LEGAL*
With quantization, 32-bit floating numbers can be represented with as few as 8 or even 4 bits, and a scale shared by a group of these weights.  This transformation is lossy and modifies the behavior of models.  While research is being conducted on how to efficiently quantize large language models for use in mobile devices, this transformation invariable results in both quality loss and a reduced amount of control over the output of the models, leading to an increased risk of undesirable responses, hallucinations and stuttering.  In effect an a developer quantizing a model, has much control and even more responsibility to quantize a model to quantify and reduce these effects.
---


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

The following models are supported by torchchat and have associated aliases. Other models, including GGUF format, can be run by specifying a URL directly.

| Model | Mobile Friendly | Notes |
|------------------|---|---------------------|
|[meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)|✅|Tuned for `chat` . Alias to `llama3`.|
|[meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)|✅|Best for `generate`. Alias to `llama3-base`.|
|[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)|✅|Tuned for `chat`. Alias to `llama2`.|
|[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)||Tuned for `chat`. Alias to `llama2-13b-chat`.|
|[meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)||Tuned for `chat`. Alias to `llama2-70b-chat`.|
|[meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)|✅|Best for `generate`. Alias to `llama2-base`.|
|[meta-llama/CodeLlama-7b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf)|✅|Tuned for Python and `generate`. Alias to `codellama`.|
|[meta-llama/CodeLlama-34b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-34b-Python-hf)|✅|Tuned for Python and `generate`. Alias to `codellama-34b`.|
|[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)|✅|Best for `generate`. Alias to `mistral-7b-v01-base`.|
|[mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)|✅|Tuned for `chat`. Alias to `mistral-7b-v01-instruct`.|
|[mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)|✅|Tuned for `chat`. Alias to `mistral`.|
|[tinyllamas/stories15M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅|Toy model for `generate`. Alias to `stories15M`.|
|[tinyllamas/stories42M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅|Toy model for `generate`. Alias to `stories42M`.|
|[tinyllamas/stories110M](https://huggingface.co/karpathy/tinyllamas/tree/main)|✅|Toy model for `generate`. Alias to `stories110M`.|
|[openlm-research/open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b)|✅|Best for `generate`. Alias to `open-llama`.|

Torchchat also supports loading of many models in the GGUF format. See the [documentation on GGUF](docs/GGUF.md) to learn how to use GGUF files.

**Examples**

```
# Llama 3 8B Instruct
python3 torchchat.py chat llama3 --dtype fp16
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

See below under [Mobile Execution](#run-mobile) if you want to deploy and execute a model in your iOS or Android app.


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

## Fine-tuned models from torchtune

torchchat supports running inference with models fine-tuned using [torchtune](https://github.com/pytorch/torchtune). To do so, we first need to convert the checkpoints into a format supported by torchchat.

Below is a simple workflow to run inference on a fine-tuned Llama3 model. For more details on how to fine-tune Llama3, see the instructions [here](https://github.com/pytorch/torchtune?tab=readme-ov-file#llama3)

```bash
# install torchtune
pip install torchtune

# download the llama3 model
tune download meta-llama/Meta-Llama-3-8B \
    --output-dir ./Meta-Llama-3-8B \
    --hf-token <ACCESS TOKEN>

# Run LoRA fine-tuning on a single device. This assumes the config points to <checkpoint_dir> above
tune run lora_finetune_single_device --config llama3/8B_lora_single_device

# convert the fine-tuned checkpoint to a format compatible with torchchat
python3 build/convert_torchtune_checkpoint.py \
  --checkpoint-dir ./Meta-Llama-3-8B \
  --checkpoint-files meta_model_0.pt \
  --model-name llama3_8B \
  --checkpoint-format meta

# run inference on a single GPU
python3 torchchat.py generate \
  --checkpoint-path ./Meta-Llama-3-8B/model.pth \
  --device cuda
```

## Acknowledgements
Thank you to the [community](docs/ACKNOWLEDGEMENTS.md) for all the awesome libraries and tools
you've built around local LLM inference.

## License
Torchchat is released under the [BSD 3 license](LICENSE). However you may have other legal obligations
that govern your use of content, such as the terms of service for third-party models.
![image](https://github.com/pytorch/torchchat/assets/61328285/1cfccb53-c025-43d7-8475-94b34cf92339)

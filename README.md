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

NOTE: This command may prompt you to request access to llama3 via HuggingFace, if you do not already have access. Simply follow the prompts and re-run the command when access is granted.

View available models with `python3 torchchat.py list`. You can also remove downloaded models
with `python3 torchchat.py remove llama3`.

### Common Issues

* **CERTIFICATE_VERIFY_FAILED**:
  Run `pip install --upgrade certifi`.
* **Access to model is restricted and you are not in the authorized list. Visit \[link\] to ask for access**:
  Some models require an additional step to access. Follow the link to fill out the request form on HuggingFace.

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
  * [Export for mobile via ExecuTorch](#exporting-for-mobile-via-executorch)
  * [Run exported ExecuTorch file on iOS or Android](#mobile-execution)
     * in Chat mode
     * in Generate mode
  * Fine-tuned models from torchtune
  

## Running via PyTorch / Python

### Chat
Designed for interactive and conversational use.
In chat mode, the LLM engages in a back-and-forth dialogue with the user. It responds to queries, participates in discussions, provides explanations, and can adapt to the flow of conversation.

**Examples**

```bash
# Llama 3 8B Instruct
python3 torchchat.py chat llama3 
```

```
# CodeLama 7B for Python
python3 torchchat.py chat codellama
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

**Examples**

```
python3 torchchat.py browser llama3 --temperature 0 --num-samples 10
```

*Running on http://127.0.0.1:5000* should be printed out on the terminal. Click the link or go to [http://127.0.0.1:5000](http://127.0.0.1:5000) on your browser to start interacting with it.

Enter some text in the input box, then hit the enter key or click the “SEND” button. After a second or two, the text you entered together with the generated text will be displayed. Repeat to have a conversation.



## Quantizing your model (suggested for mobile)

Quantization is the process of converting a model into a more memory-efficient representation.  Quantization is particularly important for accelerators -- to take advantage of the available memory bandwidth, and fit in the often limited high-speed memory in accelerators – and mobile devices – to fit in the typically very limited memory of mobile devices.

Depending on the model and the target device, different quantization recipes may be applied. Torchchat contains two example configurations to optimize performance for GPU-based systems `config/data/qconfig_gpu.json`, and mobile systems `config/data/qconfig_mobile.json`. The GPU configuration is targeted towards optimizing for memory bandwidth which is a scarce resource in powerful GPUs (and to a less degree, memory footprint to fit large models into a device's memory). The mobile configuration is targeted towards optimizing for memory fotoprint because in many devices, a single application is limited to as little as GB or less of memory.

You can use the quantization recipes in conjunction with any of the `chat`, `generate` and `browser` commands to test their impact and accelerate model execution. You will apply these recipes to the `export` comamnds below, to optimize the exported models. For example:
```
python3 torchchat.py chat llama3 --quantize config/data/qconfig_gpu.json
```
To adapt these recipes or wrote your own, please refer to the [quantization overview](docs/quantization.md).

*TO BE REPLACED BY SUITABLE ORDING PROVIDED BY LEGAL:*

With quantization, 32-bit floating numbers can be represented with as few as 8 or even 4 bits, and a scale shared by a group of these weights.  This transformation is lossy and modifies the behavior of models.  While research is being conducted on how to efficiently quantize large language models for use in mobile devices, this transformation invariable results in both quality loss and a reduced amount of control over the output of the models, leading to an increased risk of undesirable responses, hallucinations and stuttering.  In effect an a developer quantizing a model, has much control and even more responsibility to quantize a model to quantify and reduce these effects.

## Desktop Execution

### AOTI (AOT Inductor)
AOT compiles models into machine code before execution, enhancing performance and predictability. It's particularly beneficial for frequently used models or those requiring quick start times. However, it may lead to larger binary sizes and lacks the runtime flexibility of eager mode.

**Examples**
The following example uses the Llama3 8B model.
```
# Compile
python3 torchchat.py export llama3 --output-dso-path llama3.so

# Execute
python3 torchchat.py generate llama3 --quantize config/data/qconfig_gpu.json--dso-path llama3.so --prompt "Hello my name is"
```

NOTE: We use `--quantize config/data/qconfig_gpu.json` to quantize the llama3 model to reduce model size and improve performance for on-device use cases.

**Build Native Runner Binary**

We provide an end-to-end C++ [runner](runner/run.cpp) that runs the `*.so` file exported after following the previous [examples](#aoti-aot-inductor) section. To build the runner binary on your Mac or Linux:

```bash
scripts/build_native.sh aoti
```

Run:

```bash
cmake-out/aoti_run model.so -z tokenizer.model -i "Once upon a time"
```

### ExecuTorch

ExecuTorch enables you to optimize your model for execution on a mobile or embedded device, but can also be used on desktop for testing.
Before running ExecuTorch commands, you must first set-up ExecuTorch in torchchat, see [Set-up Executorch](docs/executorch_setup.md).

**Examples**
The following example uses the Llama3 8B model.
```
# Compile
python3 torchchat.py export llama3 --quantize config/data/qconfig_mobile.json --output-pte-path llama3.pte

# Execute
python3 torchchat.py generate llama3 --device cpu --pte-path llama3.pte --prompt "Hello my name is"
```
NOTE: We use `--quantize config/data/qconfig_mobile.json` to quantize the llama3 model to reduce model size and improve performance for on-device use cases.

See below under [Mobile Execution](#mobile-execution) if you want to deploy and execute a model in your iOS or Android app.



## Mobile Execution
**Prerequisites**

ExecuTorch lets you run your model on a mobile or embedded device. The exported ExecuTorch .pte model file plus runtime is all you need.

Install [ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup.html) to get started.

Read the [iOS documentation](docs/iOS.md) for more details on iOS.

Read the [Android documentation](docs/Android.md) for more details on Android.

**Build Native Runner Binary**

We provide an end-to-end C++ [runner](runner/run.cpp) that runs the `*.pte` file exported after following the previous [ExecuTorch](#executorch) section. Notice that this binary is for demo purpose, please follow the respective documentations, to see how to build a similar application on iOS and Android. To build the runner binary on your Mac or Linux:

```bash
scripts/build_native.sh et
```

Run:

```bash
cmake-out/et_run llama3.pte -z tokenizer.model -i "Once upon a time"
```

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

### Eval
Uses the lm_eval library to evaluate model accuracy on a variety of tasks. Defaults to wikitext and can be manually controlled using the tasks and limit args.

For more information run `python3 torchchat.py eval --help`

**Examples**

Eager mode:
```
python3 torchchat.py eval llama3 -d fp32 --limit 5
```

To test the perplexity for a lowered or quantized model, pass it in the same way you would to generate:

```
python3 torchchat.py eval llama3 --pte-path llama3.pte --limit 5
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

While we describe how to use torchchat using the popular llama3 model, you can perform the example commands with any of these models.



## Acknowledgements
Thank you to the [community](docs/ACKNOWLEDGEMENTS.md) for all the awesome libraries and tools
you've built around local LLM inference.

* Georgi Gerganov and his [GGML](https://github.com/ggerganov/ggml)
  project shining a spotlight on community-based enablement and
  inspiring so many other projects.

* Andrej Karpathy and his
  [llama2.c](https://github.com/karpathy/llama2.c) project.  So many
  great (and simple!) ideas in llama2.c that we have directly adopted
  (both ideas and code) from his repo.  You can never go wrong by
  following Andrej's work.

* Michael Gschwind, Bert Maher, Scott Wolchok, Bin Bao, Chen Yang,
  Huamin Li and Mu-Chu Li who built the first version of nanogpt (`DSOGPT`)
  with AOT Inductor proving that AOTI can be used to build efficient
  LLMs, and DSOs are a viable distribution format for models.
  [nanoGPT](https://github.com/karpathy/nanoGPT).

* Bert Maher and his
  [llama2.so](https://github.com/bertmaher/llama2.so), which built on
  Andrej's llama2.c and on DSOGPT to close the loop on Llama models
  with AOTInductor.

* Christian Puhrsch, Horace He, Joe Isaacson and many more for their
  many contributions in Accelerating GenAI models in the *"Anything,
  Fast!"* pytorch.org blogs, and, in particular, Horace He for [GPT,
  Fast!](https://github.com/pytorch-labs/gpt-fast), which we have
  directly adopted (both ideas and code) from his repo.

* Mobius Labs as the authors of the HQQ quantization algorithms
  included in this distribution.


## License
Torchchat is released under the [BSD 3 license](LICENSE). (Additional code in this 
distribution is covered by the MIT and Apache Open Source licenses.) However you may have other legal obligations
that govern your use of content, such as the terms of service for third-party models.
![image](https://github.com/pytorch/torchchat/assets/61328285/1cfccb53-c025-43d7-8475-94b34cf92339)

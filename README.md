# Chat with LLMs Everywhere
torchchat is a compact codebase showcasing the ability to run large language models (LLMs) seamlessly. With torchchat, you can run LLMs using Python, within your own (C/C++) application (desktop or server) and on iOS and Android.



## What can you do with torchchat?
- [Setup the Repo](#installation)
- [Download Models](#download-weights)
- [Run models via PyTorch / Python](#running-via-pytorch--python)
  - [Chat](#chat)
  - [Generate](#generate)
  - [Run chat in the Browser](#browser)
- [Export models for running on desktop/server without python](#desktopserver-execution)
  - [Use AOT Inductor for faster execution](#aoti-aot-inductor)
  - [Running in c++ using the runner](#running-native-using-our-c-runner)
- [Run on mobile](#mobile-execution)
  - [Setup](#set-up-executorch)
  - [Export a model for use on mobile](#export-for-mobile)
  - [Deploy and run on iOS](#deploy-and-run-on-ios)
  - [Deploy and run on Android](#deploy-and-run-on-android)
- [Evaluate a mode](#eval)
- [Fine-tuned models from torchtune](#fine-tuned-models-from-torchtune)
- [Supported Models](#models)
- [Troubleshooting](#troubleshooting)


## Highlights
- Command line interaction with popular LLMs such as Llama 3, Llama 2, Stories, Mistral and more
  - Supports [common GGUF formats](docs/GGUF.md) and the Hugging Face checkpoint format
- PyTorch-native execution with performance
- Supports popular hardware and OS
  - Linux (x86)
  - Mac OS (M1/M2/M3)
  - Android (Devices that support XNNPACK)
  - iOS 17+ (iPhone 13 Pro+)
- Multiple data types including: float32, float16, bfloat16, select GGUF data types
- Multiple quantization schemes
- Multiple execution modes including: Python (Eager, Compile) or Native (AOT Inductor (AOTI), ExecuTorch)

### Disclaimer
The torchchat Repository Content is provided without any guarantees about performance or compatibility. In particular, torchchat makes available model architectures written in Python for PyTorch that may not perform in the same manner or meet the same standards as the original versions of those models. When using the torchchat Repository Content, including any model architectures, you are solely responsible for determining the appropriateness of using or redistributing the torchchat Repository Content and assume any risks associated with your use of the torchchat Repository Content or any models, outputs, or results, both alone and in combination with any other technologies. Additionally, you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models, weights, data, or other technologies, and you are solely responsible for complying with all such obligations.


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
Log into huggingface:
```
huggingface-cli login
```

Once this is done, torchchat will be able to download model artifacts from
HuggingFace.

```
python3 torchchat.py download llama3
```

*NOTE: This command may prompt you to request access to llama3 via HuggingFace, if you do not already have access. Simply follow the prompts and re-run the command when access is granted.*

View available models with:
```
python3 torchchat.py list
```

You can also remove downloaded models with the remove command: `python3 torchchat.py remove llama3`

## Running via PyTorch / Python
[Follow the installation steps if you haven't](#installation)

### Chat
```bash
# Llama 3 8B Instruct
python3 torchchat.py chat llama3
```

For more information run `python3 torchchat.py chat --help`

### Generate
```bash
python3 torchchat.py generate llama3
```

For more information run `python3 torchchat.py generate --help`

### Browser

```
python3 torchchat.py browser llama3 --temperature 0 --num-samples 10
```

*Running on http://127.0.0.1:5000* should be printed out on the terminal. Click the link or go to [http://127.0.0.1:5000](http://127.0.0.1:5000) on your browser to start interacting with it.

Enter some text in the input box, then hit the enter key or click the “SEND” button. After a second or two, the text you entered together with the generated text will be displayed. Repeat to have a conversation.



## Desktop/Server Execution

### AOTI (AOT Inductor)
AOT compiles models before execution for faster inference

The following example exports and executes the Llama3 8B Instruct model
```
# Compile
python3 torchchat.py export llama3 --output-dso-path llama3.so

# Execute the exported model using Python
python3 torchchat.py generate llama3 --quantize config/data/cuda.json --dso-path llama3.so --prompt "Hello my name is"
```

NOTE: We use `--quantize config/data/cuda.json` to quantize the llama3 model to reduce model size and improve performance for on-device use cases.

### Running native using our C++ Runner

The end-to-end C++ [runner](runner/run.cpp) runs an `*.so` file exported in the previous step.

To build the runner binary on your Mac or Linux:
```bash
scripts/build_native.sh aoti
```

Execute
```bash
cmake-out/aoti_run model.so -z tokenizer.model -l 3 -i "Once upon a time"
```

## Mobile Execution
ExecuTorch enables you to optimize your model for execution on a mobile or embedded device, but can also be used on desktop for testing.

### Set Up Executorch
Before running any commands in torchchat that require ExecuTorch, you must first install ExecuTorch.

To install ExecuTorch, run the following commands *from the torchchat root directory*.
This will download the ExecuTorch repo to ./et-build/src and install various ExecuTorch libraries to ./et-build/install.
```
export TORCHCHAT_ROOT=${PWD}
export ENABLE_ET_PYBIND=true
./scripts/install_et.sh $ENABLE_ET_PYBIND
```

### Export for mobile
The following example uses the Llama3 8B Instruct model.
```
# Export
python3 torchchat.py export llama3 --quantize config/data/mobile.json --output-pte-path llama3.pte

# Execute
python3 torchchat.py generate llama3 --device cpu --pte-path llama3.pte --prompt "Hello my name is"
```
NOTE: We use `--quantize config/data/mobile.json` to quantize the llama3 model to reduce model size and improve performance for on-device use cases.

For more details on quantization and what settings to use for your use case visit our [Quanitization documentation](docs/quantization.md) or run `python3 torchchat.py export`

### Deploy and run on iOS
The following assumes you've completed the steps for [Setting up Executorch](#set-up-executorch) and

Open the xcode project
```
open et-build/src/executorch/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj
```
Then click the Play button to launch the app in Simulator.

To run on a device, given that you already have it set up for development, you'll need to have a provisioning profile with the [`increased-memory-limit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit) entitlement. Just change the app's bundle identifier to whatever matches your provisioning profile with the aforementioned capability enabled.

After the app launched successfully, copy an exported ExecuTorch model (`.pte`) and tokenizer (`.bin`) files to the iLLaMA folder.

For the Simulator, just drap&drop both files onto the Simulator window and save at `On My iPhone > iLLaMA` folder.

For a device, open it in a separate Finder window, navigate to the Files tab, drag&drop both files to the iLLaMA folder and wait till the copying finishes.

Now, follow the app's UI guidelines to pick the model and tokenizer files from the local filesystem and issue a prompt.

*Click the image below to see it in action!*
<a href="https://pytorch.org/executorch/main/_static/img/llama_ios_app.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/llama_ios_app.png" width="600" alt="iOS app running a LlaMA model">
</a>

### Deploy and run on Android


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

torchchat also supports loading of many models in the GGUF format. See the [documentation on GGUF](docs/GGUF.md) to learn how to use GGUF files.

While we describe how to use torchchat using the popular llama3 model, you can perform the example commands with any of these models.

## Troubleshooting

**CERTIFICATE_VERIFY_FAILED**:
Run `pip install --upgrade certifi`.

**Access to model is restricted and you are not in the authorized list.**
Some models require an additional step to access. Follow the link provided in the error to get access.

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
torchchat is released under the [BSD 3 license](LICENSE). (Additional code in this
distribution is covered by the MIT and Apache Open Source licenses.) However you may have other legal obligations
that govern your use of content, such as the terms of service for third-party models.

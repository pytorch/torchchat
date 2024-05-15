# Chat with LLMs Everywhere

torchchat is a small codebase showcasing the ability to run large language models (LLMs) seamlessly. With torchchat, you can run LLMs using Python, within your own (C/C++) application (desktop or server) and on iOS and Android.


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
- [Fine-tuned models from torchtune](docs/torchtune.md)
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

## Installation
The following steps require that you have [Python 3.10](https://www.python.org/downloads/release/python-3100/) installed.

[skip default]: begin
```bash
# get the code
git clone https://github.com/pytorch/torchchat.git
cd torchchat

# set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate
```
[skip default]: end
```
# install dependencies
./install_requirements.sh
```

Installations can be tested by

```bash
# ensure everything installed correctly
python3 torchchat.py --help
```

### Download Weights
Most models use HuggingFace as the distribution channel, so you will need to create a HuggingFace account.

[prefix default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}"
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

*NOTE: This command may prompt you to request access to llama3 via
 HuggingFace, if you do not already have access. Simply follow the
 prompts and re-run the command when access is granted.*

View available models with:
```
python3 torchchat.py list
```

Query the location of a particular model -- this is particularly useful in scripts when you do not want to hard-code paths:
```
python3 torchchat.py where llama3
```

Finally, you can also remove downloaded models with the remove command:
`python3 torchchat.py remove llama3`


## Running via PyTorch / Python
[Follow the installation steps if you haven't](#installation)

### Chat
[skip default]: begin
```bash
# Llama 3 8B Instruct
python3 torchchat.py chat llama3
```
[skip default]: end

For more information run `python3 torchchat.py chat --help`

### Generate
```bash
python3 torchchat.py generate llama3 --prompt "write me a story about a boy and his bear"
```

For more information run `python3 torchchat.py generate --help`


### Browser

[skip default]: begin
```
python3 torchchat.py browser llama3
```
[skip default]: end


*Running on http://127.0.0.1:5000* should be printed out on the
 terminal. Click the link or go to
 [http://127.0.0.1:5000](http://127.0.0.1:5000) on your browser to
 start interacting with it.

Enter some text in the input box, then hit the enter key or click the
“SEND” button. After a second or two, the text you entered together
with the generated text will be displayed. Repeat to have a
conversation.



## Desktop/Server Execution

### AOTI (AOT Inductor)
AOT compiles models before execution for faster inference

The following example exports and executes the Llama3 8B Instruct
model.  The first command performs the actual export, the second
command loads the exported model into the Python interface to enable
users to test the exported model.

```
# Compile
python3 torchchat.py export llama3 --output-dso-path exportedModels/llama3.so

# Execute the exported model using Python

python3 torchchat.py generate llama3 --dso-path exportedModels/llama3.so --prompt "Hello my name is"
```

NOTE: If your machine has cuda add this flag for performance
`--quantize config/data/cuda.json`

### Running native using our C++ Runner

The end-to-end C++ [runner](runner/run.cpp) runs a [DSO](https://en.wikipedia.org/wiki/Shared_library)  model (represented by a file with extension `.so`)
exported in the previous step.

To build the runner binary on your Mac or Linux:
```bash
scripts/build_native.sh aoti
```

[skip default]: begin
Execute
```bash
cmake-out/aoti_run exportedModels/llama3.so -z `python3 torchchat.py where llama3`/tokenizer.model -l 3 -i "Once upon a time"
```
[skip default]: end

## Mobile Execution

ExecuTorch enables you to optimize your model for execution on a
mobile or embedded device, but can also be used on desktop for
testing.

### Set Up Executorch

Before running any commands in torchchat that require ExecuTorch, you
must first install ExecuTorch.

To install ExecuTorch, run the following commands *from the torchchat
root directory*.  This will download the ExecuTorch repo to
./et-build/src and install various ExecuTorch libraries to
./et-build/install.

```
export TORCHCHAT_ROOT=${PWD}
./scripts/install_et.sh
```

### Export for mobile
The following example uses the Llama3 8B Instruct model.

```
# Export
python3 torchchat.py export llama3 --quantize config/data/mobile.json --output-pte-path llama3.pte

# Execute
python3 torchchat.py generate llama3 --device cpu --pte-path llama3.pte --prompt "Hello my name is"
```

NOTE: We use `--quantize config/data/mobile.json` to quantize the
llama3 model to reduce model size and improve performance for
on-device use cases.

For more details on quantization and what settings to use for your use
case visit our [Quanitization documentation](docs/quantization.md) or
run `python3 torchchat.py export`

[end default]: end

### Deploy and run on iOS

The following assumes you've completed the steps for [Setting up
Executorch](#set-up-executorch) and

Open the xcode project
```
open et-build/src/executorch/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj
```
Then click the Play button to launch the app in Simulator.

To run on a device, given that you already have it set up for
development, you'll need to have a provisioning profile with the
[`increased-memory-limit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit)
entitlement. Just change the app's bundle identifier to whatever
matches your provisioning profile with the aforementioned capability
enabled.

After the app launched successfully, copy an exported ExecuTorch model (`.pte`) and tokenizer (`.bin`) files to the iLLaMA folder.

For the Simulator, just drag&drop both files onto the Simulator window and save at `On My iPhone > iLLaMA` folder.

For a device, open it in a separate Finder window, navigate to the Files tab, drag&drop both files to the iLLaMA folder and wait till the copying finishes.

Now, follow the app's UI guidelines to pick the model and tokenizer files from the local filesystem and issue a prompt.

*Click the image below to see it in action!*
<a href="https://pytorch.org/executorch/main/_static/img/llama_ios_app.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/llama_ios_app.png" width="600" alt="iOS app running a LlaMA model">
</a>


### Deploy and run on Android

#### Approach 1: Android Studio

If you have Android Studio set up, and you have Java 17 and Android SDK 34 configured, you can follow this step.

First, you need to download the following AAR file which contains the required Java library and its corresponding JNI library, for the app to build and run. You need to put the file to `android/Torchchat/app/libs/executorch.aar`

[executorch-llama.aar](https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.2/executorch-llama.aar) (SHASUM: 09d17f7bc59589b581e45bb49511d19196d0297d)

```
curl https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.2/executorch-llama.aar -o android/Torchchat/app/libs/executorch.aar --create-dirs
echo "09d17f7bc59589b581e45bb49511d19196d0297d  android/Torchchat/app/libs/executorch.aar" | shasum --check
```

You also need to push the model and tokenizer file to your device. Please refer to the docs above on generating the pte and bin file, or use E2E script (see section below) to generate and push the file.

```
adb shell mkdir -p /data/local/tmp/llama
adb push build/android/model.pte /data/local/tmp/llama
adb push build/android/tokenizer.bin /data/local/tmp/llama
```

Now, you can open the torchchat app skeleton, which is located at `android/Torchchat`. Use Android Studio to open this directory.

Then, click the Play button (^R) to launch it to emulator/device.

Now, follow the app's UI guidelines to pick the model and tokenizer files from the local filesystem and issue a prompt.

<img src="https://pytorch.org/executorch/main/_static/img/android_llama_app.png" width="600" alt="Android app running a LlaMA model">

#### Approach 2: E2E Script

Alternatively, you can run `scripts/android_example.sh` which sets up Java, Android SDK Manager, Android SDK, Android emulator, builds the app, and launches it for you.

```
export TORCHCHAT_ROOT=$(pwd)
sh scripts/android_example.sh
```


### Eval

Uses the lm_eval library to evaluate model accuracy on a variety of
tasks. Defaults to wikitext and can be manually controlled using the
tasks and limit args.

See [Evaluation](docs/evaluation.md)

For more information run `python3 torchchat.py eval --help`

**Examples**

Eager mode:
```
python3 torchchat.py eval llama3 --dtype fp32 --limit 5
```

To test the perplexity for a lowered or quantized model, pass it in
the same way you would to generate:

```
python3 torchchat.py eval llama3 --pte-path llama3.pte --limit 5
```



## Models

The following models are supported by torchchat and have associated
aliases. Other models, including GGUF format, can be run by specifying
a URL directly.

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

torchchat also supports loading of many models in the GGUF format. See
the [documentation on GGUF](docs/GGUF.md) to learn how to use GGUF
files.

While we describe how to use torchchat using the popular llama3 model,
you can perform the example commands with any of these models.


## Design Principles

torchchat embodies PyTorch’s design philosophy [[details](https://pytorch.org/docs/stable/community/design.html)], especially "usability over everything else".

### Native PyTorch

torchchat is a native-PyTorch library. While we provide integrations with the surrounding ecosystem (eg: Hugging Face models, etc), all of the core functionality is written in PyTorch.

### Simplicity and Extensibility

torchchat is designed to be easy to understand, use and extend.

- Composition over implementation inheritance - layers of inheritance for code re-use makes the code hard to read and extend
- No training frameworks - explicitly outlining the training logic makes it easy to extend for custom use cases
- Code duplication is preferred over unnecessary abstractions
- Modular building blocks over monolithic components

### Correctness

torchchat provides well-tested components with a high-bar on correctness.
We provide

- Extensive unit-tests to ensure things operate as they should

## Community Contributions

We really value our community and the contributions made by our wonderful users. We'll use this section to call out some of these contributions! If you'd like to help out as well, please see the [CONTRIBUTING](docs/CONTRIBUTING.md) guide.

## Troubleshooting


**CERTIFICATE_VERIFY_FAILED**
Run `pip install --upgrade certifi`.


**Access to model is restricted and you are not in the authorized list**
Some models require an additional step to access. Follow the
link provided in the error to get access.

**Installing ET Fails**
If `./scripts/install_et.sh` fails with an error like `Building wheel for executorch (pyproject.toml) did not run successfully` It's possible that it's linking to an older version of pytorch installed some other way like via homebrew. You can break the link by uninstalling other versions such as `brew uninstall pytorch` Note: You may break something that depends on this, so be aware.


### Disclaimer
The torchchat Repository Content is provided without any guarantees
about performance or compatibility. In particular, torchchat makes
available model architectures written in Python for PyTorch that may
not perform in the same manner or meet the same standards as the
original versions of those models. When using the torchchat Repository
Content, including any model architectures, you are solely responsible
for determining the appropriateness of using or redistributing the
torchchat Repository Content and assume any risks associated with your
use of the torchchat Repository Content or any models, outputs, or
results, both alone and in combination with any other
technologies. Additionally, you may have other legal obligations that
govern your use of other content, such as the terms of service for
third-party models, weights, data, or other technologies, and you are
solely responsible for complying with all such obligations.


### Disclaimer
The torchchat Repository Content is provided without any guarantees about
performance or compatibility. In particular, torchchat makes available
model architectures written in Python for PyTorch that may not perform
in the same manner or meet the same standards as the original versions
of those models. When using the torchchat Repository Content, including
any model architectures, you are solely responsible for determining the
appropriateness of using or redistributing the torchchat Repository Content
and assume any risks associated with your use of the torchchat Repository Content
or any models, outputs, or results, both alone and in combination with
any other technologies. Additionally, you may have other legal obligations
that govern your use of other content, such as the terms of service for
third-party models, weights, data, or other technologies, and you are
solely responsible for complying with all such obligations.


## Acknowledgements
Thank you to the [community](docs/ACKNOWLEDGEMENTS.md) for all the
awesome libraries and tools you've built around local LLM inference.

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

torchchat is released under the [BSD 3 license](LICENSE). (Additional
code in this distribution is covered by the MIT and Apache Open Source
licenses.) However you may have other legal obligations that govern
your use of content, such as the terms of service for third-party
models.

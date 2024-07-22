# Chat with LLMs Everywhere

torchchat is a small codebase showcasing the ability to run large language models (LLMs) seamlessly. With torchchat, you can run LLMs using Python, within your own (C/C++) application (desktop or server) and on iOS and Android.


## What can you do with torchchat?
- [Run models via PyTorch / Python](#running-via-pytorch--python)
  - [Chat](#chat)
  - [Generate](#generate)
  - [Run chat in the Browser](#browser)
- [Run models on desktop/server without python](#desktopserver-execution)
  - [Use AOT Inductor for faster execution](#aoti-aot-inductor)
  - [Running in c++ using the runner](#running-native-using-our-c-runner)
- [Run models on mobile](#mobile-execution)
  - [Deploy and run on iOS](#deploy-and-run-on-ios)
  - [Deploy and run on Android](#deploy-and-run-on-android)
- [Evaluate a model](#eval)


## Highlights
- Command line interaction with popular LLMs such as Llama 3, Llama 2, Stories, Mistral and more
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

[skip default]: begin
```bash
# get the code
git clone https://github.com/pytorch/torchchat.git
cd torchchat

# set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
./install_requirements.sh
```
[skip default]: end

[shell default]: ./install_requirements.sh

Installations can be tested by running

```bash
# ensure everything installed correctly
python3 torchchat.py --help
```

### Download Weights
Most models use Hugging Face as the distribution channel, so you will need to create a Hugging Face account.
Create a Hugging Face user access token [as documented here](https://huggingface.co/docs/hub/en/security-tokens) with the `write` role.

Log into Hugging Face:

[prefix default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}"

```
huggingface-cli login
```

Once this is done, torchchat will be able to download model artifacts from
Hugging Face.

```
python3 torchchat.py download llama3
```

> [!NOTE]
> This command may prompt you to request access to Llama 3 via
> Hugging Face, if you do not already have access. Simply follow the
> prompts and re-run the command when access is granted.*


<details>
<summary>Additional Model Inventory Management Commands</summary>

### List
This subcommands shows the available models
```bash
python3 torchchat.py list
```

### Where
This subcommands shows location of a particular model. 
```bash
python3 torchchat.py list
```
This is useful in scripts when you do not want to hard-code paths


### Remove
This subcommands removes the specified model
```bash
python3 torchchat.py remove llama3
```

More information about these commands can be found by adding the `--help` option.

</details>


## Running via PyTorch / Python

> [!TIP]
> For more information about these commands, please refer to the `--help` menu.

### Chat
This mode allows you to chat with an LLM in an interactive fashion.

[skip default]: begin
```bash
# Llama 3 8B Instruct
python3 torchchat.py chat llama3
```
[skip default]: end

### Generate
This mode generates text based on an input prompt.
```bash
python3 torchchat.py generate llama3 --prompt "write me a story about a boy and his bear"
```

### Browser
This mode allows you to chat with the model using a UI in your browser
Running the command automatically open a tab in your browser.

[skip default]: begin

```
streamlit run torchchat.py -- browser llama3
```

[skip default]: end

### Server
**Note: This feature is still in progress and not all endpoints are working ATM**


<details>
<summary>This mode gives a REST API that matches the OpenAI API spec for interacting with a model</summary>

To test out the REST API, **you'll need 2 terminals**: one to host the server, and one to send the request.


In one terminal, kick off the server

[skip default]: begin

```bash
python3 torchchat.py server llama3
```
[skip default]: end

In the other terminal window, interact with the API using curl. Depending on the model configuration, this query might take a few minutes to respond
  
**Example Input + Output**

```
curl http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
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
```
{"response":" I'm a software developer with a passion for building innovative and user-friendly applications. I have experience in developing web and mobile applications using various technologies such as Java, Python, and JavaScript. I'm always looking for new challenges and opportunities to learn and grow as a developer.\n\nIn my free time, I enjoy reading books on computer science and programming, as well as experimenting with new technologies and techniques. I'm also interested in machine learning and artificial intelligence, and I'm always looking for ways to apply these concepts to real-world problems.\n\nI'm excited to be a part of the developer community and to have the opportunity to share my knowledge and experience with others. I'm always happy to help with any questions or problems you may have, and I'm looking forward to learning from you as well.\n\nThank you for visiting my profile! I hope you find my information helpful and interesting. If you have any questions or would like to discuss any topics, please feel free to reach out to me. I"}
```

</details>


## Desktop/Server Execution

### AOTI (AOT Inductor)
[AOTI](https://pytorch.org/blog/pytorch2-2/) compiles models before execution for faster inference.

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

> [!NOTE]
> If your machine has cuda add this flag for performance
`--quantize config/data/cuda.json` when exporting. You'll also need to tell generate to use `--device cuda`

### Running native using our C++ Runner

The end-to-end C++ [runner](runner/run.cpp) runs a [DSO](https://en.wikipedia.org/wiki/Shared_library)  model (represented by a file with extension `.so`)
exported in the previous step.

To build the runner binary on your Mac or Linux:
```bash
scripts/build_native.sh aoti
```

Execute
```bash
cmake-out/aoti_run exportedModels/llama3.so -z `python3 torchchat.py where llama3`/tokenizer.model -l 3 -i "Once upon a time"
```

## Mobile Execution

[ExecuTorch](https://github.com/pytorch/executorch) enables you to optimize your model for execution on a
mobile or embedded device.

### Set Up ExecuTorch

Before running any commands in torchchat that require ExecuTorch, you
must first install ExecuTorch.

To install ExecuTorch, run the following commands.  This will download the
ExecuTorch repo to ./et-build/src and install various ExecuTorch libraries to
./et-build/install.

> [!IMPORTANT]
> The following commands should be run from the torchchat root directory.

```
export TORCHCHAT_ROOT=${PWD}
./scripts/install_et.sh
```


### Export for mobile
Similar to AOTI, to deploy onto device, we first export the PTE artifact, then we load the artifact for inference.

The following example uses the Llama3 8B Instruct model.
```
# Export
python3 torchchat.py export llama3 --quantize config/data/mobile.json --output-pte-path llama3.pte
```

> [!NOTE]
> We use `--quantize config/data/mobile.json` to quantize the
llama3 model to reduce model size and improve performance for
on-device use cases.

For more details on quantization and what settings to use for your use
case visit our [Quantization documentation](docs/quantization.md).

### Deploy and run on Desktop

While ExecuTorch does not focus on desktop inference, it is capable
of doing so. This is handy for testing out PTE
models without sending them to a physical device.

Specifically there are 2 ways of doing so: Pure Python and via a Runner

<details>
<summary>Deploying via Python</summary>

```
# Execute
python3 torchchat.py generate llama3 --device cpu --pte-path llama3.pte --prompt "Hello my name is"
```

</details>


<details>
<summary>Deploying via a Runner</summary>

Build the runner
```bash
scripts/build_native.sh et
```

Execute using the runner
```bash
cmake-out/et_run llama3.pte -z `python3 torchchat.py where llama3`/tokenizer.model -l 3 -i "Once upon a time"
```

</details>


[end default]: end

### Deploy and run on iOS

The following assumes you've completed the steps for [Setting up ExecuTorch](#set-up-executorch).

<details>
<summary>Deploying with Xcode</summary>

#### Requirements
- Xcode 15.0 or later
- A development provisioning profile with the [`increased-memory-limit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit) entitlement.


#### Steps

1. Open the Xcode project:
    ```bash
    open et-build/src/executorch/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj
    ```
2. Click the Play button to launch the app in the Simulator.

3. To run on a device, ensure you have it set up for development and a provisioning profile with the `increased-memory-limit` entitlement. Update the app's bundle identifier to match your provisioning profile with the required capability.

4. After successfully launching the app, copy the exported ExecuTorch model (`.pte`) and tokenizer (`.model`) files to the iLLaMA folder. You can find the model file called `llama3.pte` in the current `torchchat` directory and the tokenizer file at `$(python3 torchchat.py where llama3)/tokenizer.model` path.

    - **For the Simulator:** Drag and drop both files onto the Simulator window and save them in the `On My iPhone > iLLaMA` folder.
    - **For a device:** Open a separate Finder window, navigate to the Files tab, drag and drop both files into the iLLaMA folder, and wait for the copying to finish.

5. Follow the app's UI guidelines to select the model and tokenizer files from the local filesystem and issue a prompt.

*Click the image below to see it in action!*

<a href="https://pytorch.org/executorch/main/_static/img/llama_ios_app.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/llama_ios_app.png" width="600" alt="iOS app running a LlaMA model">
</a>
</details>


### Deploy and run on Android

The following assumes you've completed the steps for [Setting up ExecuTorch](#set-up-executorch). In torchchat, we show 2 approaches for Android deployment:

<details>
<summary>Approach 1 (Recommended): Android Studio</summary>


If you have Android Studio set up, and you have [Java 17](https://developer.android.com/build/jdks) and [Android SDK 34](https://developer.android.com/about/versions/14/setup-sdk) configured, and [adb](https://developer.android.com/tools/adb) set up, you can follow this step.

First, you need to download the AAR file which contains the required Java library and its corresponding JNI library, for the app to build and run. You need to create directory `android/Torchchat/app/libs/` if it does not exist. You need to rename the downloaded AAR file to `executorch.aar` and move the file to `android/Torchchat/app/libs/`.

[executorch-llama-tiktoken-rc3-0719.aar](https://ossci-android.s3.amazonaws.com/executorch/main/executorch-llama-tiktoken-rc3-0719.aar) (SHASUM: c3e5d2a97708f033c2b1839a89f12f737e3bbbef)

> Note: The AAR file listed above comes with tiktoken tokenizer, which is used for llama3 model. If you want to use a model with BPE tokenizer (llama2 model for example), you can download this AAR: [executorch-llama-bpe-rc3-0719.aar](https://ossci-android.s3.amazonaws.com/executorch/main/executorch-llama-bpe-rc3-0719.aar) (SHASUM: d5fe81d9a4700c36b50ae322e6bf34882134edb0)
>
> Currently the tokenizer is built at compile time, so you need to re-build the app when you need to use a different tokenizer for different model.
> 
> The script to build the AAR can be found [here](https://github.com/pytorch/executorch/blob/main/build/build_android_library.sh). If you need to tweak with the tokenizer or runtime (for example use your own tokenizer or runtime library), you can modify the ExecuTorch code and use that script to build the AAR library. 

You also need to push the model and tokenizer file to your device. Please refer to the docs above on generating the .pte and .model/.bin file, or use E2E script (see section below) to generate and push the file.

```
adb shell mkdir -p /data/local/tmp/llama
adb push <model.pte> /data/local/tmp/llama
adb push <tokenizer.model or tokenizer.bin> /data/local/tmp/llama
```

Now, you can open the torchchat app skeleton, which is located at `android/Torchchat`. Use Android Studio to open this directory.

Then, click the Play button (^R) to launch it to emulator/device.

> Note: We recommend you to use a device with at least 12GB RAM and 20GB storage. If you use an emulated device, you can see [this post](https://stackoverflow.com/questions/45517553/cant-change-the-ram-size-in-avd-manager-android-studio) on setting the RAM.

Now, follow the app's UI guidelines to pick the model and tokenizer files from the local filesystem and issue a prompt.

<img src="https://pytorch.org/executorch/main/_static/img/android_llama_app.png" width="600" alt="Android app running a LlaMA model">

</details>
<details>
<summary>Approach 2: E2E Script</summary>

Alternatively, you can run `scripts/android_example.sh` which sets up Java, Android SDK Manager, Android SDK, Android emulator (if no physical device is found), builds the app, and launches it for you. It can be used if you don't have a GUI.

```
export TORCHCHAT_ROOT=$(pwd)
export USE_TIKTOKEN=ON # Set this only for tiktoken tokenizer
sh scripts/android_example.sh
```

</details>

## Eval

Uses the lm_eval library to evaluate model accuracy on a variety of
tasks. Defaults to wikitext and can be manually controlled using the
tasks and limit args.

See [Evaluation](docs/evaluation.md)

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
aliases.

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

While we describe how to use torchchat using the popular llama3 model,
you can perform the example commands with any of these models.


## Design Principles

torchchat embodies PyTorch’s design philosophy [details](https://pytorch.org/docs/stable/community/design.html), especially "usability over everything else".

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

We really value our community and the contributions made by our wonderful users. We'll use this section to call out some of these contributions! If you'd like to help out as well, please see the [CONTRIBUTING](CONTRIBUTING.md) guide.

## Troubleshooting


**CERTIFICATE_VERIFY_FAILED**
Run `pip install --upgrade certifi`.


**Access to model is restricted and you are not in the authorized list**
Some models require an additional step to access. Follow the
link provided in the error to get access.

**Installing ET Fails**
If `./scripts/install_et.sh` fails with an error like `Building wheel for executorch (pyproject.toml) did not run successfully` It's possible that it's linking to an older version of pytorch installed some other way like via homebrew. You can break the link by uninstalling other versions such as `brew uninstall pytorch` Note: You may break something that depends on this, so be aware.


## Disclaimer
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


## Acknowledgements
Thank you to the community for all the
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


## License

torchchat is released under the [BSD 3 license](LICENSE). (Additional
code in this distribution is covered by the MIT and Apache Open Source
licenses.) However you may have other legal obligations that govern
your use of content, such as the terms of service for third-party
models.

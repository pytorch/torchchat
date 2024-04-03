# Preamble.

*The statements contained in this README are our northstar, and we will be reality-testing the statemen, and remove any
items that are not factual.  If you find an item, that is incorrect, please tag as an issue, so we can triage and determine whether to fix,
or drop from our initial release.*

# llama-fast *NORTHSTAR*
A repo for building and using llama on servers, desktops and mobile

The llama-fast repo enables model inference of llama models (and other LLMs) on servers, desktop and mobile devices.
For a list of devices, see below, under *SUPPORTED SYSTEMS*

A goal of this repo, and the design of the PT2 components was to offer seamless integration and consistent workflows.
Both mobile and server/desktop paths start with torch.export() receiving the same model description.  Similarly,
integration into runners for Python (for initial testing) and Python-free environments (for deployment, in runner-aoti
and runner-et, respectively) offer a consistent experience across backends and offer developers consistent interfaces
and user experience whether they target server, desktop or mobile & edge use cases, and/or all of them.


# Simple and efficient pytorch-native transformer text generation.

Featuring:

* Very low latency
* <1000 lines of python
* No dependencies other than PyTorch and sentencepiece for server, and Executorch for mobile (plus, your mobile IDE, of course)
* int8/int4 quantization
* Supports Nvidia and AMD GPUs, Apple GPUs with MPS, CPU (Linux/x86 and MacOS/ARM), and xnnpack, Vulkan and MPS for mobile GPUs,
  and backend-specific mobile runtimes ("delegates", such as CoreML and Hexagon).

This is NOT intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :)
Please copy-paste and fork as you desire.

# Supported Models
The model definition (and much more!) is adopted from gpt-fast, so we support the same models.

To download llama models, go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access.
Then, login with `huggingface-cli login`

## Downloading Weights
Models tested/supported

| Model | eager | torch.compile | AOT Inductor | ET Runtime | Fits on Mobile |
|-----|------|-----|-----|-----|-----|
tinyllamas/stories15M | ✅ |  ✅ |  ✅ |  ✅ | ✅ |
tinyllamas/stories42M  | ✅ |  ✅ |  ✅ |  ✅ | ✅ |
tinyllamas/stories110M   | ✅ |  ✅ |  ✅ |  ✅ | ✅ |
openlm-research/open_llama_7b  | ✅ |  ✅ |  ✅ |  ✅ | ❹ |
meta-llama/Llama-2-7b-chat-hf | ✅ |  ✅ |  ✅ |  ✅ | ❹|
meta-llama/Llama-2-13b-chat-hf | ✅ |  ✅ |  ✅ |  ✅ | 📵 |
meta-llama/Llama-2-70b-chat-hf | ✅ |  ✅ |  ✅ |  ✅ | ❌|
codellama/CodeLlama-7b-Python-hf | ✅ |  ✅ |  ✅ |  ✅ | ❹|
codellama/CodeLlama-34b-Python-hf | ✅ |  ✅ |  ✅ |  ✅ | ❌ |
mistralai/Mistral-7B-v0.1 | ✅ |  ✅ |  ✅ |  ✅ | ❹ |
mistralai/Mistral-7B-Instruct-v0.1 | ✅ |  ✅ |  ✅ |  ✅ | ❹ |
mistralai/Mistral-7B-Instruct-v0.2 | ✅ |  ✅ |  ✅ |  ✅ | ❹ |

*Key:* ✅ works correctly; ❌ not supported; ❹ requires 4bit groupwise quantization; 📵 not on mobile phone (may fit some high-end devices such as tablets);


For example, to convert Llama-2-7b-chat-hf
```bash
export MODEL_DOWNLOAD=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_DOWNLOAD
```

See [`gpt-fast` Supported Models](https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#supported-models) for a full list.

# Introduction

We use two variables in this example, which may be set as a preparatory step:

* `MODEL_NAME` describes the name of the model.  This name is *not* free-form, as it is used to index into a table
   of supported models and their configuration properties that are needed to load the model. This variable should correspond to the
   name of the directory holding the files for the corresponding model.  You *must* follow this convention to
   ensure correct operation.

* `MODEL_PATH` describes the location of the model. Throughput the description
  herein, we will assume that MODEL_PATH starts with a subdirectory of the llama-fast repo
  named checkpoints, and that it will contain the actual model. The modelname will thus
  be of the form checpoints/${MODEL_NAME}/model.{pt,pth}.  (Both the extensions `pt` and `pth`
  are used to describe checkpoints. In addition, model may be replaced with the name of the model)

  The generate.py  sequence generator will load the tokenizer from the directory specified by the MODEL_PATH variable,
  by replacing the modelname with the name of the tokenizer model which is expected to be named `tokenizer.model`

You can set these variables as follows for the exemplary model15M model from Andrej Karpathy's tinyllamas model family:
```
MODEL_NAME=stories15M
MODEL_PATH=checkpoints/${MODEL_NAME}/stories15M.pt
```

When we export models with AOT Inductor for servers and desktops, and Executorch for mobile and edge devices,
we will save them in the current working directory under the names ${MODEL_NAME}.so (for AOTI-generated dynamic libraries),
or ${MODEL_NAME}.pte (for Executorch-generated mobile/edge models).



## Installation

Follow the [`gpt-fast` installation instructions](https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#installation).

[Download PyTorch nightly](https://pytorch.org/get-started/locally/)
Install sentencepiece and huggingface_hub
```bash
pip install sentencepiece huggingface_hub
```

If you are planning on using mobile backends, you should also install ExecuTorch and any hardware-specific libraries and IDEs.

## A note on tokenizers

There are two different formats for tokenizers, and both are used in this repo.
1 - for generate.py and Python bindings, we use the Google sentencepiece Python operator. This operator consumes a tokenization model in the `tokenizer.model` format.
2 - for C/C++ inference, we use @Andrej Karpathy's C tokenizer function.  This tokenizer consumes a tokenization model in the 'tokenizer.bin' format.

If you are using coda, you can install sentencepiece using the following command:
```
conda install sentencepiece
```

You can convert tokenizer.model into tokenizer.bin using Andrej's
tokenizer.py utility to convert the tokenizer.model to tokenizer.bin
format:

```
python utils/tokenizer.py --tokenizer-model=/path/to/tokenizer/tokenizer.model
./run ./model.{so,pte} -z path/to/tokenizer/tokenizer.bin
```

# Generate Text

## Eager Execution

Model definition in model.py, generation code in generate.py. The
model checkpoint extension may have either the extension pth or pt.

```
python generate.py --compile --checkpoint_path ${MODEL_PATH} --prompt "Hello, my name is" --device {cuda,cpu,mps}
```

To squeeze out a little bit more performance, you can also compile the
prefill with --compile_prefill. This will increase compilation times
though.

## AOT Inductor compilation and execution
```
python export_aoti.py --checkpoint_path ${MODEL_PATH} --device {cuda,cpu} --output-path ./${MODEL_NAME}.so
```

When you have exported the model, you can test the model with the
sequence generator by importing the compiled DSO model with the
`-sopath ./{modelname}.so` option.  This gives users the ability to
test their model, run any pre-existing model tests against the
exported model with the same interface, and support additional
experiments to confirm model quality and speed.

```
python generate.py --device {cuda,cpu} --dso-path ./${MODEL_NAME}.so --prompt "Hello my name is"
```

While we have shown the export and execution of a small model on CPU
or an accelerator such as GPU, most models need to be compressed to
reduce their memory bandwidth requirements and avoid stalling the
execution engines while they are waiting for data.  We use
quantization to achieve this.


## ExecuTorch mobile compilation

### The basics

Use a small model like stories15M.pt to test the instructions in the
following section.  You must first have ExecuTorch installed before
running this command, see the installation instructions in the
sections [here](#installation-instructions).

The command below will add the file "${MODEL_NAME}.pte" to your
current directory.

```
python export_et.py --checkpoint_path ${MODEL_PATH} -d fp32 --output-path ./${MODEL_NAME}.pte
```

At present, when exporting a model, the export command always uses the
xnnpack delegate to export.  (Future versions will support additional
delegates such as CoreML, MPS, HTP in addition to Xnnpack.  ***fixme:
@scott currently not true.  Making this work is the next priority***

When you have exported the model, you can test the model with the
sequence generator by importing the compiled DSO model with the
`--pte-path ./${MODEL_NAME}.pte` option.  This gives users the ability to
test their model, run any pre-existing model tests against the
exported model with the same interface, and support additional
experiments to confirm model quality and speed.

```
python generate.py --checkpoint_path ${MODEL_PATH} --pte-path ${MODEL_NAME}.pte --prompt "Hello my name is" --device cpu
```

While we have shown the export and execution of a small model to a mobile/edge
device supported by Executorch, most models need to be compressed to
fit in the target device's memory. We use quantization to achieve this.

### Setting up runner-et
This is currently tested and works with no backend.

Before using runner-et, you must build executorch with cmake.  Calling the install_requirements.sh script in the executorch repo is not enough.

To build executorch with cmake, cd to the executorch repo and do the following steps.
```
rm -rf cmake-out; mkdir cmake-out
```
```
cmake -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=ON -S . -B cmake-out; cmake --build cmake-out
```

This will build executorch and place the library and object files in cmake-out.

After this is done, you can build runner-et in llama-fast.  Make sure to set DET_DIR to the executorch path on your machine.

```
rm -rf runner-et/cmake-out; mkdir -p runner-et/cmake-out; cmake -DET_DIR:STRING=/path/to/executorch -DCMAKE_BUILD_TYPE=Release -S runner-et -B runner-et/cmake-out; cmake --build runner-et/cmake-out
```

After this builds, the runner can be called with something like:


```
./runner-et/cmake-out/runner_et run llama-fast.pte -z stories15M/tokenizer.bin -i "Once upon a time"
```

# Optimizing your model for server, desktop and mobile devices

## Making your models fit and execute fast!

Next, we'll show you how to optimize your model for mobile execution
(for ET) or get the most from your server or desktop hardware (with
AOTI). The basic model build for mobile surfaces two issues: Models
quickly run out of memory and execution can be slow. In this section,
we show you how to fit your models in the limited memory of a mobile
device, and optimize execution speed -- both using quantization. This
is the `llama-fast` repo after all!

For high-performance devices such as GPUs, quantization provides a way
to reduce the memory bandwidth required to and take advantage of the
massive compute capabilities provided by today's server-based
accelerators such as GPUs. In addition to reducing the memory bandwidth required
to compute a result faster by avoiding stalls, quantization allows
accelerators (which usually have a limited amount of memory) to store and
process larger models than they would otherwise be able to.

#### Embedding quantization (8 bit integer, groupwise)
The simplest way to quantize embedding tables is with int8 groupwise quantization, where each value is represented by an 8 bit integer, and a
floating point scale per group:
```
python export_et.py --checkpoint_path ${MODEL_PATH} -d fp32 --quant "{'embedding': {'bitwidth': 8, 'group_size': 8} }" --output-path ${MODEL_NAME}_emb8b-gw256.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_NAME}_emb8b-gw256.pte --prompt "Hello my name is"
```

#### Linear 8 bit integer quantization (tested)
The simplest way to quantize is with int8 quantization, where each value is represented by an 8 bit integer, and a
floating point scale:
```
python export_et.py --checkpoint_path ${MODEL_PATH} -d fp32 --xnnpack_dynamic --output-path ${MODEL_NAME}.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ./${MODEL_NAME}.pte --prompt "Once upon a time" --checkpoint_path ${MODEL_PATH} --device cpu
```

#### 4 bit integer quantization (8da4w)
To compress your model even more, 4 bit integer quantization may be used.  To achieve good accuracy, we recommend the use
of groupwise quantization where (small to mid-sized) groups of int4 weights share a scale.  We also quantize activations to 8 bit, giving
this scheme its name (8da4w = 8b dynamically quantized activations with 4b weights), and boost performance.
```
python export_et.py --checkpoint_path ${MODEL_PATH} -d fp32 --quant "{'linear:8da4w': {'group_size' : 7} }"  8da4w --output-path ./${MODEL_NAME}_8da4w.pte
```

Now you can run your model with the same command as before:
```
python generate.py --ptr ./${MODEL_NAME}_8da4w.pte --prompt "Hello my name is"
```

#### Quantization with GPTQ (8da4w-gptq)
TBD.

#### Adding additional quantization schemes
We invite contributors to submit established quantization schemes, with accuracy and performance results demonstrating soundness.


# Standalone Execution

## Desktop and Server Execution
This has been tested with Linux and x86 (using CPU ~and GPU~), and MacOS and ARM/Apple Silicon.

**we must support GPUs, and test execution on them.**

The runner-* directories show how to integrate AOTI- and ET-exported models in a C/C++ application when no Python environment is available.  Integrate it with your own applications and adapt it to your own application and model needs!

Build the runner like this
```
cd ./runner-aoti
cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build build
```

To run, use the following command (assuming you already generated the tokenizer.bin tokenizer model):
```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./build/run ../${MODEL_NAME}.so -z ../${MODEL_NAME}.bin
```

## Mobile and Edge Execution Test (x86)
This has been shown to run on x86. with the proper IDE environment, you can compile for your specific target.
For a GUI integration in iOS and Android, please refer to...

Build the runner like this
```
cd ./runner-et
cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build build
```

To run your pte model, use the following command (assuming you already generated the tokenizer.bin tokenizer model):
```
./build/run ../${MODEL_NAME}{,_int8,_8da4w}.pte -z ../${MODEL_NAME}.bin
```

## Running on a mobile/edge system

### Android

Check out the [tutorial on how to build an Android app running your PyTorch models with Executorch](https://pytorch.org/executorch/main/llm/llama-demo-android.html), and give your llama-fast models a spin.

![Screenshot](https://pytorch.org/executorch/main/_static/img/android_llama_app.png "Android app running Llama model")

### iOS

Open the ios Llama Xcode project at https://github.com/pytorch/executorch/tree/main/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj in Xcode and click Run.
You will need to provide a provisioning profile (similar to what's expected for any iOS dev).

Once you can run the app on you device,
1 - connect the device to you Mac,
2 - copy the model and tokenizer.bin to the iOS Llama app
3 - select the tokenizer and model with the `(...)` control (bottom left of screen, to the left of the text entrybox)

# Supported Systems

PyTorch and the mobile Executorch backend support a broad range fo devices for running PyTorch with Python (using either eager or eager + torch.compile) or using a Python-free environment with AOT Inductor , as well as runtimes for executing exported models.


| Hardware | OS | eager | eager + compile | AOT compile | ET Runtime |
|-----|------|-----|-----|-----|-----|
| x86 | Linux | ✅ |  ✅ |  ✅ |  ✅ |
| x86 | macOS | ? | ? | ? | ? |
| aarch64 | Linux | ? | ? | ? | ? |
| aarch64 | macOS | ✅ |  ✅ |  ✅ |  ✅ |
| AMD GPU | Linux |  ✅ |  ✅ |  ✅ | ❌|
| Nvidia GPU | Linux | ✅ |  ✅ |  ✅ | ❌|
| MPS | macOS | ✅ |  ❌|  ❌|  ? |
| MPS | iOS | ❌|❌|❌| ✅ |
| aarch64 | iOS | ❌|❌|❌| ✅ |
| aarch64 | Android | ❌|❌|❌| ✅ |
| Mobile GPU (Vulkan) | Android |  ❌|❌|❌| ✅ |
| CoreML | iOS |  ❌|❌|❌| ✅ |
| Hexagon DSP | Android | ❌|❌|❌| ✅ |
| Raspberry Pi 4/5 | Raspbian | ? | ? | ? | ? |
| Raspberry Pi 4/5 | Android | ? | ? | ? | ? |
| ARM 32b (up to v7) | any | ❌|❌|❌|❌|


## Runtime performance with Llama 7B, in tokens per second (4b quantization)

| Hardware | OS | eager | eager + compile | AOT compile | ET Runtime |
|-----|------|-----|-----|-----|-----|
| x86 | Linux | ? | ? | ? | ? |
| x86 | macOS | ? | ? | ? | ? |
| aarch64 | Linux | ? | ? | ? | ? |
| aarch64 | macOS | ? | ? | ? | ? |
| AMD GPU | Linux | ? | ? | ? | ? |
| Nvidia GPU | Linux | ? | ? | ? | ? |
| MPS | macOS | ? | ? | ? | ? |
| MPS | iOS | ? | ? | ? | ? |
| aarch64 | Android | ? | ? | ? | ? |
| Mobile GPU (Vulkan) | Android | ? | ? | ? | ? |
| CoreML | iOS | | ? | ? | ? | ? |
| Hexagon DSP | Android | | ? | ? | ? | ? |
| Raspberry Pi 4/5 | Raspbian | ? | ? | ? | ? |
| Raspberry Pi 4/5 | Android | ? | ? | ? | ? |
| ARM 32b (up to v7) | any | | ? | ? | ? | ? |


## Installation Instructions

Some systems require additional installation steps.

Note: External libraries have not been tested for correctness, reliability and safety. Please contact your system vendor if you have system-specific questions.

### macOS (aarch64, x86)

To use torch.compile, you should install OpenMP and a compiler with suitable OpenMP support. You can install OpenMP using conda by following the PyTorch installation instructions
at https://github.com/pytorch/pytorch?tab=readme-ov-file#install-dependencies.

Alternatively, you can also find libraries here: https://mac.r-project.org/openmp/ and from other locations. Alternatively, you may install

macOS running on x86 is reaching end-of-life. To use PyTorch on x86 running macOS, you can download prebuilt binaries up to PyTorch 2.2.  You can download recent PyTorch releases and
install them from source.

### iOS CoreML and MPS

List dependencies for these backends

### ExecuTorch
Set up executorch by following the instructions [here](https://pytorch.org/executorch/stable/getting-started-setup.html#setting-up-executorch).

Make sure when you run the installation script in the executorch repo, you enable pybind.
```
./install_requirements.sh --pybind
```



# Acknowledgements

A big thank you to

* Georgi Gerganov and his [GGML](https://github.com/ggerganov/ggml) project that helped shine a spotlight
on community-based enablement, and inspired so many other projects.

* Andrej Karpathy and his [llama2.c](https://github.com/karpathy/llama2.c) project.  So many great (and simple!) ideas in llama2.c that we
have directly adopted (both ideas and code) from his repo.  You can never go wrong by following Andrej's work!

* my colleague and friend Bert Maher and [llama2.so](https://github.com/bertmaher/llama2.so) who build on Andrej's llama2.c and closed the
loop on llama models.  The llama2.c integration with AOT Inductor comes from Bert's repo.

* my colleagues and friends Christian Puhrsch, Horace He, Joe Isaacson, and many more for their many contributions in Accelerating GenAI models in
the *"Anything, Fast!"* blog series, and in particular Horace He for [GPT, Fast!](https://github.com/pytorch-labs/gpt-fast) that we have
directly adopted (both ideas and code) from his repo.

* my colleagues and friends Bert Maher, Scott Wolchok, Bin Bao, Chen Yang, Huamin Li and Mu-Chu Li for a great collaboration
in building AOT Inductor for CPU, internal use cases and an experimental AOTI-compiled inference version of [nanoGPT](https://github.com/karpathy/nanoGPT).

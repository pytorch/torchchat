# Torchchat is still in pre-release!


Torchchat is currently in a pre-release state and under extensive development.


# The Lost Manual: torchchat

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) | [**Download**](#download) | [**Chat**](#chat) | [**Generate**](#generate) | [**Eval**](#eval) | [**Export**](#export) | [**Supported Systems**](#supported-systems) | [**Contributing**](#contributing) | [**License**](#license)

&nbsp;

This is the advanced users guide, if you're looking to get started
with LLMs, please refer to the README at the root directory of the
torchchat distro.  This is an advanced user guide, so we will have
many more concepts and options to discuss and taking advantage of them
may take some effort.

We welcome community contributions of all kinds.  If you find
instructions that are wrong, please submit a PR to the documentation,
or to the code itself.

## Introduction

Torchchat (pronounced “torch chat” and also a play on torch @ [laptop,
desktop, mobile]) is a tool and library to easily run LLMs on laptops,
desktops, and mobile devices using pure
[PyTorch](https://github.com/pytorch/pytorch) and
[ExecuTorch](https://github.com/pytorch/executorch). See below for a
[full list of supported devices](#supported-systems).

While we strive to support a broad range of models, we can't test them
all. We classify supported models as tested ✅, work in progress 🚧 or
some restrictions ❹.

We invite community contributions of new model suport and test results!

| Model | Tested | Eager | torch.compile | AOT Inductor | ExecuTorch | Fits on Mobile |
|-----|--------|-------|-----|-----|-----|-----|
meta-llama/Llama-3-7b | 🚧  | ✅ |  ✅ |  ✅ |  ✅ | ❹ |
meta-llama/Llama-2-7b-chat-hf | 🚧  | ✅ |  ✅ |  ✅ |  ✅ | ❹|
meta-llama/Llama-2-13b-chat-hf | - | ✅ |  ✅ |  ✅ |  ✅ | 📵 |
meta-llama/Llama-2-70b-chat-hf | - | ✅ |  ✅ |  ✅ |  ✅ | ❌|
tinyllamas/stories15M | ✅ | ✅ |  ✅ |  ✅ |  ✅ | ✅ |
tinyllamas/stories42M  | - | ✅ |  ✅ |  ✅ |  ✅ | ✅ |
tinyllamas/stories110M   | ✅ | ✅ |  ✅ |  ✅ |  ✅ | ✅ |
openlm-research/open_llama_7b  | 🚧 | ✅ |  ✅ |  ✅ |  ✅ | ❹ |
codellama/CodeLlama-7b-Python-hf | -| ✅ |  ✅ |  ✅ |  ✅ | ❹|
codellama/CodeLlama-34b-Python-hf | -| ✅ |  ✅ |  ✅ |  ✅ | ❌ |
mistralai/Mistral-7B-v0.1 | 🚧  |  ✅  |  ✅ |  ✅ |  ✅ | ❹ |
mistralai/Mistral-7B-Instruct-v0.1 | - | ✅ |  ✅ |  ✅ |  ✅ | ❹ |
mistralai/Mistral-7B-Instruct-v0.2 | - | ✅ |  ✅ |  ✅ |  ✅ | ❹ |

*Key:* ✅ works correctly; 🚧  work in progress; ❌ not supported; ❹ requires 4bit groupwise quantization; 📵 not on mobile (may fit some high-end devices such as tablets);

&nbsp;

---

## Get Started

Torchchat lets you access LLMs through an interactive interface, prompted single-use generation, model export (for use by AOT Inductor and ExecuTorch), and standalone C++ runtimes.

| Function | Torchchat Command | Direct Command | Tested |
|---|----|----|-----|
Download model | `torchchat --download` | n/a | 🚧 |
Interactive chat | `torchchat --chat`   | n/a | 🚧 |
GUI-based chat | `torchchat --gui`   | n/a | ⚠️ |
Generate text | `torchchat --generate` |`generate` | ✅ |
Evaluate model | `torchchat --eval` | `eval` | 🚧 |
Export model  | `torchchat --export` | `export` | ✅ |
Exported model test (dso,pte) | `torchchat --chat` | n/a  | 🚧 |
exported model test (dso,pte) | `torchchat --generate` |`generate` | ✅ |
Evaluate exported model (dso,pte) | `torchchat --eval` | `eval` | 🚧 |
Server C++ runtime | n/a | run.cpp model.so | ✅ |
Server C++ runtime | n/a | run.cpp model.pte | ✅ |
Mobile C++ runtime | n/a | app model.pte | ✅ |
Mobile C++ runtime | n/a | app + AOTI | 🚧 |

**Getting help:** Each command implements the --help option to give addititonal information about available options:

```
python3 torchchat.py [ export | generate | chat | eval | ... ] --help
```

Exported models can be loaded back into torchchat for chat or text
generation, letting you experiment with the exported model and valid
model quality. The python interface is the same in all cases and is
used for testing nad test harnesses too.

Torchchat comes with server C++ runtimes to execute AOT Inductor and
ExecuTorch models. A mobile C++ runtimes allow you to deploy
ExecuTorch-compiled .pte files on iOS, Android and Raspberry Pi 5.

## Downloading and Configuring Models

You can download any LLM model that fits the model.py model
architecture, provided you have the model weights in llama-format, the
model parameters and the tokenizer model used by your language model.
For models not specified not in the list of known configurations, you
can construct the model by initializing the `ModelArgs` dataclass that
controls model construction from a parameter json using the
`params-path ${PARAMS_PATH}` containing the appropriate model
parameters to initialize the ModelArgs for the model. (We use the
model constructor `Transformer.from_params()`).

The parameter file will should be in JSON format specifying thee
parameters.  You can find the Model Args data class in
[`model.py`](https://github.com/pytorch/torchchat/blob/main/model.py#L22).

The final way to initialize a torchchat model is from GGUF. You load a
GGUF model with the option `--load-gguf ${MODELNAME}.gguf`. Presently,
the F16, F32, Q4_0, and Q6_K formats are supported and converted into
native torchchat models.

You may also dequantize GGUF models with the GGUF quantize tool, and
then load and requantize with torchchat native quantization options.
(Please note that quantizing and dequantizing is a lossy process, and
you will get the best results by starting with the original
unquantized model checkpoint, not a previsouly quantized and thend
equantized model.)

| GGUF Model | Tested | Eager | torch.compile | AOT Inductor | ExecuTorch | Fits on Mobile |
|-----|--------|-------|-----|-----|-----|-----|
| llama-2-7b.Q4_0.gguf |  🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

You may also dequantize GGUF models with the GGUF quantize tool, and
then load and requantize with torchchat native quantization options.

**Please note that quantizing and dequantizing is a lossy process, and
you will get the best results by starting with the original
unquantized model checkpoint, not a previsoul;y quantized and thend
equantized model.**


## Chat

We use several variables in this example, which may be set as a
preparatory step:

* `MODEL_NAME` describes the name of the model.  This name is *not*
   free-form, as it is used to index into a table of supported models
   and their configuration properties that are needed to load the
   model. This variable should correspond to the name of the directory
   holding the files for the corresponding model.  You *must* follow
   this convention to ensure correct operation.

* `MODEL_DIR` is the location where we store model and tokenizer
  information for a particular model. We recommend
  `checkpoints/${MODEL_NAME}` or any other directory you already use
  to store model information.

* `MODEL_PATH` describes the location of the model. Throughput the
  description herein, we will assume that MODEL_PATH starts with a
  subdirectory of the torchchat repo named checkpoints, and that it
  will contain the actual model. In this case, the MODEL_PATH will
  thus be of the form ${MODEL_OUT}/model.{pt,pth}.  (Both the
  extensions `pt` and `pth` are used to describe checkpoints. In
  addition, model may be replaced with the name of the model.)

  The generate.py sequence generator will load the tokenizer from the
  directory specified by the MODEL_PATH variable, by replacing the
  modelname with the name of the tokenizer model which is expected to
  be named `tokenizer.model`.

* `MODEL_OUT` is a location for outputs from export for server/desktop
  and/or mobile/edge execution.  We store exported artifacts here,
  with extensions .pte for Executorch models, .so for AOT Inductor
  generated models, and .bin for tokenizers prepared for use with the
  C++ tokenizers user by `runner-aoti` and `runner-et`.

You can set these variables as follows for the exemplary model15M
model from Andrej Karpathy's tinyllamas model family:

```
MODEL_NAME=stories15M
MODEL_DIR=<root to your heckpoints>/${MODEL_NAME}
MODEL_PATH=${MODEL_OUT}/stories15M.pt
MODEL_OUT=~/torchchat-exports
```

When we export models with AOT Inductor for servers and desktops, and
Executorch for mobile and edge devices, we will save them in the
specified directory (`${MODEL_OUT}` in our example below) as a DSO
under the name `${MODEL_NAME}.so` (for AOTI-generated dynamic
libraries), or as Executorch model under the name `${MODEL_NAME}.pte`
(for Executorch-generated mobile/edge models).

We use `[ optional input ]` to indicate optional inputs, and `[ choice
1 | choice 2 | ... ]` to indicate a choice


### A note on tokenizers

There are two different formats for tokenizers, and both are used in this repo.

1 - for generate.py and Python bindings, we use the Google
  sentencepiece Python operator and the TikToken tokenizer (for
  llama3). This operator consumes a tokenization model in the
  `tokenizer.model` format.

2 - for C/C++ inference, we use @Andrej Karpathy's C tokenizer
  function, as well as a C++ TikToken tokenizer (for llama3).  This
  tokenizer consumes a tokenization model in the 'tokenizer.bin'
  format.

You can convert a SentencePiece tokenizer.model into tokenizer.bin
using Andrej's tokenizer.py utility to convert the tokenizer.model to
tokenizer.bin format:

```
python3 utils/tokenizer.py --tokenizer-model=${MODEL_DIR}tokenizer.model
```

We will later disucss how to use this model, as described under *STANDALONE EXECUTION* in a Python-free
environment:
```
runner-{et,aoti}/build/run ${MODEL_OUT}/model.{so,pte} -z ${MODEL_OUT}/tokenizer.bin
```

### Llama 3 tokenizer

Add option to load tiktoken tokenizer
```
--tiktoken
```

## Generate

Model definition in model.py, generation code in generate.py. The
model checkpoint may have extensions `pth` (checkpoint and model
definition) or `pt` (model checkpoint).  At present, we always use the
torchchat model for export and import the checkpoint into this model
definition because we have tested that model with the export
descriptions described herein.

```
python3 generate.py --compile --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --device [ cuda | cpu | mps]
```

To squeeze out a little bit more performance, you can also compile the
prefill with --compile_prefill. This will increase compilation times
though. The --compile-prefill option requires --parallel-prefill,
which are not available for exported DSO and PTE models.


## Eval

To be added. For basic eval instructions, please see the introductury
README.

In addition to running eval on models in eager mode (optionally
compiled with `torch.compile()`, you can also load dso and pte models
back into the generate.py tool.  This will allow you to run any tests
and evaluations that you want to run on the exported models without
requiring changes to your test harnesses and evaluation scripts,


## Export

Let's start by exporting and running a small model like stories15M.

```
python3 export.py --checkpoint-path ${MODEL_PATH} -d fp32 --output-pte-path ${MODEL_OUT}/model.pte
```

### AOT Inductor compilation and execution
```
python3 export.py --checkpoint-path ${MODEL_PATH} --device {cuda,cpu} --output-dso-path ${MODEL_OUT}/${MODEL_NAME}.so
```

When you have exported the model, you can test the model with the
sequence generator by importing the compiled DSO model with the
`--dso-path ${MODEL_OUT}/${MODEL_NAME}.so` option.  This gives
developers the ability to test their model, run any pre-existing model
tests against the exported model with the same interface, and support
additional experiments to confirm model quality and speed.

```
python3 generate.py --device {cuda,cpu} --dso-path ${MODEL_OUT}/${MODEL_NAME}.so --prompt "Hello my name is"
```

While we have shown the export and execution of a small model on CPU
or an accelerator such as GPU, most models need to be compressed to
reduce their memory bandwidth requirements and avoid stalling the
execution engines while they are waiting for data.  We use
quantization to achieve this, as described below.


### ExecuTorch mobile compilation

We export the model with the export.py script.  Running this script
requires you first install executorch with pybindings, see
[here](#setting-up-executorch-and-runner-et).  At present, when
exporting a model, the export command always uses the xnnpack delegate
to export.  (Future versions of torchchat will support additional
delegates such as Vulkan, CoreML, MPS, HTP in addition to Xnnpack as
they are released for Executorch.)

### Running the model

With the model exported, you can now generate text with the executorch
runtime pybindings.  Feel free to play around with the prompt.

```
python3 generate.py --checkpoint-path ${MODEL_PATH} --pte ${MODEL_OUT}/model.pte --device cpu --prompt "Once upon a time"
```

You can also run the model with the runner-et.  See below under
"Standalone Execution".

While we have shown the export and execution of a small model to a
mobile/edge device supported by Executorch, most models need to be
compressed to fit in the target device's memory. We use quantization
to achieve this.



## Optimizing your model for server, desktop and mobile devices

To compress models, torchchat offers a variety of strategies:

* Configurable floating-point precision, depending on backend
  capabilities (for activations and weights): float32, float16,
  bfloat16

* weight-quantization: embedding quantization and linear operator
  quantization

* dynamic activation quantization with weight quantization: a8w4dq

In addition, we support GPTQ and HQQ for improving the quality of 4b
weight-only quantization.  Support for HQQ is a work in progress.

| compression | FP precision |  weight quantization | dynamic activation quantization |
|--|--|--|--|
embedding table (symmetric) | fp32, fp16, bf16 | 8b (group/channel), 4b (group/channel) | n/a |
linear operator (symmetric) | fp32, fp16, bf16 | 8b (group/channel) | n/a |
linear operator (asymmetric) | n/a | 4b (group), a6w4dq | a8w4dq (group) |
linear operator (asymmetric) with GPTQ | n/a | 4b (group) | n/a |
linear operator (asymmetric) with HQQ | n/a |  work in progress | n/a |


## Model precision (dtype precision setting)

You can generate models (for both export and generate, with eager, torch.compile, AOTI, ET, for all backends - mobile at present will primarily support fp32, with all options)
specify the precision of the model with 

```
python3 generate.py --dtype [bf16 | fp16 | fp32] ...
python3 export.py --dtype [bf16 | fp16 | fp32] ...
```

**Unlike gpt-fast which uses bfloat16 as default, Torchchat uses
  float32 as the default. As a consequence you will have to set to
  `--dtype bf16` or `--dtype fp16` on server / desktop for best
  performance.**

You can find instructions for quantizing models in
[docs/quantization.md](file:///./quantization.md).  Advantageously,
quantization is available in eager mode as well as during export,
enabling you to do an early exploration of your quantization setttings
in eager mode.  However, final accuracy should always be confirmed on
the actual execution target, since all targets have different build
processes, compilers, amd kernel implementations with potentially
significant impact on accuracy.


## Loading GGUF models

GGUF is a nascent industry standard format and presently torchchat can
read the F16, F32, Q4_0, and Q6_K formats natively and convert them
into native torchchat models by using the load-gguf option:

```
python3 [ export.py | generate.py | ... ] --gguf-path <gguf_filename>
```

Ypu may then apply the standard quantization options, e.g., to add
embedding table quantization as described under quantization. (You
cannot directly requantize already quantized formats.  However, you
may dequantize them using GGUF tools, and then laod the model into
torchchat to quantize wqith torchchat's quantization workflow.)


## Loading unsupported GGUF formats in torchchat

GGUF formats not presently supported natively in torchchat may be
converted to one of the supported formats with GGUF's
`${GGUF}/quantize` utility to be loaded in torchchat. If you convert
to the FP16 or FP32 formats with GGUF's `quantize` utility, you may
then requantize these models with torchchat's quantization workflow.

**Note that quantizing and dequantizing is a lossy process, and you will
get the best results by starting with the original unquantized model
checkpoint, not a previously quantized and then dequantized
model.** Thus, while you can convert your q4_1 model to FP16 or FP32
GGUF formats and then requantize, you might get better results if you
start with the original FP16 or FP32 GGUF format.

To use the quantize tool, install the GGML tools at ${GGUF} . Then,
you can, for example, convert a quantized model to f16 format:


```
${GGUF}/quantize --allow-requantize your_quantized_model.gguf fake_unquantized_model.gguf f16
```

# Standalone Execution

In addition to running the exported and compiled models for server,
desktop/laptop and mobile/edge devices by loading them in a PyTorch
environment under the Python interpreter, these models can also be
executed directly

## Desktop and Server Execution

This has been tested with Linux and x86 (using CPU ~and GPU~), and
MacOS and ARM/Apple Silicon.

The runner-* directories show how to integrate AOTI- and ET-exported
models in a C/C++ application when no Python environment is available.
Integrate it with your own applications and adapt it to your own
application and model needs!  Each runner directory comes with a cmake
build script.  Please refer to this file for detailed build
instructions, and adapt as appropriate for your system.

Build the runner like this
```
cd ./runner-aoti
cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build build
```

To run, use the following command (assuming you already generated the
tokenizer.bin tokenizer model):

```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./build/run ../${MODEL_NAME}.so -z ../${MODEL_NAME}.bin
```

## Mobile and Edge Execution Test (x86)

You can also run the model with the runner-et.  This requires you
first build the runner.  See instructions
[here](#setting-up-executorch-and-runner-et).  After this is done, you
can run runner-et with

```
./build/cmake-out/runner_et ${MODEL_OUT}/model.pte -z ${MODEL_OUT}/tokenizer.bin -i "Once upon a time in a land far away"
```

While we have shown the export and execution of a small model to a
mobile/edge device supported by Executorch, most models need to be
compressed to fit in the target device's memory. We use quantization
to achieve this.


This has been shown to run on x86. with the proper IDE environment,
you can compile for your specific target.  For a GUI integration in
iOS and Android, please refer to "Running on a mobile/edge system" in
the section below.

Build the runner like this
```
cd ./runner-et
cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build build
```

To run your pte model, use the following command (assuming you already
generated the tokenizer.bin tokenizer model):

```
./build/run ${MODEL_OUT}/${MODEL_NAME}{,_int8,_8da4w}.pte -z ${MODEL_OUT}/${MODEL_NAME}.bin
```

## Running on a mobile/edge system

### Android

Check out the [tutorial on how to build an Android app running your
PyTorch models with
Executorch](https://pytorch.org/executorch/main/llm/llama-demo-android.html),
and give your torchchat models a spin.

![Screenshot](https://pytorch.org/executorch/main/_static/img/android_llama_app.png
 "Android app running Llama model")

Detailed step by step in conjunction with ET Android build, to run on
simulator for Android. `scripts/android_example.sh` for running a
model on an Android simulator (on Mac), and in `docs/Android.md`.



### iOS

Open the iOS Llama Xcode project at
https://github.com/pytorch/executorch/tree/main/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj
in Xcode and click Run.  You will need to provide a provisioning
profile (similar to what's expected for any iOS dev).

Once you can run the app on you device,

1 - connect the device to you Mac,

2 - copy the model and tokenizer.bin to the iOS Llama app

3 - select the tokenizer and model with the `(...)` control (bottom
  left of screen, to the left of the text entrybox)

Refer to `docs/iOS.md` for more information.


# Supported Systems

PyTorch and ExecuTorch support a broad range of devices for running
PyTorch with python (using either eager or eager + `torch.compile`) or
in a python-free environment with AOT Inductor and ExecuTorch.


| Hardware | OS | Eager | Eager + Compile | AOT Compile | ET Runtime |
|-----|------|-----|-----|-----|-----|
| x86 | Linux | ✅ |  ✅ |  ✅ |  ✅ |
| aarch64 | Linux | n/t | n/t | n/t | n/t |
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
| Raspberry Pi 4/5 | Raspbian | n/t | n/t | n/t | ✅ |
| Raspberry Pi 4/5 | Android | ❌ | ❌ | ❌ | n/t |
| ARM 32b (up to v7) | any | ❌|❌|❌|❌|

*Key*: n/t -- not tested


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


## Runtime performance with Llama3, in tokens per second (4b quantization)

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



# Setting up ExecuTorch and runner-et

Set up ExecuTorch by following the instructions [here](https://pytorch.org/executorch/stable/getting-started-setup.html#setting-up-executorch).
For convenience, we provide a script that does this for you.

From the torchchat root directory, run the following
```
export TORCHCHAT_ROOT=${PWD}
./scripts/install_et.sh
```

This will create a build directory, git clone ExecuTorch to ./build/src, applies some patches to the ExecuTorch source code, install the ExecuTorch python libraries with pip, and install the required ExecuTorch C++ libraries to ./build/install.  This will take a while to complete.

After ExecuTorch is installed, you can build runner-et from the torchchat root directory with the following

```
export TORCHCHAT_ROOT=${PWD}
cmake -S ./runner-et -B build/cmake-out -G Ninja
cmake --build ./build/cmake-out
```

The built executable is located at ./build/cmake-out/runner-et.


# Contributing to torchchat

We welcome any feature requests, bug reports, or pull requests from the community. See the [CONTRIBUTING](CONTRIBUTING.md) for instructions how to contribute to torchchat.

&nbsp;

## License

Torchchat is released under the [BSD 3 license](./LICENSE). However
you may have other legal obligations that govern your use of other
content, such as the terms of service for third-party models.

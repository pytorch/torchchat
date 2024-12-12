> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.


# The Lost Manual: torchchat

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) | [**Download**](#download) | [**Chat**](#chat) | [**Generate**](#generate) | [**Eval**](#eval) | [**Export**](#export) | [**Supported Systems**](#supported-systems) | [**Contributing**](#contributing) | [**License**](#license)

<!--

[shell default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}" huggingface-cli login

[shell default]: ./install/install_requirements.sh

[shell default]: TORCHCHAT_ROOT=${PWD} ./torchchat/utils/scripts/install_et.sh

-->

This is the advanced users' guide, if you're looking to get started
with LLMs, please refer to the README at the root directory of the
torchchat distro.  This is an advanced user guide, so we will have
many more concepts and options to discuss and take advantage of them
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

We invite community contributions of new model support and test results!

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

*Key:* ✅ works correctly; 🚧 work in progress; ❌ not supported; ❹
 requires 4bit groupwise quantization; 📵 not on mobile (may fit some
 high-end devices such as tablets);


## Get Started

Torchchat lets you access LLMs through an interactive interface,
prompted single-use generation, model export (for use by AOT Inductor
and ExecuTorch), and standalone C++ runtimes.

| Function | Torchchat Command | Direct Command | Tested |
|---|----|----|-----|
Download model | `torchchat.py download` | n/a | 🚧 |
Interactive chat | `torchchat.py chat`   | n/a | 🚧 |
GUI-based chat | `torchchat.py browser`   | n/a | ⚠️ |
Generate text | `torchchat.py generate` |`generate.py` | ✅ |
Evaluate model | `torchchat.py eval` | `eval.py` | 🚧 |
Export model  | `torchchat.py export` | `export.py` | ✅ |
Exported model test (dso,pte) | `torchchat --chat` | n/a  | 🚧 |
Exported model test (dso,pte) | `torchchat --generate` |`generate.py` | ✅ |
Evaluate exported model (dso,pte) | `torchchat --eval` | `eval.py` | 🚧 |
Server C++ runtime | n/a | run.cpp model.so | ✅ |
Server C++ runtime | n/a | run.cpp model.pte | ✅ |
Mobile C++ runtime | n/a | app model.pte | ✅ |
Mobile C++ runtime | n/a | app + AOTI | 🚧 |

**Getting help:** Each command implements the --help option to give additional information about available options:

[skip default]: begin
```
python3 torchchat.py [ export | generate | chat | eval | ... ] --help
```
[skip default]: end

Exported models can be loaded back into torchchat for chat or text
generation, letting you experiment with the exported model and valid
model quality. The Python interface is the same in all cases and is
used for testing and test harnesses, too.

Torchchat comes with server C++ runtimes to execute AOT Inductor and
ExecuTorch models. A mobile C++ runtimes allow you to deploy
ExecuTorch-compiled .pte files on iOS, Android and Raspberry Pi 5 (as
well as on desktops and servers with a native runtime such as
runner/run.cpp).

## Downloading and Configuring Models

You can download any LLM model that fits the model.py model
architecture, provided you have the model weights in llama format, the
model parameters and the tokenizer model used by your language model.

Some common models are recognized by torchchat based on their filename
through `Model.from_name()` to perform a fuzzy match against a
table of known model architectures. Alternatively, you can specify the
index into that table with the option `--params-table ${INDEX}` where
the index is the lookup key in the [the list of known
pconfigurations](https://github.com/pytorch/torchchat/tree/main/torchchat/model_params)
For example, for the stories15M model, this would be expressed as
`--params-table stories15M`. (We use the model constructor
`Model.from_table()`)

For models using a configuration not in the list of known
configurations, you can construct the model by initializing the
`ModelArgs` dataclass that controls model construction from a
parameter json using the `params-path ${PARAMS_PATH}` containing the
appropriate model parameters to initialize the `ModelArgs` for the
model. (We use the model constructor `Model.from_params()`).

The parameter file should be in JSON format specifying these
parameters. You can find the `ModelArgs` data class in
[`model.py`](https://github.com/pytorch/torchchat/blob/main/build/model.py#L70).

The final way to initialize a torchchat model is from GGUF. You load a
GGUF model with the option `--load-gguf ${MODELNAME}.gguf`. Presently,
the F16, F32, Q4_0, and Q6_K formats are supported and converted into
native torchchat models.

| GGUF Model | Tested | Eager | torch.compile | AOT Inductor | ExecuTorch | Fits on Mobile |
|-----|--------|-------|-----|-----|-----|-----|
| llama-2-7b.Q4_0.gguf |  🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

## Conventions used in this document

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

* `MODEL_PATH` describes the location of the model. Throughout the
  description herein, we will assume that `MODEL_PATH` starts with a
  subdirectory of the torchchat repo named checkpoints, and that it
  will contain the actual model. In this case, the `MODEL_PATH` will
  thus be of the form `${MODEL_OUT}/model.{pt,pth}`.  (Both the
  extensions `pt` and `pth` are used to describe checkpoints. In
  addition, model may be replaced with the name of the model.)

  The `generate.py` sequence generator will load the tokenizer from the
  directory specified by the `MODEL_PATH` variable, by replacing the
  model name with the name of the tokenizer model which is expected to
  be named `tokenizer.model`.

* `MODEL_OUT` is a location for outputs from export for server/desktop
  and/or mobile/edge execution.  We store exported artifacts here,
  with extensions `.pte` for Executorch models, `.so` for AOT Inductor
  generated models.

You can set these variables as follows for the exemplary model15M
model from Andrej Karpathy's tinyllamas model family:

```
MODEL_NAME=stories15M
MODEL_DIR=~/checkpoints/${MODEL_NAME}
MODEL_PATH=${MODEL_DIR}/stories15M.pt
MODEL_OUT=~/torchchat-exports

mkdir -p ${MODEL_DIR}
mkdir -p ${MODEL_OUT}
```

When we export models with AOT Inductor for servers and desktops, and
ExecuTorch for mobile and edge devices, we will save them in the
specified directory (`${MODEL_OUT}` in our example below) as a shared
library (also known as DSO for DYnamically Shared Object) which may
later be loaded by the AOTI (AOT Inductor (AOTI) runtime under the
name `${MODEL_NAME}.so` for AOTI-generated dynamic libraries, and as
ExecuTorch model under the name `${MODEL_NAME}.pte` (for
Executorch-generated mobile/edge models).

We use `[ optional input ]` to indicate optional inputs, and `[ choice
1 | choice 2 | ... ]` to indicate a choice


## Torchchat Overview

The torchchat Model definition may be found in `build/model.py`, the
code to build the model in `torchchat/cli/builder.py` and sequence generation
code for prompted sequence generation and chat in `generate.py`. The
model checkpoint will commonly have extensions `pth` (checkpoint and model
definition) or `pt` (model checkpoint).  At present, we always use the
torchchat model for export and import the checkpoint into this model
definition because we have tested that model with the export
descriptions described herein.

That being said, the export and execution logic of the model may be
adapted to support other models, either by extending the model
description in `model.py` or by initializing a completely different
model.  *We invite and welcome community contributions of open-source
model enablement to torchchat, as well as to our related open source
projects PyTorch, ExecuTorch (for mobile/edge models), torchao
(for architecture optimization) and other
PyTorch projects.* (Please refer to individual projects for specific
submission guidelines.)

Torchchat supports several devices.  You may also let torchchat use
heuristics to select the best device from available devices using
torchchat's virtual device named `fast`.

Torchchat supports execution using several floating-point datatypes.
Please note that the selection of execution floating point type may
affect both model quality and performance.  At present, the supported
FP data types are torch.float32, torch.float16 and torch.bfloat16.  In
addition, torchchat recognizes two virtual data types, fast which
selects the best floating point type on the present system and fast16
which chooses the best 16-bit floating point type.

The virtual device fast and virtual floating point data types fast and
fast16 are best used for eager/torch.compiled execution.  For export,
specify your device choice for the target system with --device for
AOTI-exported DSO models, and using ExecuTorch delegate selection for
ExecuTorch-exported PTE models.


## PyTorch eager mode and JIT-compiled execution
```
python3 torchchat.py generate [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --device [ cuda | mps | cpu ]
```

To improve performance, you can compile the model with `--compile`
trading off the time to first token processed with time per token.  To
improve performance further, you may also compile the prefill with
`--compile-prefill`. This will increase further compilation times though. 
For CPU, you can use `--max-autotune` to further improve the performance
with `--compile` and `compile-prefill`. See [`max-autotune on CPU tutorial`](https://pytorch.org/tutorials/prototype/max_autotune_on_CPU_tutorial.html).

Parallel prefill is not yet supported by exported models, and may be
supported in a future release.


## Model quality evaluation

For an introduction to the model evaluation tool `eval`, please see
the introductory README.

In addition to running eval on models in eager mode and JIT-compiled
mode with `torch.compile()`, you can also load dso and pte models back
into the PyTorch to evaluate the accuracy of exported model objects
(e.g., after applying quantization or other transformations to
improve speed or reduce model size).

Loading exported models back into a Python-based Pytorch allows you to
run any tests and evaluations that you want to run on the exported
models without requiring changes to your test harnesses and evaluation
scripts.

Learn more about model evaluation in [torchchat/utils/evaluation.md].


## Model Export for Native Execution

Export generates binary model objects that may be executed in a
variety of Python-free native execution environments.

Let's start by exporting and running a small model like stories15M
with ExecuTorch to generate a portable compact model representation,
and AOT Inductor for native optimized performance on CPUs and GPUs.
We export the model with the `export.py` or `torchchat.py export`
command.

Export for mobile backends requires that you first install executorch
with pybindings, as described in [README.md]. At present, when
exporting a model for deployment with the ExecuTorch runtome, the
export command always uses the XNNPACK delegate to export. (Future
versions of torchchat will support additional delegates such as
Vulkan, CoreML, MPS, HTP in addition to XNNPACK as they are released
for ExecuTorch.)

We export the stories15M model with the following command for
execution with the ExecuTorch runtime (and enabling execution on a
wide range of community and vendor-supported backends):

```
python3 torchchat.py export --checkpoint-path ${MODEL_PATH} --output-pte-path ${MODEL_NAME}.pte
```

Alternatively, we may generate a native instruction stream binary
using AOT Inductor for CPU or GPUs (the latter using Triton for
optimizations such as operator fusion):

```
python3 torchchat.py export --checkpoint-path ${MODEL_PATH} --device [ cuda | cpu ] --output-dso-path ${MODEL_NAME}.so
```


## Test and Evaluation of Exported Models

As mentioned earlier, after you have exported the model, you can load
the exported model artifact back into a model container with a
compatible API surface for the `model.forward()` function.  This
enables users to test, evaluate and exercise the exported model
artifact with familiar interfaces, and in conjunction with
pre-existing Python model unit tests and common environments such as
Jupyter notebooks and/or Google colab.

Here is how to load an exported model into the Python environment using an exported model with the `generate` command.

```
python3 torchchat.py generate --checkpoint-path ${MODEL_PATH} --pte-path ${MODEL_NAME}.pte --device cpu --prompt "Once upon a time"
```

After you have exported the model, you can test the model with the
sequence generator by importing the compiled DSO model with the
`--dso-path ${MODEL_NAME}.so` option.  This gives
developers the ability to test their model, run any pre-existing model
tests against the exported model with the same interface, and support
additional experiments to confirm model quality and speed.

```
python3 torchchat.py generate --device [ cuda | cpu ] --dso-path ${MODEL_NAME}.so --prompt "Once upon a time"
```


For native Python-free execution, see below under "Standalone
Execution", for an example application and its model integration.

While we have shown the export and execution of a small model to a
mobile/edge device supported by ExecuTorch, most models need to be
compressed to fit in the target device's memory. We use quantization
to achieve this.

### Visualizing the backend delegate on ExecuTorch export

By default, export will lower to the XNNPACK delegate for improved
performance. ExecuTorch export provides APIs to visualize what happens
after the `to_backend()` call in the lowering process.

- `get_delegation_info()`: provide a summary of the model after the
  `to_backend()` call, including the total delegated subgraphs, number
  of delegated nodes and number of non-delegated nodes.

- `format_delegated_graph`: a formatted str of the whole graph, as
  well as the subgraph/s consumed by the backend.

See the
[debug backend delegate documentation](https://pytorch.org/executorch/main/debug-backend-delegate.html)
for more details.


## Optimizing your model for server, desktop and mobile devices

To compress models, torchchat offers a variety of strategies:

* Configurable floating-point precision, depending on backend
  capabilities (for activations and weights): float32, float16,
  bfloat16

* weight-quantization: embedding quantization and linear operator
  quantization

* dynamic activation quantization with weight quantization: a8w4dq

| compression | FP precision |  weight quantization | dynamic activation quantization |
|--|--|--|--|
embedding table (symmetric) | fp32, fp16, bf16 | 8b (group/channel), 4b (group/channel) | n/a |
linear operator (symmetric) | fp32, fp16, bf16 | 8b (group/channel) | n/a |
linear operator (asymmetric) | n/a | 4b (group), a6w4dq | a8w4dq (group) |


## Model precision (dtype precision setting)
On top of quantizing models with quantization schemes mentioned above, models can be converted
to lower precision floating point representations to reduce the memory bandwidth requirement and
take advantage of higher density compute available. For example, many GPUs and some of the CPUs
have good support for bfloat16 and float16. This can be taken advantage of via `--dtype arg` as shown below.

[skip default]: begin
```
python3 torchchat.py generate --dtype [bf16 | fp16 | fp32] ...
python3 torchchat.py export --dtype [bf16 | fp16 | fp32] ...
```
[skip default]: end

You can find instructions for quantizing models in
[docs/quantization.md](file:///./quantization.md).  Advantageously,
quantization is available in eager mode as well as during export,
enabling you to do an early exploration of your quantization setttings
in eager mode.  However, final accuracy should always be confirmed on
the actual execution target, since all targets have different build
processes, compilers, and kernel implementations with potentially
significant impact on accuracy.


## Loading GGUF models

GGUF is a nascent industry standard format and presently torchchat can
read the F16, F32, Q4_0, and Q6_K formats natively and convert them
into native torchchat models by using the load-gguf option:

[skip default]: begin
```
python3 torchchat.py [ export | generate | ... ] --gguf-path <gguf_filename>
```
[skip default]: end

You may then apply the standard quantization options, e.g., to add
embedding table quantization as described under quantization. (You
cannot directly requantize already quantized formats. However, you
may dequantize them using GGUF tools, and then laod the model into
torchchat to quantize with torchchat's quantization workflow.)


## Optimizing your model for server, desktop and mobile devices

While we have shown the export and execution of a small model on CPU
or an accelerator such as GPU, most models need to be compressed to
reduce their memory bandwidth requirements and avoid stalling the
execution engines while they are waiting for data.  We use
quantization to achieve this, as described below.

To compress models to minimize memory requirements for both bandwidth
and storage, as well as speed, torchchat offers a variety of
strategies:

* Configurable floating-point precision, depending on backend
  capabilities (for activations and weights): float32, float16,
  bfloat16

* weight-quantization: embedding quantization and linear operator
  quantization

* dynamic activation quantization with weight quantization: a8w4dq

You can find instructions for quantizing models in
[docs/quantization.md](file:///./quantization.md).  Advantageously,
quantization is available in eager mode as well as during export,
enabling you to do an early exploration of your quantization settings
in eager mode.  However, final accuracy should always be confirmed on
the actual execution target, since all targets have different build
processes, compilers, and kernel implementations with potentially
significant impact on accuracy.





## Native (Stand-Alone) Execution of Exported Models

Refer to the [README](README.md) for an introduction to native
execution on servers, desktops, and laptops.  Mobile and Edge execution for Android and iOS are
described under [torchchat/edge/docs/Android.md] and [torchchat/edge/docs/iOS.md], respectively.



# Supported Systems

PyTorch and ExecuTorch support a broad range of devices for running
PyTorch with python (using either eager or eager + `torch.compile`) or
in a Python-free environment with AOT Inductor and ExecuTorch.


| Hardware | OS | Eager | Eager + Compile | AOT Compile | ET Runtime |
|-----|------|-----|-----|-----|-----|
| x86 | Linux | ✅ |  ✅ |  ✅ |  ✅ |
| aarch64 | Linux | ✅ | ✅ | ✅ | n/t |
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
| Raspberry Pi 4/5 | Raspbian | ✅ | ✅ | ✅ | ✅ |
| Raspberry Pi 4/5 | Android | ❌ | ❌ | ❌ | n/t |
| ARM 32b (up to v7) | any | ❌|❌|❌|❌|

*Key*: n/t -- not tested


# LICENSE

Torchchat is released under the [BSD 3 license](./LICENSE). However
you may have additional legal obligations that govern your use of other
content, such as the terms of service for third-party models, the
Llama2 and Llama3 community licenses..

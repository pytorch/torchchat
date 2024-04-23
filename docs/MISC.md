# PLEASE DO NOT UPDATE THIS FILE RIGHT NOW
## All updates should be made [here](https://docs.google.com/document/d/1y0D09JtKl81k6Vf1iCEafzj45B_BmnB-KQQNR7p9DDQ/edit) for now
## Refer to [this](https://fb.workplace.com/groups/pytorch.edge.team/permalink/1486105605277507/) for more details

# Torchchat is still in pre-release!


Torchchat is currently in a pre-release state and under extensive development.


# Torchchat

[**Introduction**](#introduction) | [**Installation**](#installation) | [**Get Started**](#get-started) | [**Download**](#download) | [**Chat**](#chat) | [**Generate**](#generate) | [**Eval**](#eval) | [**Export**](#export) | [**Supported Systems**](#supported-systems) | [**Contributing**](#contributing) | [**License**](#license)

&nbsp;

## Introduction

Torchchat (pronounced “torch chat” and also a play on torch @ [laptop, desktop, mobile]) is a tool and library to easily run LLMs on laptops, desktops, and mobile devices using pure [PyTorch](https://github.com/pytorch/pytorch) and [ExecuTorch](https://github.com/pytorch/executorch). See below for a [full list of supported devices](#supported-systems).

The library provides:

- Command line interaction with popular LLMs through PyTorch eager and torch.compile
- Export to laptop and desktop through AOT Inductor
- Export to Android and iOS through [ExecuTorch](https://github.com/pytorch/executorch)
- Very low latency through quantization and optimized kernels
- Hackable PyTorch models and integration to [torchtune](https://github.com/pytorch/torchtune) for model fine-tuning
- Import of GGUF models
- <1000 lines of python
- Quantization to int8 and int4 for linear and embedding operators
- Support for Nvidia and AMD GPUs, Apple GPUs with MPS, CPU (Linux/x86, MacOS/ARM),  Mobile CPU with XNNPACK, Mobile GPU with Vulkan and CoreML. Hardware-specific delegates through CoreML and HTP.

While we strive to support a broad range of models, we can't test them all. We classify supported models as tested ✅,
work in progress 🚧 or some restrictions ❹.  As always, we invite community contributions of new model suport and test results!

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

## Installation

Currently `torchchat` must be built via cloning the repository and installing as follows:

```
git clone https://github.com/pytorch/torchchat.git
cd torchchat
pip install -r requirements.txt
```

To confirm that the package is installed correctly, you can run the following command:

```
torchchat --help
```

And should see the following output:

```
usage: torchchat [-h] {chat,generate,eval,export} ...

Welcome to the torchchat CLI!

options:
  -h, --help            show this help message and exit

...
```

If you are planning on use mobile backends, [install ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup.html) and any hardware-specific libraries and IDEs.

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

Exported models can be loaded back into torchchat for chat or text generation, letting you experiment with the exported model and valid model quality. The python interface is the same in all cases and is used for testing nad test harnesses too.

Torchchat comes with server C++ runtimes to execute AOT Inductor and ExecuTorch models. Mobile C++ runtimes allow you to deploy ExecuTorch-compiled .pte files on iOS, Android and Raspberry Pi 5.

## Download

For Llama 2 and 3, follow the instructions on the official [`meta-llama`](https://huggingface.co/meta-llama/Llama-2-7b) repository to ensure you have access to the Llama 2 model weights. Once you have confirmed access, you can run the following command to download the weights to your local machine. This will also download the tokenizer model and a responsible use guide.

```
huggingface-cli login
torchchat --download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf
```

Note: While the ``torchchat download`` command allows you to download *any* model from the hub, there's no guarantee that the model can be run with torchchat. Currently supported models can be found [here](#introduction)

For stories15M, which we use in this quick start guide, run the following:

```
huggingface-cli login
torchchat --download tinyllamas/stories15M --output-dir /tmp/stories15M
```

Some common models are recognized by torchchat based on their filename through `Transformer.from_name()` to perform a fuzzy match against a table of known model architectures. Alternatively, you can specify the index into that table with the option `--params-table ${INDEX}` where the index is the dictionary key in the `transformer_configs`
dictionary specified [here](https://github.com/pytorch/torchchat/blob/main/model.py#L85).  For our example, with the stories15M model, this would be expressed as
`--params-table stories15M`. (We use the model constructor `Transformer.from_table()`)

For models not specified not in the list of known configurations, you can construct the model by initializing the `ModelArgs` dataclass that controls model construction from a parameter json using the `params-path ${PARAMS_PATH}` containing the appropriate model parameters to initialize the ModelArgs for the model. (We use the model constructor `Transformer.from_params()`).

The parameter file will should be in JSON format specifying thee parameters.  You can find the Model Args data class in [`model.py`](https://github.com/pytorch/torchchat/blob/main/model.py#L22).

The final way to initialize a torchchat model is from GGUF. You load a GGUF model with the option `--load-gguf ${MODELNAME}.gguf`. Presently, the F16, F32, Q4_0, and Q6_K formats are supported and converted into native torchchat models.

You may also dequantize GGUF models with the GGUF quantize tool, and then load and requantize with torchchat native quantization options.  (Please note that quantizing and dequantizing is a lossy process, and you will get the best results by starting with the original unquantized model checkpoint, not a previsouly quantized and thend equantized model.)

| GGUF Model | Tested | Eager | torch.compile | AOT Inductor | ExecuTorch | Fits on Mobile |
|-----|--------|-------|-----|-----|-----|-----|
| llama-2-7b.Q4_0.gguf |  🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

You may also dequantize GGUF models with the GGUF quantize tool, and then load and requantize with torchchat native quantization options.  (Please note that quantizing and dequantizing is a lossy process, and you will get the best results by starting with the original unquantized model checkpoint, not a previsoul;y quantized and thend equantized model.)


## Chat

We use several variables in this example, which may be set as a preparatory step:

* `MODEL_NAME` describes the name of the model.  This name is *not* free-form, as it is used to index into a table
   of supported models and their configuration properties that are needed to load the model. This variable should correspond to the
   name of the directory holding the files for the corresponding model.  You *must* follow this convention to
   ensure correct operation.

* `MODEL_DIR` is the location where we store model and tokenizer information for a particular model. We recommend `checkpoints/${MODEL_NAME}`
  or any other directory you already use to store model information.

* `MODEL_PATH` describes the location of the model. Throughput the description
  herein, we will assume that MODEL_PATH starts with a subdirectory of the torchchat repo
  named checkpoints, and that it will contain the actual model. In this case, the MODEL_PATH will thus
  be of the form ${MODEL_OUT}/model.{pt,pth}.  (Both the extensions `pt` and `pth`
  are used to describe checkpoints. In addition, model may be replaced with the name of the model.)

  The generate.py sequence generator will load the tokenizer from the directory specified by the MODEL_PATH variable,
  by replacing the modelname with the name of the tokenizer model which is expected to be named `tokenizer.model`.

* `MODEL_OUT` is a location for outputs from export for server/desktop and/or mobile/edge execution.  We store exported
  artifacts here, with extensions .pte for ExecuTorch models, .so for AOT Inductor generated models, and .bin for tokenizers
  prepared for use with the C++ tokenizers user by `runner-aoti` and `runner-et`.

You can set these variables as follows for the exemplary model15M model from Andrej Karpathy's tinyllamas model family:
```
MODEL_NAME=stories15M
MODEL_DIR=checkpoints/${MODEL_NAME}
MODEL_PATH=${MODEL_OUT}/stories15M.pt
MODEL_OUT=~/torchchat-exports
```

When we export models with AOT Inductor for servers and desktops, and ExecuTorch for mobile and edge devices,
we will save them in the specified directory (`${MODEL_OUT}` in our example below) as a DSO under the name `${MODEL_NAME}.so` (for AOTI-generated dynamic libraries),
or as ExecuTorch model under the name `${MODEL_NAME}.pte` (for Executorch-generated mobile/edge models).

We use `[ optional input ]` to indicate optional inputs, and `[ choice 1 | choice 2 | ... ]` to indicate a choice


### A note on tokenizers

There are two different formats for tokenizers, and both are used in this repo.
1 - for generate.py and Python bindings, we use the Google sentencepiece Python operator. This operator consumes a tokenization model in the `tokenizer.model` format.
2 - for C/C++ inference, we use @Andrej Karpathy's C tokenizer function.  This tokenizer consumes a tokenization model in the 'tokenizer.bin' format.

You can convert tokenizer.model into tokenizer.bin using Andrej's
tokenizer.py utility to convert the tokenizer.model to tokenizer.bin
format:

```
python utils/tokenizer.py --tokenizer-model=${MODEL_DIR}tokenizer.model
```

We will later disucss how to use this model, as described under *STANDALONE EXECUTION* in a Python-free
environment:
```
./run ${MODEL_OUT}/model.{so,pte} -z ${MODEL_OUT}/tokenizer.bin
```

### Llama 3 tokenizer

Add option to load tiktoken tokenizer
```
--tiktoken
```

## Generate

Model definition in model.py, generation code in generate.py. The
model checkpoint may have extensions `pth` (checkpoint and model definition) or `pt` (model checkpoint).
At present, we always use the torchchat model for export and import the checkpoint into this model definition
because we have tested that model with the export descriptions described herein.

```
python generate.py --compile --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --device [ cuda | cpu | mps]
```

To squeeze out a little bit more performance, you can also compile the
prefill with --compile_prefill. This will increase compilation times
though.

## Eval

## Export

Let's start by exporting and running a small model like stories15M.

```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --output-pte-path ${MODEL_OUT}/model.pte
```

### AOT Inductor compilation and execution
```
python export.py --checkpoint-path ${MODEL_PATH} --device {cuda,cpu} --output-dso-path ${MODEL_OUT}/${MODEL_NAME}.so
```

When you have exported the model, you can test the model with the
sequence generator by importing the compiled DSO model with the
`--dso-path ${MODEL_OUT}/${MODEL_NAME}.so` option.  This gives developers the ability to
test their model, run any pre-existing model tests against the
exported model with the same interface, and support additional
experiments to confirm model quality and speed.

```
python generate.py --device {cuda,cpu} --dso-path ${MODEL_OUT}/${MODEL_NAME}.so --prompt "Hello my name is"
```

While we have shown the export and execution of a small model on CPU
or an accelerator such as GPU, most models need to be compressed to
reduce their memory bandwidth requirements and avoid stalling the
execution engines while they are waiting for data.  We use
quantization to achieve this, as described below.


### ExecuTorch mobile compilation

We export the model with the export.py script.  Running this script requires you first install executorch with pybindings, see [here](#setting-up-executorch-and-runner-et).
At present, when exporting a model, the export command always uses the
xnnpack delegate to export.  (Future versions of torchchat will support additional
delegates such as Vulkan, CoreML, MPS, HTP in addition to Xnnpack as they are released for ExecuTorch.)

### Running the model

With the model exported, you can now generate text with the executorch runtime pybindings.  Feel free to play around with the prompt.

```
python generate.py --checkpoint-path ${MODEL_PATH} --pte ${MODEL_OUT}/model.pte --device cpu --prompt "Once upon a time"
```

You can also run the model with the runner-et.  See below under "Standalone Execution".

While we have shown the export and execution of a small model to a mobile/edge
device supported by ExecuTorch, most models need to be compressed to
fit in the target device's memory. We use quantization to achieve this.


## Llama 3 support

How to obtain snapshot (to be filled in when published by Meta, we use internal snapshot]

enable llama3 tokenizer with option `--tiktoken` (see also discussion under tokenizer)

Enable all export options for llama3 as described below

Identify and enable a runner/run.cpp with a binary tiktoken optimizer.  (May already be available in OSS)
we cannot presently run runner/run.cpp with llama3, until we have a C/C++ tokenizer im[plementation
(initial tiktoken is python)

## Optimizing your model for server, desktop and mobile devices

To compress models, torchchat offers a variety of strategies:
* Configurable floating-point precision, depending on backend capabilities (for activations and weights): float32, float16, bfloat16
* weight-quantization: embedding quantization and linear operator quantization
* dynamic activation quantization with weight quantization: a8w4dq

In addition, we support GPTQ for improving the quality of 4b weight-only quantization.  Support for HQQ is a work in progress.

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
python generate.py --dtype [bf16 | fp16 | fp32] ...
python export.py --dtype [bf16 | fp16 | fp32] ...
```

Unlike gpt-fast which uses bfloat16 as default, Torch@ uses float32 as the default. As a consequence you will have to set to `--dtype bf16` or `--dtype fp16` on server / desktop for best performance.

## Making your models fit and execute fast!

Next, we'll show you how to optimize your model for mobile execution
(for ET) or get the most from your server or desktop hardware (with
AOTI). The basic model build for mobile surfaces two issues: Models
quickly run out of memory and execution can be slow. In this section,
we show you how to fit your models in the limited memory of a mobile
device, and optimize execution speed -- both using quantization. This
is the `torchchat` repo after all!

For high-performance devices such as GPUs, quantization provides a way
to reduce the memory bandwidth required to and take advantage of the
massive compute capabilities provided by today's server-based
accelerators such as GPUs. In addition to reducing the memory bandwidth required
to compute a result faster by avoiding stalls, quantization allows
accelerators (which usually have a limited amount of memory) to store and
process larger models than they would otherwise be able to.

We can specify quantization parameters with the --quantize option. The
quantize option takes a JSON/dictionary with quantizers and
quantization options.

generate and export (for both ET and AOTI) can both accept quantization options.  We only show a subset of the combinations
to avoid combinatorial explosion.

#### Embedding quantization (8 bit integer, channelwise & groupwise)

The simplest way to quantize embedding tables is with int8 "channelwise"
(symmetric) quantization, where each value is represented by an 8 bit integer, and
a floating point scale per embedding (channelwise quantization) or one scale for each group of values
in an embedding (groupwise quantization).

*Channelwise quantization*:

We can do this in eager mode (optionally with torch.compile), we use the `embedding` quantizer with
groupsize set to 0 which uses channelwise quantization:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --device cpu
```

Then, export as follows:
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"embedding": {"bitwidth": 8, "groupsize": 0} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_emb8b-gw256.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_int8.pte --prompt "Hello my name is"
```

*Groupwise quantization*:

We can do this in eager mode (optionally with `torch.compile`), we use the `embedding` quantizer by specifying the group size:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"embedding" : {"bitwidth": 8, "groupsize": 8}}' --device cpu
```

Then, export as follows:
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"embedding": {"bitwidth": 8, "groupsize": 8} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_emb8b-gw256.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_emb8b-gw256.pte --prompt "Hello my name is"
```

#### Embedding quantization (4 bit integer, channelwise & groupwise)

Quantizing embedding tables  with int4 provides even higher compression of embedding tables, potentially at
the cost of embedding quality and model outcome quality. In 4-bit embedding table quantization, each value is represented by a 4 bit integer with two values are packed into each byte
to provide greater compression efficiency (potentially at the cost of model quality) over int8 embedding quantization.


*Channelwise quantization*:

We can do this in eager mode (optionally with torch.compile), we use the `embedding` quantizer with
groupsize set to 0 which uses channelwise quantization:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"embedding" : {"bitwidth": 4, "groupsize": 0}}' --device cpu
```

Then, export as follows:
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"embedding": {"bitwidth": 4, "groupsize": 0} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_emb8b-gw256.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_int8.pte --prompt "Hello my name is"
```

*Groupwise quantization*:

We can do this in eager mode (optionally with `torch.compile`), we use the `embedding` quantizer by specifying the group size:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"embedding" : {"bitwidth": 4, "groupsize": 8}}' --device cpu
```

Then, export as follows:
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"embedding": {"bitwidth": 4, "groupsize": 0} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_emb8b-gw256.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_emb8b-gw256.pte --prompt "Hello my name is"
```

#### Linear 8 bit integer quantization (channel-wise and groupwise)
The simplest way to quantize linear operators is with int8 quantization, where each value is represented by an 8-bit integer, and a
floating point scale:

*Channelwise quantization*:

The simplest way to quantize embedding tables is with int8 groupwise
quantization, where each value is represented by an 8 bit integer, and
a floating point scale per group.

We can do this in eager mode (optionally with torch.compile), we use the `linear:int8` quantizer with
groupsize set to 0 which uses channelwise quantization:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --device cpu
```

Then, export as follows using ExecuTorch for mobile backends:
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"linear:int8": {"bitwidth": 8, "groupsize": 0} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_int8.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_int8.pte --checkpoint-path ${MODEL_PATH}  --prompt "Hello my name is"
```

Or, export as follows for server/desktop deployments:
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"linear:int8": {"bitwidth": 8, "groupsize": 0} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_int8.so
```

Now you can run your model with the same command as before:
```
python generate.py --dso-path ${MODEL_OUT}/${MODEL_NAME}_int8.so --checkpoint-path ${MODEL_PATH}  --prompt "Hello my name is"
```

*Groupwise quantization*:

We can do this in eager mode (optionally with `torch.compile`), we use the `linear:int8` quantizer by specifying the group size:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --device cpu
```

Then, export as follows using ExecuTorch:
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"linear:int8": {"bitwidth": 8, "groupsize": 0} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.pte
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.pte --checkpoint-path ${MODEL_PATH} --prompt "Hello my name is"
```

Or, export as follows for :
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"linear:int8": {"bitwidth": 8, "groupsize": 0} }' --output-dso-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.so
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.so --checkpoint-path ${MODEL_PATH} -d fp32 --prompt "Hello my name is"
```

Please note that group-wise quantization works functionally, but has
not been optimized for CUDA and CPU targets where the best
performnance requires a group-wise quantized mixed dtype linear
operator.

#### 4-bit integer quantization (int4)
To compress your model even more, 4-bit integer quantization may be used.  To achieve good accuracy, we recommend the use
of groupwise quantization where (small to mid-sized) groups of int4 weights share a scale.
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant "{'linear:int4': {'groupsize' : 32} }" [ --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.pte | --output-dso-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.dso]
```

Now you can run your model with the same command as before:
```
python generate.py [ --pte-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.pte | --dso-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.dso]  --prompt "Hello my name is"
```

#### 4-bit integer quantization (8da4w)
To compress your model even more, 4-bit integer quantization may be used.  To achieve good accuracy, we recommend the use
of groupwise quantization where (small to mid-sized) groups of int4 weights share a scale.  We also quantize activations to 8-bit, giving
this scheme its name (8da4w = 8b dynamically quantized activations with 4b weights), and boost performance.
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant "{'linear:8da4w': {'groupsize' : 7} }" [ --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_8da4w.pte | ...dso... ]
```

Now you can run your model with the same command as before:
```
python generate.py [ --pte-path ${MODEL_OUT}/${MODEL_NAME}_8da4w.pte | ...dso...]  --prompt "Hello my name is"
```

#### Quantization with GPTQ (gptq)

```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant "{'linear:gptq': {'groupsize' : 32} }" [ --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_gptq.pte | ...dso... ] # may require additional options, check with AO team
```

Now you can run your model with the same command as before:
```
python generate.py [ --pte-path ${MODEL_OUT}/${MODEL_NAME}_gptq.pte | ...dso...]  --prompt "Hello my name is"
```

#### Adding additional quantization schemes (hqq)
We invite contributors to submit established quantization schemes, with accuracy and performance results demonstrating soundness.


# Loading GGUF models

GGUF is a nascent industry standard format and presently torchchat can read  the F16, F32, Q4_0, and Q6_K formats natively and convert them into native torchchat models by using the load-gguf option:

```
--gguf-path <gguf_filename> # all other options as described elsewhere, works for generate and export, for all backends, but cannot be used with --quantize
```

Ypu may then apply the standard quantization options, e.g., to add embedding table quantization as described under quantization. (You cannot directly requantize already quantized formats.  However, you may dequantize them using GGUF tools, and then laod the model into torchchat to quantize wqith torchchat's quantization workflow.)

## Loading unsupported GGUF formats in torchchat

GGUF formats not presently supported natively in torchchat may be converted to one of the supported formats with GGUF's `${GGUF}/quantize` utility to be loaded in torchchat. If you convert to the FP16 or FP32 formats with GGUF's `quantize` utility, you may then requantize these models with torchchat's quantization workflow.

Note that quantizing and dequantizing is a lossy process, and you will get the best results by starting with the original unquantized model checkpoint, not a previously quantized and then dequantized model. This, while you can convert your q4_1 model to FP16 or FP32 GGUF formats and then requantize, you might get better results if you start with the original FP16 or FP32 GGUF format.

To use the quantize tool, install the GGML tools at ${GGUF} . Then, you can, for example, convert a quantized model to f16 format:

```
${GGUF}/quantize --allow-requantize your_quantized_model.gguf fake_unquantized_model.gguf f16
```

# Standalone Execution

In addition to running the exported and compiled models for server, desktop/laptop and mobile/edge devices by loading them in a PyTorch environment under the Python interpreter,
these models can also be executed directly

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

You can also run the model with the runner-et.  This requires you first build the runner.  See instructions [here](#setting-up-executorch-and-runner-et).
After this is done, you can run runner-et with

```
./build/cmake-out/runner_et ${MODEL_OUT}/model.pte -z ${MODEL_OUT}/tokenizer.bin -i "Once upon a time in a land far away"
```

While we have shown the export and execution of a small model to a mobile/edge
device supported by ExecuTorch, most models need to be compressed to
fit in the target device's memory. We use quantization to achieve this.


This has been shown to run on x86. with the proper IDE environment, you can compile for your specific target.
For a GUI integration in iOS and Android, please refer to "Running on a mobile/edge system" in the section below.

Build the runner like this
```
cd ./runner-et
cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build build
```

To run your pte model, use the following command (assuming you already generated the tokenizer.bin tokenizer model):
```
./build/run ${MODEL_OUT}/${MODEL_NAME}{,_int8,_8da4w}.pte -z ${MODEL_OUT}/${MODEL_NAME}.bin
```

## Running on a mobile/edge system

### Android

Check out the [tutorial on how to build an Android app running your PyTorch models with ExecuTorch](https://pytorch.org/executorch/main/llm/llama-demo-android.html), and give your torchchat models a spin.

![Screenshot](https://pytorch.org/executorch/main/_static/img/android_llama_app.png "Android app running Llama model")

Detailed step by step in conjunction with ET Android build, to run on simulator for Android. `scripts/android_example.sh` for running a model on an Android simulator (on Mac)


### iOS

Open the iOS Llama Xcode project at https://github.com/pytorch/executorch/tree/main/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj in Xcode and click Run.
You will need to provide a provisioning profile (similar to what's expected for any iOS dev).

Once you can run the app on you device,
1 - connect the device to you Mac,
2 - copy the model and tokenizer.bin to the iOS Llama app
3 - select the tokenizer and model with the `(...)` control (bottom left of screen, to the left of the text entrybox)


Detailed step by step in conjunction with ET iOS build, to run on simulator for iOS.

# Supported Systems

PyTorch and ExecuTorch support a broad range of devices for running PyTorch with python (using either eager or eager + `torch.compile`) or in a python-free environment with AOT Inductor and ExecuTorch.


| Hardware | OS | Eager | Eager + Compile | AOT Compile | ET Runtime |
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

## Installation Instructions

Some systems require additional installation steps.

Note: External libraries have not been tested for correctness, reliability and safety. Please contact your system vendor if you have system-specific questions.

### macOS (aarch64, x86)

To use `torch.compile`, you should install OpenMP and a compiler with suitable OpenMP support. You can install OpenMP using conda by following the PyTorch installation instructions
at https://github.com/pytorch/pytorch?tab=readme-ov-file#install-dependencies.

Alternatively, you can also find libraries here: https://mac.r-project.org/openmp/ and from other locations. Alternatively, you may install

macOS running on x86 is reaching end-of-life. To use PyTorch on x86 running macOS, you can download prebuilt binaries up to PyTorch 2.2.  You can download recent PyTorch releases and
install them from source.

### iOS CoreML, Vulkan, MPS

List dependencies for these backends

### Setting up ExecuTorch and runner-et
To set-up ExecuTorch, see the instructions [here](executorch_setup.md).

To build and insall the runners, see instructions [here](runner_build.md).

### Tiktoken instructions & instructions for running llama3 without a python environment

for mobile and runner, if we can get a C/C++ tokenizer

&nbsp;

---

&nbsp;

## Acknowledgements

* Georgi Gerganov and his [GGML](https://github.com/ggerganov/ggml) project shining a spotlight on community-based enablement and inspiring so many other projects.
* Andrej Karpathy and his [llama2.c](https://github.com/karpathy/llama2.c) project.  So many great (and simple!) ideas in llama2.c that we have directly adopted (both ideas and code) from his repo.  You can never go wrong by following Andrej's work.
* Bert Maher and his [llama2.so](https://github.com/bertmaher/llama2.so), which built on Andrej's llama2.c and closed the
loop on Llama models with AOTInductor.
* Christian Puhrsch, Horace He, Joe Isaacson and many more for their many contributions in Accelerating GenAI models in
the *"Anything, Fast!"* pytorch.org blogs, and, in particular, Horace He for [GPT, Fast!](https://github.com/pytorch-labs/gpt-fast), which we have
directly adopted (both ideas and code) from his repo.
* Bert Maher, Scott Wolchok, Bin Bao, Chen Yang, Huamin Li and Mu-Chu Li for great collaborations
in building AOTInductor for CPU including for [nanoGPT](https://github.com/karpathy/nanoGPT).

&nbsp;

## Contributing

We welcome any feature requests, bug reports, or pull requests from the community. See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

&nbsp;

## License

Torchchat is released under the [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.

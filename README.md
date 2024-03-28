# llama-fast
A repo for building and using llama on servers, desktops and mobile

The llama-fast repo enables model inference of llama models (and other LLMs) on servers, desktop and mobile devices.
For a list of devices, see below, under *DEVICES*

A goal of this repo, and the design of the PT2 components was to offer seamless integration and consistent workflows.  
Both mobile and server/desktop paths start with torch.export() receiving the same model description.  Similarly,
integration into runners for Python (for initial testing) and Python-free environments (for deployment, in runner-posix
and runner-mobile, respectively) offer very consistent experiences across backends and offer developers consistent interfaces 
and user experience whether they target server, desktop or mobile & edge use cases, and/or all of them.


# Simple and efficient pytorch-native transformer text generation.

Featuring:

* Very low latency
* <1000 lines of python
* No dependencies other than PyTorch and sentencepiece for server, and Executorch for mobile (plus, your mobile IDE, of course)
* int8/int4 quantization
* Supports Nvidia and AMD GPUs, MPS, CPU (Linux/x86 and MacOS/ARM), xnnpack, and backend-specific mobile runtimes ("delegates").

This is NOT intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :) 
Please copy-paste and fork as you desire.

# Supported Models 
The model definition (and much more!) is adopted from gpt-fast, so we support the same models.
See [`gpt-fast` Supported Models](https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#supported-models) for a full list.

# Installation
Follow the [`gpt-fast` installation instructions](https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#installation).

If you are planning on using mobile backends, you should also install ExecuTorch and any hardware-specific libraries and IDEs.

# A note on tokenizers

There are two different formats for tokenizers, and both are used in this repo.
1 - for generat.py and Python bindings, we use the Google sentencepiece Python operator. This operator consumes a tokenization model in the 'tokenizer.model' format.
2 - for C/C++ inference, we use @Andrej Karpathy's C tokenizer function.  This tokenizer consumes a tokenization model in the 'tokenizer.bin' format.

You can convert tokenizer.model into tokenizer.bin using Andrej's tokenizer.py utility to convert the tokenizer.model to tokenizer.bin format:
```
python utils/tokenizer.py --tokenizer-model=/path/to/tokenizer/tokenizer.model
./run ./model.{so,pte} -z path/to/tokenizer/tokenizer.bin
```

# Generate Text

## Eager Execution

Model definition in model.py, generation code in generate.py.

```
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is" --device {cuda,cpu,mps}
```
To squeeze out a little bit more performance, you can also compile the prefill with --compile_prefill. This will increase compilation times though.

## AOT Inductor compilation and execution
```
python export.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --device {cuda,cpu} --out-path ./${MODEL_REPO}.so
```

When you have exported the model, 
Note to self: sopath is missing in the current version. Copy the reported path to ./${MODEL_REPO}.so

```
python generate.py --device {cuda,cpu} --dso ./${MODEL_REPO}.so --prompt "Hello my name is"
```

Note to self: --dso does not currently take an argument, and always loads stories15M.so.

## ExecuTorch mobile compilation

### The basics

Use a small model like stories15M.pt to test the instructions in the following section.

```
python et_export.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth -d fp32 {-xnnpack|-coreml|--mps} --out-path ./${MODEL_REPO}.pte
```

How do run is problematic -- I would love to run it with 
```
python generate.py --pte ./${MODEL_REPO}.pte --prompt "Hello my name is"
```
but *that requires xnnpack to work in python!* 

### Making your models fit and execute fast!

Next, we'll show you how to optimize your model for mobile execution. The basic model build for mobile surfaces two issues:
Models quickly run out of memory and execution can be slow. In this section, we show you how to fit your models in the limited 
memory of a mobile device, and optimize execution speed -- both using quantization. This is the `llama-fast` repo after all!

#### 8 bit integer quantization
The simplest way to quantize is with int8 quantization, where each value is represented by an 8 bit integer, and a 
floating point scale:  
```
python et_export.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth -d fp32 --quant int8 {-xnnpack|-coreml|--mps} --out-path ./${MODEL_REPO}_int8.pte
```

Now you can run your model with the same command as before:
```
python generate.py --ptr ./${MODEL_REPO}_int8.pte --prompt "Hello my name is"
```

#### 4 bit integer quantization (8da4w)
To compress your model even more, 4 bit integer quantization may be used.  To achieve good accuracy, we recommend the use 
of groupwise quantization where (small to mid-sized) groups of int4 weights share a scale.  We also quantize activations to 8 bit, giving 
this scheme its name (8da4w = 8b dynamically quantized activations with 4b weights).
```
python et_export.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth -d fp32 --quant 8da4w {-xnnpack|-coreml|--mps} --out-path ./${MODEL_REPO}_8da4w.pte
```

Now you can run your model with the same command as before:
```
python generate.py --ptr ./${MODEL_REPO}_8da4w.pte --prompt "Hello my name is"
```

#### Quantization with GPTQ (8da4w-gptq)
TBD.


# Standalone Execution 

## Desktop and Server Execution
This has been tested with Linux and x86 (using CPU ~and GPU~), and MacOS and ARM/Apple Silicon.

In addition to running with the generate.py driver in Python, you can also run PyTorch models without the Python runtime, based on Andrej's magnificent llama2.c code.
(Installation instructions courtesy of @Bert Maher's llama2.so)

Build the runner like this
```
cd ./runner-posix
cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build build
```

To run, use the following command (assuming you already generated the tokenizer.bin tokenizer model):
```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./build/run ../${MODEL_REPO}.so -z ../${MODEL_REPO}.bin
```

## Mobile and Edge Execution
This has been shown to run on x86. with the proper IDE environment, you can compile for your specific target. 
For a GUI integration in iOS and Android, please refer to...

Build the runner like this
```
cd ./runner-mobile
cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build build
```

To run your pte model, use the following command (assuming you already generated the tokenizer.bin tokenizer model):
```
./build/run ../${MODEL_REPO}{,_int8,_8da4w}.pte -z ../${MODEL_REPO}.bin
```

# Supported Systems

PyTorch and the mobile Executorch backend support a broad range fo devices for running PyTorch with Python (using either eager or eager + torch.compile) or using a Python-free environment with AOT Inductor , as well as runtimes for executing exported models.


| Hardware | OS | eager | eager + compile | AOT compile | ET Runtime |
|-----|------|-----|-----|-----|-----|
| x86 | Linux | ❎ |  ❎ |  ❎ |  ❎ | 
| x86 | macOS | ? | ? | ? | ? | 
| aarch64 | Linux | ? | ? | ? | ? | 
| aarch64 | macOS | ❎ |  ❎ |  ❎ |  ❎ | 
| AMD GPU | Linux |  ❎ |  ❎ |  ❎ |  ?| 
| Nvidia GPU | Linux | ❎ |  ❎ |  ❎ |  ? | 
| MPS | macOS | ❎ |  ? |  ? |  <chen lai> | 
| MPS | iOS | ❌|❌|❌| ❎ | 
| aarch64 | Android | ❌|❌|❌| ❎ | 
| Mobile GPU (Vulkan) | Android |  ❌|❌|❌| ❎ | 
| CoreML | iOS |  ❌|❌|❌| ❎ | 
| Hexagon DSP | Android | ❌|❌|❌| ❎ | 
| Raspberry Pi 4/5 | Raspbian | ? | ? | ? | ? |
| Raspberry Pi 4/5 | Android | ? | ? | ? | ? |
| ARM 32b (up to v7) | any | ❌|❌|❌|❌|


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

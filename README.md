# llama-fast
A repo for building and using llama on servers, desktops and mobile

The llama-fast repo enables model inference of llama models (and other LLMs) on servers, desktop and mobile devices.
For a list of devices, see below, under *DEVICES*

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
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --device {cuda,cpu} --dso ./${MODEL_REPO}.so
```

Note to self: --dso does not currently take an argument, and always loads stories15M.so.

## ExecuTorch mobile compilation

```
python et_export.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth -d fp32 {-xnnpack|-coreml|--mps} --out-path ./${MODEL_REPO}.pte
```

How do run is problematic -- I would love to run it with 
```
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --device {cuda,cpu} --dso ./${MODEL_REPO}.so
```
but *that requires xnnpack to work in python!* 


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

To run, use the following command:
```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./build/run ../${MODEL_REPO}.so
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

To run your pte model, use the following command:
```
./build/run ../${MODEL_REPO}.pte
```


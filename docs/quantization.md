# Quantization

<!--
[shell default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}" huggingface-cli login
[shell default]: ./install/install_requirements.sh
[shell default]: TORCHCHAT_ROOT=${PWD} ./torchchat/utils/scripts/install_et.sh
-->

## Introduction
Quantization focuses on reducing the precision of model parameters and computations from floating-point to lower-bit integers, such as 8-bit integers.
This approach aims to minimize memory requirements, accelerate inference speeds, and decrease power consumption, making models more feasible for
deployment on edge devices with limited computational resources. For high-performance devices such as GPUs, quantization provides a way to
reduce the required memory bandwidth and take advantage of the massive compute capabilities provided by today's server-based accelerators such as GPUs.

While quantization can potentially degrade the model's performance, the methods supported by torchchat are designed to mitigate this effect,
maintaining a balance between efficiency and accuracy. In this document we provide details on the supported quantization schemes, how to quantize
models with these schemes and a few example of running such quantized models on supported backends.

## Supported Quantization Schemes
### Weight Quantization
| compression | bitwidth| group size | dynamic activation quantization | Eager | AOTI | ExecuTorch |
|--|--|--|--|--|--|--|
| linear (asymmetric) | [4, 8]* | [32, 64, 128, 256]^ | | ✅ | ✅ | 🚧 |
| linear with dynamic activations (symmetric) | | [32, 64, 128, 256]* | a8w4dq | 🚧 |🚧 | ✅ |

### Embedding Quantization

To support the larger vocabularies (e.g. Llama 3), we also recommend
quantizing the embeddings to further reduce the model size for
on-device usecases.

| compression | weight quantization (bitwidth)| weight quantization (group size) | dynamic activation quantization | Eager | AOTI | ExecuTorch |
|--|--|--|--|--|--|--|
| embedding (symmetric) | [4, 8]* | [32, 64, 128, 256]+ | | ✅ | ✅ | ✅ |


>\* These are the only valid bitwidth options.

>** There are many valid group size options, including 512, 1024,
   etc. Note that smaller groupsize tends to be better for preserving
   model quality and accuracy, and larger groupsize for further
   improving performance. Set 0 for channelwise quantization.

>\+ Should support non-power-of-2-groups as well.


## Quantization API

Quantization options are passed in json format either as a config file
(see [cuda.json](../torchchat/quant_config/cuda.json) and
[mobile.json](../torchchat/quant_config/mobile.json)) or a JSON string.

The expected JSON format is described below. Refer to the tables above
for valid `bitwidth` and `groupsize` values.

| compression | JSON string |
|--|--|
| linear (asymmetric) | `'{"linear:int<bitwidth>" : {"groupsize" : <groupsize>}}'` |
| linear with dynamic activations (symmetric) | `'{"linear:a8w4dq" : {"groupsize" : <groupsize>}}'`|
| embedding | `'{"embedding": {"bitwidth": <bitwidth>, "groupsize":<groupsize>}}'` |

See the available quantization schemes [here](https://github.com/pytorch/torchchat/blob/b809b69e03f8f4b75a4b27b0778f0d3695ce94c2/torchchat/utils/quantize.py#L887-L894).

In addition to quantization, the [accelerator](model_customization.md#device)
and [precision](model_customization.md#model-precision) can also be specified.
Preference is given to the args provided in the quantization API over those
provided explicitly (e.g. `--device`).

The expected JSON format is described below. Refer to the links above for valid `device` and `dtype` values.
| config | JSON string |
|--|--|
| accelerator | `'{"executor": {"accelerator": <device>}}'` |
| precision | `'{"precision": {"dtype": <dtype>}}'`|

## Examples
Here are some examples of quantization configurations

[skip default]: begin
* Config file
  ```
  --quantize quant_config.json
  ```
* Only quantize linear layers
  ```
  --quantize '{"linear:a8w4dq": {"groupsize" : 256}}'
  ```
* Quantize linear layers and embedding lookup
  ```
  --quantize '{"embedding": {"bitwidth": 4, "groupsize":32}, "linear:a8w4dq": {"groupsize" : 256}}'
  ```
* Quantize linear layers with specified dtype and device
  ```
  --quantize '{"executor": {"accelerator": "cuda"},
    "precision": {"dtype": "bf16"},
    "linear:int4": {"groupsize" : 256}}'
  ```
[skip default]: end

Quantization recipes can be applied in conjunction with any of the
`chat`, `generate`, `browser`, `server`, and `export` commands.

Below are
examples showcasing eager mode with `generate` and AOTI and ExecuTorch
with `export`.

### Eager mode
```
python3 torchchat.py generate llama3 --prompt "Hello, my name is" --quantize '{"embedding" : {"bitwidth": 8, "groupsize": 0}}'
```
### AOTI
```
python3 torchchat.py export llama3 --quantize '{"embedding": {"bitwidth": 4, "groupsize":32}, "linear:int4": {"groupsize" : 256}}' --output-dso-path llama3.so
python3 torchchat.py generate llama3 --dso-path llama3.so  --prompt "Hello my name is"
```
### ExecuTorch
```
python3 torchchat.py export llama3 --quantize '{"embedding": {"bitwidth": 4, "groupsize":32}, "linear:a8w4dq": {"groupsize" : 256}}' --output-pte-path llama3.pte
python3 torchchat.py generate llama3 --pte-path llama3.pte  --prompt "Hello my name is"
```

## Experimental TorchAO lowbit kernels

WARNING: These kernels only work on devices with ARM CPUs, for example on Mac computers with Apple Silicon.

### Use

#### linear:a8wxdq
The quantization scheme linear:a8wxdq dynamically quantizes activations to 8 bits, and quantizes the weights in a groupwise manner with a specified bitwidth and groupsize.
It takes arguments bitwidth (1, 2, 3, 4, 5, 6, 7), groupsize, and has_weight_zeros (true, false).
The argument has_weight_zeros indicates whether the weights are quantized with scales only (has_weight_zeros: false) or with both scales and zeros (has_weight_zeros: true).
Roughly speaking, {bitwidth: 4, groupsize: 32, has_weight_zeros: false} is similar to GGML's Q4_0 quantization scheme.

You should expect high performance on ARM CPU if groupsize is divisible by 16.  With other platforms and argument choices, a slow fallback kernel will be used.  You will see warnings about this during quantization.

#### embedding:wx
The quantization scheme embedding:wx quantizes embeddings in a groupwise manner with the specified bitwidth and groupsize.  It takes arguments bitwidth (1, 2, 3, 4, 5, 6, 7) and groupsize.  Unlike linear:a8wxdq, embedding:wx always quantizes with scales and zeros.

You should expect high performance on ARM CPU if groupsize is divisible by 32.  With other platforms and argument choices, a slow fallback kernel will be used.  You will see warnings about this during quantization.

### Setup
To use linear:a8wxdq and embedding:wx, you must set up the torchao experimental kernels.  These will only work on devices with ARM CPUs, for example on Mac computers with Apple Silicon.

From the torchchat root directory, run
```
bash torchchat/utils/scripts/build_torchao_ops.sh
```

This should take about 10 seconds to complete.

Note: if you want to use the new kernels in the AOTI and C++ runners, you must pass the flag link_torchao_ops when running the scripts the build the runners.

```
bash torchchat/utils/scripts/build_native.sh aoti link_torchao_ops
```

```
bash torchchat/utils/scripts/build_native.sh et link_torchao_ops
```

Note before running `bash torchchat/utils/scripts/build_native.sh et link_torchao_ops`, you must first install executorch with `bash torchchat/utils/scripts/install_et.sh` if you have not done so already.

### Examples

Below we show how to use the new kernels.  Except for ExecuTorch, you can specify the number of threads used by setting OMP_NUM_THREADS (as is the case with PyTorch in general).  Doing so is optional and a default number of threads will be chosen automatically if you do not specify.

#### Eager mode
```
OMP_NUM_THREADS=6 python3 torchchat.py generate llama3.1 --device cpu --dtype float32 --quantize '{"embedding:wx": {"bitwidth": 2, "groupsize": 32}, "linear:a8wxdq": {"bitwidth": 3, "groupsize": 128, "has_weight_zeros": false}}' --prompt "Once upon a time,"  --num-samples 5
```

#### torch.compile
```
OMP_NUM_THREADS=6 python3 torchchat.py generate llama3.1 --device cpu --dtype float32 --quantize '{"embedding:wx": {"bitwidth": 2, "groupsize": 32}, "linear:a8wxdq": {"bitwidth": 3, "groupsize": 128, "has_weight_zeros": false}}' --compile --prompt "Once upon a time,"  --num-samples 5
```

#### AOTI
```
OMP_NUM_THREADS=6 python torchchat.py export llama3.1 --device cpu --dtype float32 --quantize '{"embedding:wx": {"bitwidth": 2, "groupsize": 32}, "linear:a8wxdq": {"bitwidth": 3, "groupsize": 128, "has_weight_zeros": false}}' --output-dso llama3_1.so
OMP_NUM_THREADS=6 python3 torchchat.py generate llama3.1 --dso-path llama3_1.so --prompt "Once upon a time,"  --num-samples 5
```

If you built the AOTI runner with link_torchao_ops as discussed in the setup section, you can also use the C++ runner:

```
OMP_NUM_THREADS=6 ./cmake-out/aoti_run llama3_1.so -z $HOME/.torchchat/model-cache/meta-llama/Meta-Llama-3.1-8B-Instruct/tokenizer.model -i "Once upon a time," # -l 3
```

#### ExecuTorch
```
python torchchat.py export llama3.1 --device cpu --dtype float32 --quantize '{"embedding:wx": {"bitwidth": 2, "groupsize": 32}, "linear:a8wxdq": {"bitwidth": 3, "groupsize": 128, "has_weight_zeros": false}}' --output-pte llama3_1.pte
```

Note: only the ExecuTorch C++ runner in torchchat when built using the instructions in the setup can run the exported *.pte file.  It will not work with the `python torchchat.py generate` command.

```
./cmake-out/et_run llama3_1.pte -z $HOME/.torchchat/model-cache/meta-llama/Meta-Llama-3.1-8B-Instruct/tokenizer.model -l3 -i "Once upon a time,"
```

## Experimental TorchAO MPS lowbit kernels

WARNING: These kernels only work on devices with Apple Silicon.

### Use

#### linear:afpwx
The quantization scheme linear:afpwx quantizes only the weights in a groupwise manner with a specified bitwidth and groupsize.
It takes arguments bitwidth (1, 2, 3, 4, 5, 6, 7) and groupsize (32, 64, 128, 256).

### Setup
To use linear:afpwx, you must set up the torchao mps experimental kernels. These will only work on device with Apple Silicon.
Currently, torchchat can only run them on Eager mode.

From the torchchat root directory, run
```
bash torchchat/utils/scripts/build_torchao_ops.sh mps
```

### Examples

#### Eager mode
```
python3 torchchat.py generate stories110M --device mps --dtype float32 --quantize '{"linear:afpwx": {"bitwidth": 4, "groupsize": 256}}' --prompt "Once upon a time," --num-samples 5
```

## Quantization Profiles

Four [sample profiles](https://github.com/pytorch/torchchat/tree/main/torchchat/quant_config/) are included with the torchchat distribution: `cuda.json`, `desktop.json`, `mobile.json`, `pi5.json`
with profiles optimizing for execution on cuda, desktop, mobile and
raspberry Pi devices.

## Adding additional quantization schemes
We invite contributors to submit established quantization schemes, with accuracy and performance results demonstrating soundness.

- Explain terminology, weight size vs activation size, per-channel vs groupwise vs per-tensor, embedding quantization, linear quantization.
- Explain GPTQ, RTN quantization approaches, examples
- Show general form of –quantize parameter
- Describe how to choose a quantization scheme. Which factors should they take into account? Concrete recommendations for use cases, esp. mobile.
- Quantization reference, describe options for --quantize parameter
- Show a table with performance/accuracy metrics
- Quantization support matrix? torchchat Quantization Support Matrix

[end default]: end

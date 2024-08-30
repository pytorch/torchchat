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

See the available quantization schemes [here](https://github.com/pytorch/torchchat/blob/main/torchchat/utils/quantize.py#L1260-L1266).

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

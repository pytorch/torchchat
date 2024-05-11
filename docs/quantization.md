
# Quantization

<!--
[shell default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}" huggingface-cli login
[shell default]: TORCHCHAT_ROOT=${PWD} ./scripts/install_et.sh
-->

## Introduction
Quantization focuses on reducing the precision of model parameters and computations from floating-point to lower-bit integers, such as 8-bit integers. This approach aims to minimize memory requirements, accelerate inference speeds, and decrease power consumption, making models more feasible for deployment on edge devices with limited computational resources. For high-performance devices such as GPUs, quantization provides a way to reduce the required memory bandwidth and take advantage of the massive compute capabilities provided by today's server-based accelerators such as GPUs.

While quantization can potentially degrade the model's performance, the methods supported by torchchat are designed to mitigate this effect, maintaining a balance between efficiency and accuracy. In this document we provide details on the supported quantization schemes, how to quantize models with these schemes and a few example of running such quantized models on supported backends.

## Supported Quantization Schemes
### Weight Quantization
| compression | FP Precision | bitwidth| group size | dynamic activation quantization | Eager | AOTI | ExecuTorch |
|--|--|--|--|--|--|--|--|
| linear (asymmetric) | fp32, fp16, bf16 | [8, 4]* | [32, 64, 128, 256]** | | ✅ | ✅ | 🚧 |
| linear with GPTQ*** (asymmetric) | | |[32, 64, 128, 256]**  | | ✅ | ✅ | ❌ |
| linear with HQQ*** (asymmetric) | | |[32, 64, 128, 256]**  | | ✅ | ✅ | ❌ |
| linear with dynamic activations (symmetric) | fp32^ | | [32, 64, 128, 256]* | a8w4dq | 🚧 |🚧 | ✅ |

### Embedding Quantization

Due to the larger vocabulary size of llama3, we also recommend
quantizing the embeddings to further reduce the model size for
on-device usecases.

| compression | FP Precision | weight quantization (bitwidth)| weight quantization (group size) | dynamic activation quantization | Eager | AOTI | ExecuTorch |
|--|--|--|--|--|--|--|--|
| embedding (symmetric) | fp32, fp16, bf16 | [8, 4]* | [ any > 1 ] | | ✅ | ✅ | ✅ |

^ a8w4dq quantization scheme requires model to be converted to fp32,
  due to lack of support for fp16 and bf16 in the kernels provided with
  ExecuTorch.

* These are the only valid bitwidth options.

** There are many valid group size options, including 512, 1024,
   etc. Note that smaller groupsize tends to be better for preserving
   model quality and accuracy, and larger groupsize for further
   improving performance. Set 0 for channelwise quantization.

*** [GPTQ](https://arxiv.org/abs/2210.17323) and
    [HQQ](https://mobiusml.github.io/hqq_blog/) are two different
    algorithms to address accuracy loss when using lower bit
    quantization. Due to HQQ relying on data/calibration free
    quantization, it tends to take less time to quantize model.

## Quantization Profiles

Torchchat quantization supports profiles with multiple settings such
as accelerator, dtype, and quantization specified in a JSON file.
Four sample profiles are included wwith the torchchat distributin in
config/data: `cuda.json`, `desktop.json`, `mobile.json`, `pi5.json`
with profiles optimizing for execution on cuda, desktop, mobile and
raspberry Pi devices.

In addition to quantization recipes described below, the profiles also
enable developers to specify the accelerator and dtype to be used.

At present torchchat supports the fast, cuda, mps, and cpu devices.
The default device in torchchat is "fast". The "fast" device is a
virtual device that defaults to the fastest executor available in the
system, selecting cuda, mps, and cpu in this order.

At present torchchat supports the fast16, fast, bf16, fp16 and fp32
data types. The default data type for models is "fast16".  The
"fast16" data type is a virtual data type that defaults to the best
16-bit floating point data type available on the selected device. The
"fast" data type is a virtual data type that defaults to the best
floating point data type available on the selected device.  ("Best"
tangibly representing a combination of speed and accuracy.)

## Quantization API

Quantization options are passed in json format either as a config file
(see [cuda.json](../config/data/cuda.json) and
[mobile.json](../config/data/mobile.json)) or a JSON string.

The expected JSON format is described below. Refer to the tables above
for valid `bitwidth` and `groupsize` values.

| compression | JSON string |
|--|--|
| linear (asymmetric) | `'{"linear:int<bitwidth>" : {"groupsize" : <groupsize>}}'` |
| linear with dynamic activations (symmetric) | `'{"linear:a8w4dq" : {"groupsize" : <groupsize>}}'`|
| linear with GPTQ (asymmetric) | `'{"linear:int4-gptq" : {"groupsize" : <groupsize>}}'`|
| linear with HQQ (asymmetric) |`'{"linear:hqq" : {"groupsize" : <groupsize>}}'`|
| embedding | `'{"embedding": {"bitwidth": <bitwidth>, "groupsize":<groupsize>}}'` |

See the available quantization schemes [here](https://github.com/pytorch/torchchat/blob/main/quantize.py#L1260-L1266).

## Examples
We can mix and match weight quantization with embedding quantization.

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
[skip default]: end

Quantization recipes can be applied in conjunction with any of the
`chat`, `generate`, `browser` and `export` commands. Below are
examples showcasing eager mode with `generate` and AOTI and ExecuTorch
with `export`.

### Eager mode
```
python3 generate.py [--compile] llama3 --prompt "Hello, my name is" --quantize '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --device cpu
```
### AOTI
```
python3 torchchat.py export llama3 --quantize '{"embedding": {"bitwidth": 4, "groupsize":32}, "linear:int4": {"groupsize" : 256}}' --output-dso-path llama3.so

python3 generate.py llama3 --dso-path llama3.so  --prompt "Hello my name is"
```
### ExecuTorch
```
python3 torchchat.py export llama3 --dtype fp32 --quantize '{"embedding": {"bitwidth": 4, "groupsize":32}, "linear:a8w4dq": {"groupsize" : 256}}' --output-pte-path llama3.pte

python3 generate.py llama3 --pte-path llama3.pte  --prompt "Hello my name is"
```

## Model precision (dtype precision setting)
On top of quantizing models with integer quantization schemes mentioned above, models can be converted to lower bit floating point precision to reduce the memory bandwidth requirement and take advantage of higher density compute available. For example, many GPUs and some of the CPUs have good support for BFloat16 and Float16. This can be taken advantage of via `--dtype` arg as shown below.

[skip default]: begin
```
python3 generate.py --dtype [ fast16 | fast | bf16 | fp16 | fp32] ...
python3 export.py --dtype [ fast16 | fast | bf16 | fp16 | fp32] ...
```
[skip default]: end

Unlike gpt-fast which uses bfloat16 as default, torchchat uses the dtype "fast16" as the default. Torchchat will pick the appropriate 16-bit floating point type available and offering the best performance (for execution with Executorch, macOS/ARM and Linux/x86 platforms).  For macOS, support depends on the OS version, with versions starting with 14.0 supporting bfloat16 as support, and float16 for earlier OS version based on system support for these data types.  

Support for FP16 and BF16 is limited in many embedded processors and -dtype fp32 may be required in some environments. Additional ExecuTorch support for 16-bit floating point types may be added in the future based on hardware support.

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

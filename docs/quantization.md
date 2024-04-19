
# Quantization

### Introduction
Quantization focuses on reducing the precision of model parameters and computations from floating-point to lower-bit integers, such as 8-bit integers. This approach aims to minimize memory requirements, accelerate inference speeds, and decrease power consumption, making models more feasible for deployment on edge devices with limited computational resources. While quantization can potentially degrade the model's performance, the methods supported by torchchat are designed to mitigate this effect, maintaining a balance between efficiency and accuracy.

### Supported quantization techniques

| compression | FP precision |  weight quantization | dynamic activation quantization |
|--|--|--|--|
embedding table (symmetric) | fp32, fp16, bf16 | 8b (group/channel), 4b (group/channel) | n/a |
linear operator (symmetric) | fp32, fp16, bf16 | 8b (group/channel) | n/a |
linear operator (asymmetric) | n/a | 4b (group), a6w4dq | a8w4dq (group) |
linear operator (asymmetric) with GPTQ | n/a | 4b (group) | n/a |
linear operator (asymmetric) with HQQ | n/a |  work in progress | n/a |

### Model precision (dtype precision setting)
You can generate models (for both export and generate, with eager, torch.compile, AOTI, ET, for all backends - mobile at present will primarily support fp32, with all options) specify the precision of the model with

TODO: These need to be commands that can be copy paste
```
python generate.py --dtype [bf16 | fp16 | fp32] ...
python export.py --dtype [bf16 | fp16 | fp32] ...
```

Unlike gpt-fast which uses bfloat16 as default, Torch@ uses float32 as the default. As a consequence you will have to set to --dtype bf16 or --dtype fp16 on server / desktop for best performance.
Support for FP16 and BF16 is limited in many embedded processors.  Additional executorch support for 16-bit floating point types may be added in the future based on hardware support.

## Making your models fit and execute fast!

Next, we'll show you how to optimize your model for mobile execution (for ET) or get the most from your server or desktop hardware (with AOTI). The basic model build for mobile surfaces two issues: Models quickly run out of memory and execution can be slow. In this section, we show you how to fit your models in the limited memory of a mobile device, and optimize execution speed -- both using quantization. This is the torchchat repo after all!
For high-performance devices such as GPUs, quantization provides a way to reduce the memory bandwidth required to and take advantage of the massive compute capabilities provided by today's server-based accelerators such as GPUs. In addition to reducing the memory bandwidth required to compute a result faster by avoiding stalls, quantization allows accelerators (which usually have a limited amount of memory) to store and process larger models than they would otherwise be able to.
We can specify quantization parameters with the --quantize option. The quantize option takes a JSON/dictionary with quantizers and quantization options.
generate and export (for both ET and AOTI) can both accept quantization options. We only show a subset of the combinations to avoid combinatorial explosion.

### Embedding quantization (8 bit integer, channelwise & groupwise)
The simplest way to quantize embedding tables is with int8 "channelwise" (symmetric) quantization, where each value is represented by an 8 bit integer, and a floating point scale per embedding (channelwise quantization) or one scale for each group of values in an embedding (groupwise quantization).

*Channelwise quantization:*

We can do this in eager mode (optionally with torch.compile), we use the embedding quantizer with groupsize set to 0 which uses channelwise quantization:

TODO: Write this so that someone can copy paste
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

*Groupwise quantization:*
We can do this in eager mode (optionally with torch.compile), we use the embedding quantizer by specifying the group size:

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

### Embedding quantization (4 bit integer, channelwise & groupwise)
Quantizing embedding tables with int4 provides even higher compression of embedding tables, potentially at the cost of embedding quality and model outcome quality. In 4-bit embedding table quantization, each value is represented by a 4 bit integer with two values packed into each byte to provide greater compression efficiency (potentially at the cost of model quality) over int8 embedding quantization.

*Channelwise quantization:*
We can do this in eager mode (optionally with torch.compile), we use the embedding quantizer with groupsize set to 0 which uses channelwise quantization:

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

*Groupwise quantization:*
We can do this in eager mode (optionally with torch.compile), we use the embedding quantizer by specifying the group size:

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

### Linear 8 bit integer quantization (channel-wise and groupwise)

The simplest way to quantize linear operators is with int8 quantization, where each value is represented by an 8-bit integer, and a floating point scale:

*Channelwise quantization:*

The simplest way to quantize embedding tables is with int8 groupwise quantization, where each value is represented by an 8 bit integer, and a floating point scale per group.

We can do this in eager mode (optionally with torch.compile), we use the linear:int8 quantizer with groupsize set to 0 which uses channelwise quantization:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --device cpu
```

Then, export as follows using Executorch for mobile backends:

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

*Groupwise quantization:*
We can do this in eager mode (optionally with torch.compile), we use the linear:int8 quantizer by specifying the group size:

```
python generate.py [--compile] --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --device cpu
```
Then, export as follows using Executorch:

```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"linear:int8": {"bitwidth": 8, "groupsize": 0} }' --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.pte
```

**Now you can run your model with the same command as before:**

```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.pte --checkpoint-path ${MODEL_PATH} --prompt "Hello my name is"
```
*Or, export*
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant '{"linear:int8": {"bitwidth": 8, "groupsize": 0} }' --output-dso-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.so
```

Now you can run your model with the same command as before:
```
python generate.py --pte-path ${MODEL_OUT}/${MODEL_NAME}_int8-gw256.so --checkpoint-path ${MODEL_PATH} -d fp32 --prompt "Hello my name is"
```
Please note that group-wise quantization works functionally, but has not been optimized for CUDA and CPU targets where the best performnance requires a group-wise quantized mixed dtype linear operator.

**4-bit integer quantization (int4)**
To compress your model even more, 4-bit integer quantization may be used. To achieve good accuracy, we recommend the use of groupwise quantization where (small to mid-sized) groups of int4 weights share a scale.

```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant "{'linear:int4': {'groupsize' : 32} }" [ --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.pte | --output-dso-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.dso]
```
Now you can run your model with the same command as before:

```
python generate.py [ --pte-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.pte | --dso-path ${MODEL_OUT}/${MODEL_NAME}_int4-gw32.dso]  --prompt "Hello my name is"
```
**4-bit integer quantization (8da4w)**
To compress your model even more, 4-bit integer quantization may be used. To achieve good accuracy, we recommend the use of groupwise quantization where (small to mid-sized) groups of int4 weights share a scale. We also quantize activations to 8-bit, giving this scheme its name (8da4w = 8b dynamically quantized activations with 4b weights), and boost performance.

```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant "{'linear:8da4w': {'groupsize' : 7} }" [ --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_8da4w.pte | ...dso... ]
```

Now you can run your model with the same command as before:

```
python generate.py [ --pte-path ${MODEL_OUT}/${MODEL_NAME}_8da4w.pte | ...dso...]  --prompt "Hello my name is"
```

**Quantization with GPTQ (gptq)**
```
python export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant "{'linear:gptq': {'groupsize' : 32} }" [ --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_gptq.pte | ...dso... ] # may require additional options, check with AO team
```
Now you can run your model with the same command as before:

```
python generate.py [ --pte-path ${MODEL_OUT}/${MODEL_NAME}_gptq.pte | ...dso...]  --prompt "Hello my name is"
```
**Adding additional quantization schemes (hqq)**
We invite contributors to submit established quantization schemes, with accuracy and performance results demonstrating soundness.

- Explain terminology, weight size vs activation size, per-channel vs groupwise vs per-tensor, embedding quantization, linear quantization.
- Explain GPTQ, RTN quantization approaches, examples
- Show general form of –quantize parameter
- Describe how to choose a quantization scheme. Which factors should they take into account? Concrete recommendations for use cases, esp. mobile.
- Quantization reference, describe options for –quant parameter
- Show a table with performance/accuracy metrics
- Quantization support matrix? torchat Quantization Support Matrix

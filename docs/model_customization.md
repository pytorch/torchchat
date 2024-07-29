# Model Customization

By default, torchchat (and PyTorch) default to unquantized [eager execution](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/).

This page goes over the different options torchchat provides for customizing the model upon which inference is done.
- Compilation
- Quantization
- Model Precision
- Device


## Compilation: JIT-compiled execution
```
python3 (chat | generate | browser | server | eval) [--compile] ...
```

To improve performance, you can compile the model with `--compile`;
trading off the time to first token processed with time per token.

To improve performance further, you may also compile the prefill with
`--compile_prefill`. This will increase further compilation times though.

To learn more about compilation, check out: https://pytorch.org/get-started/pytorch-2.0/


## Quantization

```
python3 (chat | generate | browser | server | export | eval) [--quantize] <quant.json> ...
```

To minimize memory requirements, accelerate inference speeds, and decrease power consumption the model can be also be quantized.
Torchchat leverages [torchao](https://github.com/pytorch/ao) for quantization.

See the [quantization guide](quantization.md) for more details.


## Model Precision

```
python3 (chat | generate | browser | server | export | eval) --dtype [fast | fast16 | bf16 | fp16 | fp32] ...
```

To reduce the memory bandwidth requirement and take advantage of higher density compute available, the model can use lower precision floating point representations.
For example, many GPUs and some of the CPUs have good support for bfloat16 and
float16.

See the [precision guide](quantization.md#model-precision-dtype-precision-setting) for more details.


## Device

```
python3 (chat | generate | browser | server | export | eval) --device [fast | cpu | cuda | mps] ...
```

To leverage a specific accelerator, the target device can be set.

A virtual "fast" device is provided that defaults to the fastest executor available
in the system, selecting cuda, mps, and cpu in this order.

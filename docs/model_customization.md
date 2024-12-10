# Model Customization

By default, torchchat (and PyTorch) defaults to unquantized [eager execution](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/).

This page goes over the different options torchchat provides for customizing the model execution for inference.
- Device
- Compilation
- Model Precision
- Quantization


## Device

```
python3 (chat | generate | browser | server | export | eval) --device [cpu | cuda | mps] ...
```

To leverage a specific accelerator, the target device can be set.

By default, torchchat defaults to the fastest executor available in the system, chosen in this
order: cuda, mps, and cpu.


## Compilation: JIT-compiled execution
```
python3 (chat | generate | browser | server | eval) [--compile][--compile_prefill] ...
```

To improve performance, you can compile the model with `--compile`;
trading off the time to first token processed with time per token.

To improve performance further, at the cost of increased compile time, you may also compile the
prefill with `--compile_prefill`.

To learn more about compilation, check out: https://pytorch.org/get-started/pytorch-2.0/

For CPU, you can use `--max-autotune` to further improve the performance with `--compile` and `compile-prefill`.

See [`max-autotune on CPU tutorial`](https://pytorch.org/tutorials/prototype/max_autotune_on_CPU_tutorial.html).

## Model Precision

```
python3 (chat | generate | browser | server | export | eval) --dtype [fast | fast16 | bf16 | fp16 | fp32] ...
```

To reduce the memory bandwidth requirement and to take advantage of higher density compute available,
the model can use lower precision floating point representations.
For example, many GPUs and some of the CPUs have good support for bfloat16 and float16.

Unlike gpt-fast which uses bfloat16 as default, torchchat uses the dtype
"fast16". This picks the best performing 16-bit floating point type
available (for execution with Executorch, macOS/ARM and Linux/x86 platforms).
For example on macOS, support depends on the OS version, with versions starting
with 14.0 supporting bfloat16 as support, and float16 for earlier OS version
based on system support for these data types.

The "fast" data type is also provided as a virtual data type that defaults
to the best floating point data type available on the selected device.
Currently, this behaves the same as "fast16", but with "fp32" when exporting
to ExecuTorch.


## Quantization

```
python3 (chat | generate | browser | server | export | eval) [--quantize] <quant.json> ...
```

To further minimize memory requirements, accelerate inference speeds, and
decrease power consumption the model can also be quantized.
Torchchat leverages [torchao](https://github.com/pytorch/ao) for quantization.

See the [quantization guide](quantization.md) for examples and more details.

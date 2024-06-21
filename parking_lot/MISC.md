| compression | FP precision |  weight quantization | dynamic activation quantization |
|--|--|--|--|
embedding table (symmetric) | fp32, fp16, bf16 | 8b (group/channel), 4b (group/channel) | n/a |
linear operator (symmetric) | fp32, fp16, bf16 | 8b (group/channel) | n/a |
linear operator (asymmetric) | n/a | 4b (group), a6w4dq | a8w4dq (group) |
linear operator (asymmetric) with GPTQ | n/a | 4b (group) | n/a |

## Model precision (dtype precision setting)
On top of quantizing models with quantization schemes mentioned above, models can be converted to lower bit floating point precision to reduce the memory bandwidth requirement and take advantage of higher density compute available. For example, many GPUs and some of the CPUs have good support for bfloat16 and float16. This can be taken advantage of via `--dtype arg` as shown below.

```
python3 generate.py --dtype [bf16 | fp16 | fp32] ...
python3 export.py --dtype [bf16 | fp16 | fp32] ...
```

**Unlike gpt-fast which uses bfloat16 as default, Torchchat uses
  float32 as the default. As a consequence you will have to set to
  `--dtype bf16` or `--dtype fp16` on server / desktop for best
  performance.**




#### Quantization with GPTQ (gptq)

```
python3 export.py --checkpoint-path ${MODEL_PATH} -d fp32 --quant "{'linear:gptq': {'groupsize' : 32} }" [ --output-pte-path ${MODEL_OUT}/${MODEL_NAME}_gptq.pte | ...dso... ] # may require additional options, check with AO team
```

Now you can run your model with the same command as before:
```
python3 generate.py [ --pte-path ${MODEL_OUT}/${MODEL_NAME}_gptq.pte | ...dso...]  --prompt "Hello my name is"
```



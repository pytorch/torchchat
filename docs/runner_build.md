# Building runner-aoti and runner-et
Building the runners is straightforward and is covered in the next sections.

## Building and running runner-aoti
To build runner-aoti, run the following commands *from the torchchat root directory*

```
cmake -S ./runner-aoti -B ./runner-aoti/cmake-out -G Ninja -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build ./runner-aoti/cmake-out
```

After running these, the runner-aoti binary is located at ./runner-aoti/cmake-out/run.  It can be run as follows:

```
./runner-aoti/cmake-out/run ${AOTI_SO_PATH} -z ${TOKENIZER_DOT_BIN_PATH} -i ${PROMPT}
```

## Building and running runner-et
Before building runner-et, you must first set-up ExecuTorch by following [Set-up Executorch](executorch_setup.md).


To build runner-et, run the following commands *from the torchchat root directory*

```
export TORCHCHAT_ROOT=${PWD}
cmake -S ./runner-et -B ./runner-et/cmake-out -G Ninja
cmake --build ./runner-et/cmake-out
```

After running these, the runner-et binary is located at ./runner-et/cmake-out/runner-et.  It can be run as follows:

```
./runner-et/cmake-out/runner_et ${ET_PTE_PATH} -z ${TOKENIZER_DOT_BIN_PATH} -i ${PROMPT}
```

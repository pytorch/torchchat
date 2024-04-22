# Building runner-aoti and runner-et
Building the runners is straightforward and is covered in the next sections.  We will showcase the runners using stories15M.

The runners accept the following CLI arguments:

```
Options:
-t <float>  temperature in [0,inf], default 1.0
-p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
-s <int>    random seed, default time(NULL)
-n <int>    number of steps to run for, default 256. 0 = max_seq_len
-i <string> input prompt
-z <string> optional path to custom tokenizer
-m <string> mode: generate|chat, default: generate
-y <string> (optional) system prompt in chat mode
```

## Building and running runner-aoti
To build runner-aoti, run the following commands *from the torchchat root directory*

```
cmake -S ./runner-aoti -B ./runner-aoti/cmake-out -G Ninja -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build ./runner-aoti/cmake-out
```

After running these, the runner-aoti binary is located at ./runner-aoti/cmake-out/run.

Let us try using it with an example.
We first download stories15M and export it to AOTI.

```
python torchchat.py download stories15M
python torchchat.py export stories15M --output-dso-path ./model.so
```

We can now execute the runner with:

```
./runner-aoti/cmake-out/run ./model.so -i "Once upon a time"
```

## Building and running runner-et
Before building runner-et, you must first set-up ExecuTorch by following [Set-up Executorch](executorch_setup.md).


To build runner-et, run the following commands *from the torchchat root directory*

```
export TORCHCHAT_ROOT=${PWD}
cmake -S ./runner-et -B ./runner-et/cmake-out -G Ninja
cmake --build ./runner-et/cmake-out
```

After running these, the runner-et binary is located at ./runner-et/cmake-out/run.

Let us try using it with an example.
We first download stories15M and export it to ExecuTorch.

```
python torchchat.py download stories15M
python torchchat.py export stories15M --output-pte-path ./model.pte
```

We can now execute the runner with:

```
./runner-et/cmake-out/run ./model.pte -i "Once upon a time"
```

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
# Pull submodules (re2, abseil) for Tiktoken
git submodule sync
git submodule update --init

cmake -S . -B ./cmake-out -G Ninja -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build ./cmake-out --target et_run
```

After running these, the runner-aoti binary is located at ./cmake-out/aoti_run.

Let us try using it with an example.
We first download stories15M and export it to AOTI.

```
python3 torchchat.py download stories15M
python3 torchchat.py export stories15M --output-dso-path ./model.so
```

We can now execute the runner with:

```
wget -O ./tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
./cmake-out/aoti_run ./model.so -z ./tokenizer.bin -i "Once upon a time"
```

## Building and running runner-et
Before building runner-et, you must first setup ExecuTorch by following [setup ExecuTorch steps](executorch_setup.md).


To build runner-et, run the following commands *from the torchchat root directory*

```
# Pull submodules (re2, abseil) for Tiktoken
git submodule sync
git submodule update --init

export TORCHCHAT_ROOT=${PWD}
cmake -S . -B ./cmake-out -G Ninja
cmake --build ./cmake-out --target et_run
```

After running these, the runner-et binary is located at ./cmake-out/et_run.

Let us try using it with an example.
We first download stories15M and export it to ExecuTorch.

```
python3 torchchat.py download stories15M
python3 torchchat.py export stories15M --output-pte-path ./model.pte
```

We can now execute the runner with:

```
wget -O ./tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
./cmake-out/et_run ./model.pte -z ./tokenizer.bin -i "Once upon a time"
```

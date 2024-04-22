# Building runner-aoti and runner-et
Building the runners is straightforward and is covered in the next sections.

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
python torchchat.py export --output-dso-path ./model.dso
```

We also need a tokenizer.bin file for the stories15M model:

```
wget ./tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

We can now execute the runner with:

```
./runner-aoti/cmake-out/run ./model.dso -z ./tokenizer.bin -i "Once upon a time"
```

## Building and running runner-et
Before building runner-et, you must first set-up ExecuTorch by following [Set-up Executorch](executorch_setup.md).


To build runner-et, run the following commands *from the torchchat root directory*

```
export TORCHCHAT_ROOT=${PWD}
cmake -S ./runner-et -B ./runner-et/cmake-out -G Ninja
cmake --build ./runner-et/cmake-out
```

After running these, the runner-et binary is located at ./runner-et/cmake-out/runner-et.

Let us try using it with an example.
We first download stories15M and export it to ExecuTorch.

```
python torchchat.py download stories15M
python torchchat.py export stories15M --output-pte-path ./model.pte
```

We also need a tokenizer.bin file for the stories15M model:

```
wget ./tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

We can now execute the runner with:

```
./runner-et/cmake-out/runner_et ./model.pte -z ./tokenizer.bin -i "Once upon a time"
```

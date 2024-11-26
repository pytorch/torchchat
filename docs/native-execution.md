> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.

# Native Execution

[shell default]: ./install/install_requirements.sh

While Python offers a great environment for training models and
experimentation and research with models, developers often are looking
to use a native execution environment, either to achieve a certain
performance level, or when including a Python is undesirable (e.g., in
a game application that wants to use an LLM for user interaction) or
impossible (devices with limited functionality and memory capacity).

The 'llama runner' is a native standalone application capable of
running a model exported and compiled ahead-of-time with either
Executorch (ET) or AOT Inductor (AOTI). Which model format to use
depends on your requirements and preferences.  Executorch models are
optimized for portability across a range of devices, including mobile
and edge devices.  AOT Inductor models are optimized for a particular
target architecture, which may result in better performance and
efficiency.

Building the runners is straightforward with the included cmake build
files and is covered in the next sections.  We will showcase the
runners using  llama2 7B and llama3.

## What can you do with torchchat's llama runner for native execution?

* Run models natively:
  * [Chat](#chat)
  * [Generate](#generate)
  * ~~[Run via Browser](#browser)~~
* [Building and using llama runner for exported .so files](#run-server)
     * in Chat mode
     * in Generate mode
* [Building and using llama runner for exported .pe files](#run-portable)
     * in Chat mode
     * in Generate mode
* [Building and using llama runner on mobile devices](#run-mobile)
* Appendix:
      * [Tokenizers](#tokenizers)
      * [Validation](#validation)


The runners accept the following command-line arguments:

[skip default]: begin
```
Options:
  -t <float>  temperature in [0,inf], default 1.0
  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9
  -s <int>    random seed, default time(NULL)
  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
  -i <string> input prompt
  -z <string> path to tokenizer
  -m <string> mode: generate|chat, default: generate
  -y <string> (optional) system prompt in chat mode
  -v <int>    (optional) vocab size, default is model-specific.
  -l <int>    (optional) llama version (2 or 3), default 2.
```
[skip default]: end

## Building and running runner-aoti

To build runner-aoti, run the following commands *from the torchchat
root directory*

```
./torchchat/utils/scripts/build_native.sh aoti
```

After running these, the runner-aoti binary is located at ./cmake-out/aoti_run.

Let us try using it with an example.
We first download stories15M and export it to AOTI.

```
python3 torchchat.py download stories15M
python3 torchchat.py export stories15M --output-dso-path ./model.so
```

We can now execute the runner with:

[shell default]: pip install wget
```
curl -OL https://github.com/karpathy/llama2.c/raw/master/tokenizer.model
./cmake-out/aoti_run ./model.so -z ./tokenizer.model -l 2 -i "Once upon a time"
```

The `-l 2` indicates that the model and tokenizer use the llama2 architecture.  If your model is based on llama3, use `-l 3`.

## Building and running runner-et

Before building runner-et, you must first setup ExecuTorch by
following [setup ExecuTorch steps](torchchat/edge/docs/executorch_setup.md).


To build runner-et, run the following commands *from the torchchat
root directory*

```
./torchchat/utils/scripts/build_native.sh et
```

Note: the above script will wipe ./et-build if present and re-install
ExecuTorch to ./et-build, which can take a while.  If you already
installed ExecuTorch, running the commands below will build the
runner, without re-installing ExecuTorch from source:

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
curl -o ./tokenizer.model https://github.com/karpathy/llama2.c/raw/master/tokenizer.model
./cmake-out/et_run ./model.pte -z ./tokenizer.model -l 2 -i "Once upon a time"
```

The `-l 2` indicates that the model and tokenizer use the llama2 architecture.  If your model is based on llama3, use `-l 3`.

## Appendix: Llama runner tokenizers

Tokenizers are essential tools in Natural Language Processing (NLP)
that convert text into smaller units, such as words or phrases, known
as tokens. Two popular tokenizers are SentencePiece and
Tiktoken. [SentencePiece](https://github.com/google/sentencepiece) is
an unsupervised text tokenizer and detokenizer mainly for Neural
Network-based text generation systems where the vocabulary size is
predetermined prior to the neural model training.

Llama2-style models typically use the SentencePiece
tokenizer. Tiktoken is a newer tokenizer originally developed by
OpenAI that allows you to see how many tokens a text string will use
without making an API call. Llama3 uses the Tiktoken tokenizer.

Torchchat includes both Python and C/C++ implementations of both the
SentencePiece and Tiktoken tokenizers that may be used with the Python
and native execution environments, respectively.

## Appendix: Native model verification using the Python environment

After exporting a model, you will want to verify that the model
delivers output of high quality, and works as expected.  Both can be
achieved with the Python environment.  All torchchat Python commands
can work with exported models.  Instead of loading the model from a
checkpoint or GGUF file, use the `--dso-path model.so` and
`--pte-path model.pte` for loading both types of exported models. This
enables you to verify the quality of the exported models, and run any
tests that you may have developed in conjunction with exported models
to enable you to validate model quality.

The `eval` tool evaluates model quality using metrics such as
'perplexity' that are commonly used in the NLP community to evaluate
output quality.  Load your model exported model to evaluate quality
metrics for exported models.  You can find an introduction to the eval
tool in the [README](../README.md) file.

The `generate`, `chat` and `browser` tools enable you to verify that
the exported model works correctly, as a debugging aid if you are
developing your own native execution environment based on the llama
runner provided with torchchat.

[end default]: end

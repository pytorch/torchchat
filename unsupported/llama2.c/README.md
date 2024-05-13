# Enabling Models from Server to Mobile

THIS DIRECTORY AND ITS SUBDIRECTORIES CONTAIN AN UNSUPPORTED EXAMPLE.

This directory is a minimal example for integrating PyTorch models
exported with either AOT Inductor as a shared library, also known as
dynamic shared object (DSO), and as ExecuTorch-exported PTE model file
in a C/C++ app.

The example is derived from Andrej Karpathy's llama2.c executor, as
modified by Bert Maher for llama2.so, and distributed under Andrej's
original license.

Please refer to the documentation at
https://github.com/karpathy/llama2.c (and Bert Maher's
https://github.com/bertmaher/llama2.so for modifications to serve as
execution environment for PyTorch models) for a discussion of
downloading and and preparing tokenizer models and invoking the model.

This runner is limited to llama2-style models using the SentencePiece
tokenizer to highlight the minimum example of how to enable an
arbitrary application to call a PyTorch model in either a DSO or PTE
format.  In additioon to header files, these changes include
maintaining a pointer to the AOT Inductor or ExecTorch runtime
executor, and the `forward()` function in runner/run.cpp as well as
CMake files in runner-aoti and runner-et to build the runner with
Executorch and AOT Inductor runtimes, specifically.


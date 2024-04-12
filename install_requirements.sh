#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install snakeviz for cProfile flamegraph
# Install sentencepiece for llama tokenizer
pip install snakeviz sentencepiece
pip install torchao==0.1

# Install lm-eval for Model Evaluation with lm-evalution-harness
# Install tiktoken for tokenizer
pip install lm-eval tiktoken blobfile

DEVICE="${1:-cpu}"
TORCH_CUDA_NIGHTLY_URL=https://download.pytorch.org/whl/nightly/cu121
TORCH_NIGHTLY_URL=https://download.pytorch.org/whl/nightly/cpu

if [ $DEVICE = "cuda" ]; then
    pip install --pre torch torchvision torchaudio --index-url {TORCH_CUDA_NIGHTLY_URL}
else
    pip install --pre torch torchvision torchaudio --index-url {TORCH_NIGHTLY_URL}
fi

pip install -r ./requirements.txt

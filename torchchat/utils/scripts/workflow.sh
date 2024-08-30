#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Usage:
#   bash script.sh [cpu|cuda] [model_repo] [optional_command]
# Arguments:
#   cpu|cuda: Specify the device to run validation on (cpu or cuda).
#   model_repo: Model repository name to validate (e.g., tinyllamas/stories15M).
#   optional_command: (optional) Specify additional command "compile", "aoti" or "executorch" to run the selected validation.
################################################################################

set -eu

function download_tinyllamas() {
    local MODEL_REPO="$1"
    local FORCE_DOWNLOAD="${2:-false}"
    local CHECKPOINT_DIR="checkpoints/$MODEL_REPO"
    local MODEL_NAME="${MODEL_REPO##*/}"

    if [ "$FORCE_DOWNLOAD" = true ] || [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A "$CHECKPOINT_DIR")" ]; then
        echo "Download checkpoint for $MODEL_REPO"
        rm -rf "$CHECKPOINT_DIR"

        mkdir -p checkpoints/$MODEL_REPO
        pushd checkpoints/$MODEL_REPO
        wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/${MODEL_NAME}.pt"
        wget "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model"
        wget "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"
        popd
    else
        echo "Checkpoint directory for $MODEL_REPO is not empty. Skipping download."
    fi
}

function download_checkpoint() {
    local MODEL_REPO="$1"
    local FORCE_DOWNLOAD="${2:-false}"
    local CHECKPOINT_DIR="checkpoints/$MODEL_REPO"

    if [ "$MODEL_REPO" = "tinyllamas/stories15M" ] || [ "$MODEL_REPO" = "tinyllamas/stories42M" ] || [ "$MODEL_REPO" = "tinyllamas/stories110M" ]; then
        echo "Download checkpoint for $MODEL_REPO"
        download_tinyllamas "$MODEL_REPO" "$FORCE_DOWNLOAD"
        return 0
    fi

    if [ "$FORCE_DOWNLOAD" = true ] || [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A "$CHECKPOINT_DIR")" ]; then
        echo "Download checkpoint for $MODEL_REPO"
        rm -rf "$CHECKPOINT_DIR"
        python3 torchchat.py download --repo-id "$MODEL_REPO"
    else
        echo "Checkpoint directory for $MODEL_REPO is not empty. Skipping download."
    fi
}

function run_validation_e2e() {
    local MODEL_REPO="$1"

    echo ""
    echo "############### Validating ${MODEL_REPO##*/} ###############"

    if [ ! -f "download.py" ]; then
        echo "download.py doesn't exist."
        exit 1
    fi
    download_checkpoint "$MODEL_REPO"

    if [ ! -f "torchchat/cli/convert_hf_checkpoint.py" ]; then
        echo "torchchat/cli/convert_hf_checkpoint.py doesn't exist."
        exit 1
    fi
    bash .ci/scripts/convert_checkpoint.sh "$MODEL_REPO"

    set +e
    CHECKPOINT_PATH="checkpoints/$MODEL_REPO/$CHECKPOINT_FILENAME"
    if [ -z "$ADDITIONAL_ARG" ]; then
        bash .ci/scripts/validate.sh "$CHECKPOINT_PATH" "$DEVICE"
    else
        bash .ci/scripts/validate.sh "$CHECKPOINT_PATH" "$DEVICE" "$ADDITIONAL_ARG"
    fi
}


# List of models to validate
MODEL_REPOS=(
    "tinyllamas/stories15M"
    # "tinyllamas/stories42M"
    "tinyllamas/stories110M"
    "mistralai/Mistral-7B-v0.1"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "openlm-research/open_llama_7b"
    "codellama/CodeLlama-7b-Python-hf"
    "codellama/CodeLlama-34b-Python-hf"
    # "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Llama-2-13b-chat-hf"
    # "meta-llama/Llama-2-70b-chat-hf"
)

PROMPT="Hello, my name is"
DEVICE="${1:-cpu}"
INPUT_MODEL_REPO="${2:-}"
ADDITIONAL_ARG="${3:-}"
CHECKPOINT_FILENAME="model.pth"

echo "###############################################################"
echo "############## Start LLama-fast Model Validation ##############"
echo "###############################################################"
if [ -z "$INPUT_MODEL_REPO" ]; then
    for MODEL_REPO in "${MODEL_REPOS[@]}"; do
        run_validation_e2e "$MODEL_REPO" "$DEVICE"
    done
else
    run_validation_e2e "$INPUT_MODEL_REPO" "$DEVICE" "$ADDITIONAL_ARG"
fi

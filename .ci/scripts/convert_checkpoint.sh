#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


set -eu

function convert_checkpoint() {
    local MODEL_REPO="$1"
    local CHECKPOINT_NAME="${MODEL_REPO##*/}"

    if [[ $CHECKPOINT_NAME == *"stories15M"* || $CHECKPOINT_NAME == *"stories42M"* || $CHECKPOINT_NAME == *"stories110M"* ]]; then
        # We need this to make the workflow unique for all models because convert_hf_checkpoint will always convert the checkpoint to model.pth
        pushd "checkpoints/${MODEL_REPO}"
        if [ ! -f "model.pth" ]; then
            mv "$CHECKPOINT_NAME.pt" "model.pth"
        fi
        popd
        return 0
    fi

    [ -f "build/convert_hf_checkpoint.py" ] || exit 1

    if [ -f "checkpoints/$MODEL_REPO/model.pth" ]; then
        echo "Converted checkpoint already exists. Skipping conversion for $MODEL_REPO."
        return 0
    fi
    echo "Convert Huggingface checkpoint for $MODEL_REPO"
    python3 build/convert_hf_checkpoint.py --checkpoint-dir "checkpoints/$MODEL_REPO"
}


convert_checkpoint $1

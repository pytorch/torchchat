#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


set -u

function generate_eager_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')
    echo "Run inference with eager model for $MODEL_NAME"
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager"
    cat "$MODEL_DIR/output_eager"
}

function generate_compiled_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')
    echo ""############### Run inference with torch.compile for $MODEL_NAME "###############"
    python -W ignore generate.py --compile --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled"
    cat "$MODEL_DIR/output_compiled"
}

function generate_aoti_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')
    echo ""############### Run inference with AOTInductor for $MODEL_NAME "###############"
    python -W ignore export.py --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path "${MODEL_DIR}/${MODEL_NAME}.so" --device "$TARGET_DEVICE"
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --dso-path "$MODEL_DIR/${MODEL_NAME}.so" --prompt "$PROMPT" > "$MODEL_DIR/output_aoti"
    cat "$MODEL_DIR/output_aoti"
}

function generate_executorch_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')
    echo ""############### Run inference with ExecuTorch using XNNPACK for $MODEL_NAME "###############"
    python -W ignore export.py --checkpoint-path "$CHECKPOINT_PATH" --output-pte-path "$MODEL_DIR/${MODEL_NAME}.pte" -d "fp32"
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" --pte-path "$MODEL_DIR/${MODEL_NAME}.pte" > "$MODEL_DIR/output_et"
    cat "$MODEL_DIR/output_et"
}


CHECKPOINT_PATH="$1"
TARGET_DEVICE="${2:-cpu}"
PROMPT="Hello, my name is"

generate_compiled_model_output $CHECKPOINT_PATH $TARGET_DEVICE
generate_aoti_model_output $CHECKPOINT_PATH $TARGET_DEVICE
generate_executorch_model_output $CHECKPOINT_PATH $TARGET_DEVICE

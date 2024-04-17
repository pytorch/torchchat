
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
    echo "Run inference with eager model"
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
}

function generate_compiled_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')

    echo ""############### Run inference with torch.compile "###############"
    echo ""
    echo "******************************************"
    echo "************** non-quantized *************"
    echo "******************************************"
    python -W ignore generate.py --compile --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"

    echo "******************************************"
    echo "******* Emb: channel-wise quantized ******"
    echo "******************************************"
    python -W ignore generate.py --quant '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
    python -W ignore generate.py --compile --quant '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"

    echo "******************************************"
    echo "******** Emb: group-wise quantized *******"
    echo "******************************************"
    python -W ignore generate.py --quant '{"embedding" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
    python -W ignore generate.py --compile --quant '{"embedding" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"

    echo "***********************************************"
    echo "******* Emb: 4bit channel-wise quantized ******"
    echo "***********************************************"
    python -W ignore generate.py --quant '{"embedding" : {"bitwidth": 4, "groupsize": 0, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
    python -W ignore generate.py --compile --quant '{"embedding" : {"bitwidth": 4, "groupsize": 0, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"

    echo "***********************************************"
    echo "******** Emb: 4bit group-wise quantized *******"
    echo "***********************************************"
    python -W ignore generate.py --quant '{"embedding" : {"bitwidth": 4, "groupsize": 8, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
    python -W ignore generate.py --compile --quant '{"embedding" : {"bitwidth": 4, "groupsize": 8, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"

    echo "******************************************"
    echo "******* INT8 channel-wise quantized ******"
    echo "******************************************"
    python -W ignore generate.py --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
    python -W ignore generate.py --compile --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"

    echo "******************************************"
    echo "******** INT8 group-wise quantized *******"
    echo "******************************************"
    python -W ignore generate.py --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
    python -W ignore generate.py --compile --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"

    echo "******************************************"
    echo "******** INT4 group-wise quantized *******"
    echo "******************************************"
    python -W ignore generate.py --quant '{"linear:int4" : {"groupsize": 32}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    cat "$MODEL_DIR/output_eager"
    python -W ignore generate.py --compile --quant '{"linear:int4" : {"groupsize": 32}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
    cat "$MODEL_DIR/output_compiled"
}

function generate_aoti_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')

    echo ""############### Run inference with AOT Inductor "###############"
    echo ""
    echo "******************************************"
    echo "************** non-quantized *************"
    echo "******************************************"
    python -W ignore export.py --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path "${MODEL_DIR}/${MODEL_NAME}.so" --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --dso-path "$MODEL_DIR/${MODEL_NAME}.so" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"

    echo "******************************************"
    echo "******* Emb: channel-wise quantized ******"
    echo "******************************************"
    python -W ignore export.py --quant '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"

    echo "******************************************"
    echo "******** Emb: group-wise quantized *******"
    echo "******************************************"
    python -W ignore export.py --quant '{"embedding" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"

    echo "***********************************************"
    echo "******* Emb: 4bit channel-wise quantized ******"
    echo "***********************************************"
    python -W ignore export.py --quant '{"embedding" : {"bitwidth": 4, "groupsize": 0, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"

    echo "***********************************************"
    echo "******** Emb: 4bit group-wise quantized *******"
    echo "***********************************************"
    python -W ignore export.py --quant '{"embedding" : {"bitwidth": 4, "groupsize": 8, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"

    echo "******************************************"
    echo "******* INT8 channel-wise quantized ******"
    echo "******************************************"
    python -W ignore export.py --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"

    echo "******************************************"
    echo "******** INT8 group-wise quantized *******"
    echo "******************************************"
    python -W ignore export.py --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"

    echo "******************************************"
    echo "******** INT4 group-wise quantized *******"
    echo "******************************************"
    python -W ignore export.py --quant '{"linear:int4" : {"groupsize": 32}}' --checkpoint-path "$CHECKPOINT_PATH" --output-dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --dso-path ${MODEL_DIR}/${MODEL_NAME}.so --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
    cat "$MODEL_DIR/output_aoti"
}

function generate_executorch_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')
    echo ""############### Run inference with ExecuTorch using XNNPACK "###############"
    python -W ignore export.py --checkpoint-path "$CHECKPOINT_PATH" --output-pte-path "$MODEL_DIR/${MODEL_NAME}.pte" -d "fp32" || exit 1
    python -W ignore generate.py --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" --pte-path "$MODEL_DIR/${MODEL_NAME}.pte" > "$MODEL_DIR/output_et" || exit 1
    cat "$MODEL_DIR/output_et"
}

function run_compile() {
    generate_compiled_model_output "$CHECKPOINT_PATH" "$TARGET_DEVICE" || exit 1
}

function run_aoti() {
    generate_aoti_model_output "$CHECKPOINT_PATH" "$TARGET_DEVICE" || exit 1
}

function run_executorch() {
    if [ "$TARGET_DEVICE" = "cpu" ]; then
        generate_executorch_model_output "$CHECKPOINT_PATH" "$TARGET_DEVICE" || exit 1
    else
        echo "Skipped: Executorch doesn't run on ${TARGET_DEVICE}"
    fi
}


CHECKPOINT_PATH="$1"
TARGET_DEVICE="${2:-cpu}"
PROMPT="Hello, my name is"


if [ "$#" -gt 2 ]; then
    # Additional arguments provided
    for arg in "${@:3}"; do
        case "$arg" in
            "compile")
                run_compile || exit 1
                ;;
            "aoti")
                run_aoti || exit 1
                ;;
            "executorch")
                run_executorch || exit 1
                ;;
            *)
                echo "Unknown argument: $arg" >&2
                exit 1
                ;;
        esac
    done
else
    # No additional arguments provided, run all functions
    run_compile
    run_aoti
    run_executorch
fi

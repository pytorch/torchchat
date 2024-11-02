
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
    python3 -W ignore torchchat.py generate --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
    .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
}

function generate_compiled_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')


    if [[ $CHECKPOINT_PATH != *"stories"* && $TARGET_DEVICE == "cuda" ]]; then
        DTYPES="bfloat16"
        EXCLUDE_INT8_QUANT=true
    else
        DTYPES="float32 bfloat16 float16"
        EXCLUDE_INT8_QUANT=false
    fi

    for DTYPE in $DTYPES; do
        echo ""############### Run inference with torch.compile for dtype $DTYPE "###############"
        echo ""
        echo "******************************************"
        echo "************** non-quantized *************"
        echo "******************************************"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
        .ci/scripts/check_gibberish  "$MODEL_DIR/output_compiled"

        echo "******************************************"
        echo "******* Emb: channel-wise quantized ******"
        echo "******************************************"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --quant '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_compiled"

        echo "******************************************"
        echo "******** Emb: group-wise quantized *******"
        echo "******************************************"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --quant '{"embedding" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_compiled"

        echo "***********************************************"
        echo "******* Emb: 4bit channel-wise quantized ******"
        echo "***********************************************"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 4, "groupsize": 0, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --quant '{"embedding" : {"bitwidth": 4, "groupsize": 0, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_compiled"

        echo "***********************************************"
        echo "******** Emb: 4bit group-wise quantized *******"
        echo "***********************************************"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 4, "groupsize": 8, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --quant '{"embedding" : {"bitwidth": 4, "groupsize": 8, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_compiled"

        if [ "${EXCLUDE_INT8_QUANT:-false}" == false ]; then
            echo "******************************************"
            echo "******* INT8 channel-wise quantized ******"
            echo "******************************************"
            python3 -W ignore torchchat.py generate --dtype ${DTYPE} --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
            .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
            python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
            .ci/scripts/check_gibberish "$MODEL_DIR/output_compiled"

            echo "******************************************"
            echo "******** INT8 group-wise quantized *******"
            echo "******************************************"
            python3 -W ignore torchchat.py generate --dtype ${DTYPE} --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
            .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
            python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
            .ci/scripts/check_gibberish "$MODEL_DIR/output_compiled"

            if [[ $TARGET_DEVICE != "cuda" || "$DTYPE" == "bfloat16" ]]; then
                # For CUDA, only bfloat16 makes sense for int4 mm kernel
                echo "******************************************"
                echo "******** INT4 group-wise quantized *******"
                echo "******************************************"
                python3 -W ignore torchchat.py generate --dtype ${DTYPE} --quant '{"linear:int4" : {"groupsize": 32}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_eager" || exit 1
                .ci/scripts/check_gibberish "$MODEL_DIR/output_eager"
                python3 -W ignore torchchat.py generate --dtype ${DTYPE} --compile --quant '{"linear:int4" : {"groupsize": 32}}' --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_compiled" || exit 1
                .ci/scripts/check_gibberish "$MODEL_DIR/output_compiled"
            fi
        fi
    done
}

function generate_aoti_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local DTYPES="${3:-default}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')

    echo "Local DTYPES=$DTYPES"

    if [[ $DTYPES == "default" ]]; then
        if [[ $CHECKPOINT_PATH != *"stories"* && $TARGET_DEVICE == "cuda" ]]; then
            DTYPES="bfloat16"
            EXCLUDE_INT8_QUANT=true
        else
            DTYPES="float32 bfloat16 float16"
            EXCLUDE_INT8_QUANT=false
        fi
    fi

    echo "Local after default DTYPES=$DTYPES"

    for DTYPE in $DTYPES; do
        echo ""############### Run inference with AOT Inductor  for dtype $DTYPE "###############"
        echo ""
        echo "******************************************"
        echo "************** non-quantized *************"
        echo "******************************************"
        python3 -W ignore torchchat.py export --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path "${MODEL_DIR}/${MODEL_NAME}.pt2" --device "$TARGET_DEVICE" || exit 1
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --aoti-package-path "$MODEL_DIR/${MODEL_NAME}.pt2" --prompt "$PROMPT" --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"

        echo "******************************************"
        echo "******* Emb: channel-wise quantized ******"
        echo "******************************************"
        python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" || exit 1
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"

        echo "******************************************"
        echo "******** Emb: group-wise quantized *******"
        echo "******************************************"
        python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" || exit 1
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"

        echo "***********************************************"
        echo "******* Emb: 4bit channel-wise quantized ******"
        echo "***********************************************"
        python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 4, "groupsize": 0, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" || exit 1
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"

        echo "***********************************************"
        echo "******** Emb: 4bit group-wise quantized *******"
        echo "***********************************************"
        python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant '{"embedding" : {"bitwidth": 4, "groupsize": 8, "packed": "True"}}' --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" || exit 1
        python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
        .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"

        if [ "${EXCLUDE_INT8_QUANT:-false}" == false ]; then
            echo "******************************************"
            echo "******* INT8 channel-wise quantized ******"
            echo "******************************************"
            python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 0}}' --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" || exit 1
            python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
            .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"

            echo "******************************************"
            echo "******** INT8 group-wise quantized *******"
            echo "******************************************"
            python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant '{"linear:int8" : {"bitwidth": 8, "groupsize": 8}}' --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" || exit 1
            python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
            .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"
        fi
        echo "******************************************"
        echo "******** INT4 group-wise quantized *******"
        echo "******************************************"
        if [[ "$TARGET_DEVICE" != "cuda" || "$DTYPE" == "bfloat16" ]]; then
            # For CUDA, only bfloat16 makes sense for int4 mm kernel
            python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant '{"linear:int4" : {"groupsize": 32}}' --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" || exit 1
            python3 -W ignore torchchat.py generate --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --temperature 0 --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" > "$MODEL_DIR/output_aoti" || exit 1
            .ci/scripts/check_gibberish "$MODEL_DIR/output_aoti"
        fi
    done
}

function generate_executorch_model_output() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')
    echo ""############### Run inference with ExecuTorch using XNNPACK "###############"
    python3 -W ignore torchchat.py export --checkpoint-path "$CHECKPOINT_PATH" --output-pte-path "$MODEL_DIR/${MODEL_NAME}.pte" -d "fp32" || exit 1
    python3 -W ignore torchchat.py generate --checkpoint-path "$CHECKPOINT_PATH" --prompt "$PROMPT" --device "$TARGET_DEVICE" --pte-path "$MODEL_DIR/${MODEL_NAME}.pte" > "$MODEL_DIR/output_et" || exit 1
    .ci/scripts/check_gibberish "$MODEL_DIR/output_et"
}

function eval_model() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')

    for DTYPE in float32 bfloat16 float16; do
        echo ""############### Run eval with torch.compile for dtype $DTYPE "###############"
        echo ""
        echo "******************************************"
        echo "************** non-quantized *************"
        echo "******************************************"
        python -W ignore torchchat.py eval --compile --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --device "$TARGET_DEVICE" > "$MODEL_DIR/eval" || exit 1
        cat "$MODEL_DIR/eval"
        # extract perplexity number and compare with a constant
        export REF_PERPLEXITY=100000
        export PERPLEXITY=cat "$MODEL_DIR/eval" | tail -n 1 log | awk -F '[, ]' '{print $4}'
        # == 1 meaning the check succeeded
        if [ "$(echo "$PERPLEXITY >= $REF_PERPLEXITY" | bc)" == 1]; then
            echo "perplexity checking failed for non-quantized model $MODEL_NAME with $DTYPE $TARGET_DEVICE"
        else
            echo "perplexity checking succeeded for non-quantized model $MODEL_NAME with $DTYPE $TARGET_DEVICE"
        fi;

        if [[ "$TARGET_DEVICE" != "cuda" || "$DTYPE" == "bfloat16" ]]; then
            echo "******************************************"
            echo "******** INT4 group-wise quantized *******"
            echo "******************************************"

            export QUANT_OPTIONS='{"linear:int4" : {"groupsize": 32}}'
            python -W ignore torchchat.py eval --compile --dtype ${DTYPE} --quant "$QUANT_OPTIONS" --checkpoint-path "$CHECKPOINT_PATH" --device "$TARGET_DEVICE" > "$MODEL_DIR/eval" || exit 1
            cat "$MODEL_DIR/eval"
            export REF_PERPLEXITY=100000
            export PERPLEXITY=cat "$MODEL_DIR/eval" | tail -n 1 log | awk -F '[, ]' '{print $4}'
            # == 1 meaning the check succeeded
            if [ "$(echo "$PERPLEXITY >= $REF_PERPLEXITY" | bc)" == 1]; then
                echo "perplexity checking failed for int4-quantized model $MODEL_NAME with $DTYPE $TARGET_DEVICE $QUANT_OPTIONS"
            else
                echo "perplexity checking succeeded for int4-quantized model $MODEL_NAME with $DTYPE $TARGET_DEVICE $QUANT_OPTIONS"
            fi;
        fi

    done
}

function eval_model_sanity_check() {
    local CHECKPOINT_PATH="$1"
    local TARGET_DEVICE="${2:-cpu}"
    local DTYPES="$3"
    local MODEL_DIR="${CHECKPOINT_PATH%/*}"
    local MODEL_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/\.[^.]*$//')

    for DTYPE in $DTYPES; do
        echo ""############### Run eval with torch.compile for dtype $DTYPE "###############"
        echo ""
        echo "******************************************"
        echo "************** non-quantized *************"
        echo "******************************************"
        python -W ignore torchchat.py eval --compile --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --device "$TARGET_DEVICE" --limit 5 > "$MODEL_DIR/eval" || exit 1
        cat "$MODEL_DIR/eval"

        if [[ "$TARGET_DEVICE" != "cuda" || "$DTYPE" == "bfloat16" ]]; then
            echo "******************************************"
            echo "******** INT4 group-wise quantized *******"
            echo "******************************************"

            export QUANT_OPTIONS='{"linear:int4" : {"groupsize": 32}}'
            python -W ignore torchchat.py eval --compile --dtype ${DTYPE} --quant "$QUANT_OPTIONS" --checkpoint-path "$CHECKPOINT_PATH" --device "$TARGET_DEVICE" --limit 5 > "$MODEL_DIR/eval" || exit 1
            cat "$MODEL_DIR/eval"

            echo "**************************************************"
            echo "******** INT4 group-wise quantized (eager) *******"
            echo "**************************************************"

            python -W ignore torchchat.py eval --dtype ${DTYPE} --quant "$QUANT_OPTIONS" --checkpoint-path "$CHECKPOINT_PATH" --device "$TARGET_DEVICE" --limit 5 > "$MODEL_DIR/eval_eager" || exit 1
            cat "$MODEL_DIR/eval_eager"

            # there is some issues with AOTI cpu and cuda, need to fix and enable the test for cuda as well
            echo "*************************************************"
            echo "******** INT4 group-wise quantized (AOTI) *******"
            echo "*************************************************"
            if [ "$DTYPE" != "float16" ]; then
                python3 -W ignore torchchat.py export --dtype ${DTYPE} --quant "$QUANT_OPTIONS" --checkpoint-path "$CHECKPOINT_PATH" --output-aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --dynamic-shapes --device "$TARGET_DEVICE" || exit 1
                python3 -W ignore torchchat.py eval --dtype ${DTYPE} --checkpoint-path "$CHECKPOINT_PATH" --aoti-package-path ${MODEL_DIR}/${MODEL_NAME}.pt2 --device "$TARGET_DEVICE" --limit 5 > "$MODEL_DIR/output_eval_aoti" || exit 1
                cat "$MODEL_DIR/output_eval_aoti"
            fi;
        fi;

    done
}

function run_compile() {
    generate_compiled_model_output "$CHECKPOINT_PATH" "$TARGET_DEVICE" || exit 1
}

function run_aoti() {
    echo "Passing DTYPES=$DTYPES"
    generate_aoti_model_output "$CHECKPOINT_PATH" "$TARGET_DEVICE" "$DTYPES" || exit 1
}

function run_executorch() {
    if [ "$TARGET_DEVICE" == "cpu" ]; then
        generate_executorch_model_output "$CHECKPOINT_PATH" "$TARGET_DEVICE" || exit 1
    else
        echo "Skipped: Executorch doesn't run on ${TARGET_DEVICE}"
    fi
}

function run_eval(){
    eval_model "$CHECKPOINT_PATH" "$TARGET_DEVICE" || exit 1
}

function run_eval_sanity_check(){
    echo "Passing DTYPES=$DTYPES"
    eval_model_sanity_check "$CHECKPOINT_PATH" "$TARGET_DEVICE" "$DTYPES" || exit 1
}

CHECKPOINT_PATH="$1"
TARGET_DEVICE="${2:-cpu}"
PROMPT="Hello, my name is"

if [ "$#" -gt 2 ]; then
    # Additional arguments provided
    for arg in "${@:3}"; do
        case "$arg" in
            "compile")
                echo "arg:$arg"
                run_compile || exit 1
                ;;
            "aoti")
                echo "arg:$arg"
                DTYPES="default"
                run_aoti || exit 1
                ;;
            "aoti-bfloat16")
                echo "arg:$arg"
                DTYPES="bfloat16"
                run_aoti || exit 1
                ;;
            "aoti-float16")
                echo "arg:$arg"
                DTYPES="float16"
                run_aoti || exit 1
                ;;
            "aoti-float32")
                echo "arg:$arg"
                DTYPES="float32"
                run_aoti || exit 1
                ;;
            "executorch")
                echo "arg:$arg"
                run_executorch || exit 1
                ;;
            "eval")
                echo "arg:$arg"
                run_eval || exit 1
                ;;
            "eval_sanity_check")
                echo "arg:$arg"
                DTYPES="bfloat16 float16 float32"
                run_eval_sanity_check || exit 1
                ;;
            "eval_sanity_check-bfloat16")
                echo "arg:$arg"
                DTYPES="bfloat16"
                run_eval_sanity_check || exit 1
                ;;
            "eval_sanity_check-float16")
                echo "arg:$arg"
                DTYPES="float16"
                run_eval_sanity_check || exit 1
                ;;
            "eval_sanity_check-float32")
                echo "arg:$arg"
                DTYPES="float32"
                run_eval_sanity_check || exit 1
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

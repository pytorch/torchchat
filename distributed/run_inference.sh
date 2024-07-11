#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_dist_inference.sh

NGPU=${NGPU:-"8"}

# TODO: We need to decide how to log for inference.
# by default log just rank 0 output,
# TODO for PP, we need to log the last rank!
LOG_RANK=${LOG_RANK:-0}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

CONFIG_FILE=${CONFIG_FILE:-"./inference_configs/llama3_8B.toml"}

torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
inference.py --job.config_file ${CONFIG_FILE} $overrides

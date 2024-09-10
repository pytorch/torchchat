# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from dataclasses import dataclass
from datetime import timedelta

import torch

from distributed.logging_utils import setup_logging

logger = setup_logging(__name__)


def _warn_overwrite_env(env, val):
    if env in os.environ:
        logger.warning(
            f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
        )
    os.environ[env] = val


TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP = "3"


def init_distributed(init_timeout_seconds: int = 120):
    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    torch.distributed.init_process_group(
        "nccl", timeout=timedelta(seconds=init_timeout_seconds)
    )

    # to mitigate the memory issue that collectives using
    # async_op=True hold memory longer than they should
    # such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    readable_num_params = format_model_params(num_params)
    return readable_num_params


def get_module_size(stage):
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(stage.parameters(), stage.buffers())
        ]
    )
    return model_size


def format_model_params(params):
    """turn the num_params into a readable formatted number"""
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.2f}B"
    elif params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M"
    else:
        return f"{params:,}"


def bytes_to_readable(bytes_value: int, round_to: int = 2) -> str:
    """formatting function to make reading model (stage) sizes easy"""
    GiB = 1024**3  # 1 GiB in bytes
    MiB = 1024**2  # 1 MiB in bytes

    if bytes_value >= GiB:
        value = bytes_value / GiB
        unit = "GiB"
    else:
        value = bytes_value / MiB
        unit = "MiB"

    # Round to 2 decimal places
    rounded_value = round(value, round_to)

    return f"{rounded_value} {unit}"


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""

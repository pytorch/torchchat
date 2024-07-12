# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import time

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from timeit import default_timer as timer
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel
import torch.nn as nn
from torch.distributed._tensor import Replicate, Shard
from distributed.parallel_config import ParallelDims
from torch.distributed.device_mesh import DeviceMesh


from .config_manager import InferenceConfig
from distributed.logging_utils import init_logger, logger



def launch_distributed(
    toml_config: str,
) -> Tuple[Optional[DeviceMesh], Optional[ParallelDims]]:
    """ 
    Initialize distributed related setups if the user specified 
    using distributed inference. If not, this is a no-op.

    Args:
        config: str:
            toml file for the inference config.
    Returns:
        Tuple[Optional[DeviceMesh], Optional[ParallelDims]]: 
            - The first element is an optional DeviceMesh object, 
            which which describes the mesh topology of devices for the DTensor.
            - The second element is an optional ParallelDims object, 
            which represents the parallel dimensions configuration.
    """
    init_logger()
    world_size = int(os.environ["WORLD_SIZE"])
    config = InferenceConfig()
    config.parse_args(toml_config)
    
    print(f"logging here...")
    logger.info(f"***************** from logger")

    assert False, "check"

    parallel_dims = ParallelDims(
        tp=8,
        pp=1,
        world_size=world_size,
    )
    init_distributed()
    world_mesh = parallel_dims.build_mesh(device_type="cuda")

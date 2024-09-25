# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple

from torch.distributed.device_mesh import DeviceMesh

from torchchat.distributed.parallel_config import ParallelDims
from torchchat.distributed.utils import init_distributed
from torchchat.distributed.logging_utils import SingletonLogger

from .config_manager import InferenceConfig


logger = SingletonLogger.get_logger()


def launch_distributed(
    toml_config: str,
) -> Tuple[Optional[DeviceMesh], Optional[ParallelDims]]:
    """ 
    Initialize distributed related setups if the user specified 
    using distributed inference. If not, this is a no-op.

    Args:
        toml_config: str:
            toml file for the inference config.
    Returns:
        Tuple[Optional[DeviceMesh], Optional[ParallelDims]]: 
            - The first element is an optional DeviceMesh object, 
            which which describes the mesh topology of devices for the DTensor.
            - The second element is an optional ParallelDims object, 
            which represents the parallel dimensions configuration.
    """
    #init_logger()  TODO - do we want formatted logging? 
    world_size = int(os.environ["WORLD_SIZE"])
    config = InferenceConfig()
    config.parse_args(toml_config)

    
    logger.info(f"toml parsing completed.  Launching with {world_size} GPUs")
    # review parallel config
    tp = config.parallel.tensor_parallel_degree
    pp = config.parallel.pipeline_parallel_degree
    
    parallel_dims = ParallelDims(
        tp=tp,
        pp=pp,
        world_size=world_size,
    )
    init_distributed()
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    logger.info(f"world_mesh created: {world_mesh}")
    return world_mesh, parallel_dims

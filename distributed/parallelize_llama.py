# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)

import torch.nn as nn
from torch.distributed._tensor import Replicate, Shard
from distributed.parallel_config import ParallelDims
from torch.distributed.device_mesh import DeviceMesh
from distributed.utils import logger


def apply_tp(
    model: nn.Module,
    world_mesh: DeviceMesh,
) -> nn.Module:
    """
    Apply tensor parallelism to the given model. More details can be
    found in https://pytorch.org/tutorials/intermediate/TP_tutorial.html.

    NOTE: The way we apply tp is based on the assumption that the model is a LLaMA model.
    One needs to change the ``parallelize_plan`` we pass in to the TP api if the model
    is not a LLaMA model.


    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        world_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
    Return:
        A :class:`nn.Module` object tensor-parallelized.
    """

    tp_mesh = world_mesh["tp"]

    # TODO: To figure out the TP for the tok_embedding and the linear proj layer.
    # # 1. Parallelize the first embedding and the last linear proj layer
    # # 2. Parallelize the root norm layer over the sequence dim
    # # 3. Shard the first transformer block's inputs
    # model = parallelize_module(
    #     model,
    #     tp_mesh,
    #     {
    #         "tok_embeddings": RowwiseParallel(
    #             input_layouts=Replicate(),
    #             output_layouts=Replicate(),
    #         ),
    #         "output": ColwiseParallel(
    #             input_layouts=Shard(1),
    #             output_layouts=Replicate(),
    #             use_local_output=True,
    #         ),
    #     },
    # )

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.layers:
        layer_plan = {
            "attention": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(
                output_layouts=Replicate(),
                use_local_output=True,
            ),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(
                output_layouts=Replicate(),
                use_local_output=True
            ),
            "feed_forward.w3": ColwiseParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_local_heads = attn_layer.n_local_heads // tp_mesh.size()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism to the model")
    return model


def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """
    Apply tensor parallelism and other parallelism(TODO) to the model for inference.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        world_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        parallel_dims (:class:`ParallelDims`):
            The object of the util class which contains the degree for each parallelism.
    Return:
        A :class:`nn.Module` object parallelized.
    """

    if parallel_dims.tp_enabled:
        model = apply_tp(model, world_mesh)

    return model

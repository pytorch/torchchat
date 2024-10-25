# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)


from torchchat.distributed.parallel_config import ParallelDims

from torchchat.distributed.logging_utils import SingletonLogger
logger = SingletonLogger.get_logger()


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
        model (:class:`nn.Module`):
            Module to be parallelized.
        world_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
    Return:
        A :class:`nn.Module` object tensor-parallelized.
    """

    tp_mesh = world_mesh["tp"]

    # TODO: The commented part can further help with scaling but it will
    # make inference very slow so we disable it for now.
    # Parallelize the token embedding and the last linear proj layer
    # model = parallelize_module(
    #     model,
    #     tp_mesh,
    #     {
    #         "tok_embeddings": ColwiseParallel(
    #             output_layouts=Replicate(),
    #         ),
    #         "output": ColwiseParallel(
    #             output_layouts=Replicate(),
    #         ),
    #     },
    # )

    # NOTE: This is indeed a hack because it assumes that we create cache
    # after we apply TP to the model. Because we don't want to change model code 
    # when applying TP. We need to have change to ensure KVCache has the correct
    # size as k and v.
    model.text_transformer_args.n_local_heads = model.text_transformer_args.n_local_heads // tp_mesh.size()

    # Apply tensor parallelism to every transformer block
    for transformer_block in model.layers:
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        assert attn_layer.n_heads % tp_mesh.size() == 0
        assert attn_layer.n_local_heads % tp_mesh.size() == 0
        assert attn_layer.dim % tp_mesh.size() == 0
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_local_heads = attn_layer.n_local_heads // tp_mesh.size()
        attn_layer.dim = attn_layer.dim // tp_mesh.size()

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
        model (:class:`nn.Module`):
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

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
    SequenceParallel,
)

from distributed.parallel_config import ParallelConfig


def get_tp_parallel_strategy(
    config: ParallelConfig,
) -> Tuple[RowwiseParallel, ColwiseParallel, PrepareModuleInput]:
    """Get the parallel strategy for the transformer model.

    This function handles the special case of using float8 with tensor parallelism.
    """
    if config.fp8_linear == "dynamic":
        from float8_experimental.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        return Float8RowwiseParallel, Float8ColwiseParallel, PrepareFloat8ModuleInput
    return RowwiseParallel, ColwiseParallel, PrepareModuleInput


def apply_tp(model, world_mesh, parallel_dims, config: ParallelConfig):
    """
    Apply tensor parallelism.
    """

    tp_mesh = world_mesh["tp"]
    (
        row_parallel_strategy,
        col_parallel_strategy,
        prepare_module_input,
    ) = get_tp_parallel_strategy(config)
    loss_parallel = parallel_dims.loss_parallel_enabled

    # 1. Parallelize the first embedding and the last linear proj layer
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Shard the first transformer block's inputs
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "output": col_parallel_strategy(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
            "norm": SequenceParallel(),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    for layer_id, transformer_block in model.layers.items():
        layer_plan = {
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": col_parallel_strategy(),
            "attention.wk": col_parallel_strategy(),
            "attention.wv": col_parallel_strategy(),
            "attention.wo": row_parallel_strategy(output_layouts=Shard(1)),
            "attention_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": col_parallel_strategy(),
            "feed_forward.w2": row_parallel_strategy(output_layouts=Shard(1)),
            "feed_forward.w3": col_parallel_strategy(),
            "ffn_norm": SequenceParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_local_heads = attn_layer.n_local_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism to the model")
    return model


def parallelize_llama(model, world_mesh, parallel_dims, config: ParallelConfig):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        model = apply_tp(model, world_mesh, parallel_dims, job_config)

    # only enable TP for now.
    # if job_config.training.compile:
    #     model = apply_compile(model, job_config)

    return model

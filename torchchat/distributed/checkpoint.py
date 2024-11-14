# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Mapping

import torch
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

STATE_DICT_SHARDING_DIM_MAP = {
    "tok_embeddings.weight": 0,
    "attention.wq.weight" : 0,
    "attention.wk.weight" : 0,
    "attention.wv.weight" : 0,
    "attention.wo.weight" : 1,
    "feed_forward.w1.weight" : 0,
    "feed_forward.w2.weight" : 1,
    "feed_forward.w3.weight" : 0,
    "output.weight":0,
}


def _look_up_maybe_shard_for_weight(fqn: str) -> int:
    """
    Look up the sharding dim for the given fqn. If not found, return -1.

    Args:
        fqn (str): Fully qualified name of the parameter.
    Returns:
        int: sharding dim of the parameter.
    """    
    for pattern, value in STATE_DICT_SHARDING_DIM_MAP.items():
        if fqn.endswith(pattern):
            return value
    return -1


def _build_distributed_state_dict(
    state_dict: Mapping[str, Any],
    tp_mesh: DeviceMesh,
) -> Mapping[str, DTensor]:
    """
    Covert the original LLaMa checkpoint from local disk to DTensor
    based distributed state dict so that we can leverage distributed
    checkpoint(DCP) for state_dict resharding and materialization.

    Args:
        state_dict (dict):
            A dict of state_dict loaded from local disk.
        tp_mesh (:class:`DeviceMesh`):
            Object which describes the mesh sub-topology
            of devices for the Tensor Parallelsim.
    Returns:
        A dict of state_dict converted all to DTensor as values.
    """
    dist_state_dict = {}
    for k, v in state_dict.items():
        shard = _look_up_maybe_shard_for_weight(k)
        if shard > 0:
            dist_state_dict[k] = DTensor.from_local(v, tp_mesh, [Shard(shard)], run_check=False)
        else:
            dist_state_dict[k] = DTensor.from_local(v, tp_mesh, [Replicate()], run_check=False)
    return dist_state_dict
            

def _load_checkpoints_from_storage(
    builder_args, #TODO: Need to remove the circular dependency before specifying the type.
    local_rank: int,
)-> Mapping[str, Any]:
    """
    Load the original LLaMa checkpoint from local disk.

    Args:
        builder_args (:class:`BuilderArgs`):
            Command args for model building.
        local_rank (int):
            Local rank for Tensor parallel.
    Returns:
        A dict of state_dict loaded from local disk.
    """
    assert builder_args.dcp_dir is not None, "One needs to specify --dcp-dir to load from storage"
    # NOTE: We made a couple assumptions here:
    # The download.py in TorchChat changed the name of `consolidated.00.pth` to `model.pth`
    # so that we have this hacky logic here. We need to revisit this logic once we can better
    # support large model checkpointing downloading in TorchChat.
    cp_name = "model.pth" if local_rank == 0 else f"consolidated.0{local_rank}.pth"
    checkpoint_path = str(builder_args.checkpoint_path) if local_rank == 0 else os.path.join(builder_args.dcp_dir, cp_name)
    print(f"Loading {cp_name} on rank {local_rank}")
    return torch.load(
        checkpoint_path,
        map_location=builder_args.device,
        mmap=True,
        weights_only=False,
    )


def load_checkpoints_to_model(
    model: nn.Module,
    builder_args, #TODO: Need to remove the circular dependency before specifying the type.
    world_mesh: DeviceMesh,
) -> nn.Module:
    """
    We parallelize the module and load the distributed checkpoint to the model.

    Args:
        model (:class:`nn.Module`):
            Module to be parallelized.
        builder_args (:class:`BuilderArgs`):
            Command args for model building.
        world_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
    Returns:
        A :class:`nn.Module` object which is parallelized and checkpoint loaded.
    """
    tp_mesh = world_mesh["tp"]
    local_rank = tp_mesh.get_local_rank()
    state_dict_storage = _load_checkpoints_from_storage(builder_args, local_rank)
    dist_state_dict = _build_distributed_state_dict(state_dict_storage, tp_mesh)  
    # The format of the state_dict loaded from disk is different from 
    # what we are going to use it for inference. As long as we can represent it 
    # using DTensor, we can leverage DCP for the resharding and materialization.
    CHECKPOINT_DIR = builder_args.dcp_dir / "converted_checkpoints"
    dist_cp.save(
        state_dict=dist_state_dict,
        storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
    )

    model_state_dict = model.state_dict()
    dist_cp.load(
        state_dict=model_state_dict,
        storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
    )
    model.load_state_dict(model_state_dict, assign=True)
    return model

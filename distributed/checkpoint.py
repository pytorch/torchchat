# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed._tensor import DTensor, Replicate, Shard

STATE_DICT_SHARDING_DIM_MAP = {
    "tok_embeddings.weight": 0,
    "attention.wq.weight" : 0,
    "attention.wk.weight" : 0,
    "attention.wv.weight" : 0,
    "attention.wo.weight" : 1,
    "feed_forward.w1.weight" : 0,
    "feed_forward.w2.weight" : 1,
    "feed_forward.w3.weight" : 0,

    "attention_norm.weight" : -1, 
    "ffn_norm.weight": -1,
    "norm.weight" : -1, 
    "output.weight":0,
}


def _get_maybe_shard_for_weight(fqn_key):
    for pattern, value in STATE_DICT_SHARDING_DIM_MAP.items():
        if fqn_key.endswith(pattern):
            return value
    return -1


def _build_distributed_state_dict(state_dict, tp_mesh):
    dist_state_dict = {}
    for k, v in state_dict.items():
        shard = _get_maybe_shard_for_weight(k)
        if shard > 0:
            dist_state_dict[k] = DTensor.from_local(v, tp_mesh, [Shard(shard)], run_check=False)
        else:
            dist_state_dict[k] = DTensor.from_local(v, tp_mesh, [Replicate()], run_check=False)
    return dist_state_dict
            

def _load_checkpoints_from_storage(builder_args, local_rank):
    assert builder_args.checkpoint_dir is not None, "One needs to specify --checkpoint-path to load from storage"
    #NOTE: We made a couple assumptions here: 
    cp_name = "model.pth" if local_rank == 0 else f"consolidated.0{local_rank}.pth"
    checkpoint_path = str(builder_args.checkpoint_path) if local_rank == 0 else os.path.join(builder_args.checkpoint_dir, cp_name)
    print(f"Loading {cp_name} on rank {local_rank}")
    return torch.load(
        checkpoint_path,
        map_location=builder_args.device,
        mmap=True,
    )


def load_checkpoints_to_model(model, builder_args, world_mesh):
    tp_mesh = world_mesh["tp"]
    local_rank = tp_mesh.get_local_rank()
    state_dict_storage = _load_checkpoints_from_storage(builder_args, local_rank)
    dist_state_dict = _build_distributed_state_dict(state_dict_storage, tp_mesh)  
    CHECKPOINT_DIR="converted_checkpoints"
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

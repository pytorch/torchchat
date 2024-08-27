# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run command:
# torchrun --nproc-per-node 4 dist_run.py

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from build.model import TransformerArgs
from build.model_dist import TransformerStage

from distributed.logging_utils import setup_logging
from distributed.safetensor_utils import (
    get_hf_config_file,
    get_hf_weight_map_and_path,
)

_model_name = "Transformer-2-7b-chat-hf"

_name_to_hf_model_id = {
    "Transformer-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
}

# Model config
def main():
    logger = setup_logging(__name__)
    
    config = TransformerArgs.from_name(_model_name)
    logger.info(config)

    # make sure we have valid HF cache for weights and tokenizer
    hf_model_name = _name_to_hf_model_id[_model_name]
    hf_config = get_hf_config_file(hf_model_name)
    logger.info(f"Using HF model weights from {hf_model_name}")


    _mesh_dimensions = (2, 2)  

    # Construct a device mesh with available devices (multi-host or single host)
    device_mesh = dist.init_device_mesh("cuda", _mesh_dimensions, mesh_dim_names=("pp", "tp"))
    tp_mesh = device_mesh["tp"]
    pp_mesh = device_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    nstages = pp_mesh.size()

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Create parallel model with device_mesh context
    with device:
        with tp_mesh:
            model = TransformerStage(config, pp_rank, nstages)
            model.setup_caches(1, 4096)

    logger.info(model)

    # Distributed run
    mbs = 2                         # number of micro-batches
    mb_size = 1                     # micro-batch size
    batch_size = mbs * mb_size      # total batch size
    seqlen = 4096                   # sequence length
    dim = 4096                      # embedding dimension

    # Example input for pipeline stages
    mb_ids = torch.randint(0, config.vocab_size, (mb_size, seqlen), device=device)
    activation = torch.rand(mb_size, seqlen, dim, device=device)
    example_args = mb_ids if pp_rank == 0 else activation

    # Create pipeline stages
    logger.info(f"Creating pipeline stage {pp_rank=}, {nstages=}")
    stage = PipelineStage(
        model, pp_rank, nstages, device,
        input_args=(example_args,),
        group=pp_mesh.get_group(),
    )

    # load weights
    stage_module = stage.submod
    #logger.info(f"{stage.submod=}")
    logger.info(f"{stage_module=}")
    weight_map, weight_path, key_map = get_hf_weight_map_and_path(hf_model_name)
    logger.info(f"{weight_map=}, {weight_path=}, {key_map=}")

    assert False, "check weightmap"

    # Run pipeline
    schedule = ScheduleGPipe(stage, mbs)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), device=device)
    if pp_rank == 0:
        schedule.step(input_ids)
    else:
        output = schedule.step()
        print(f"{output=}")

    dist.destroy_process_group()
    print(f"Rank {rank} completes.")

if __name__ == "__main__":
    main()

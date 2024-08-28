# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run command:
# torchrun --nproc-per-node 4 dist_run.py

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from torchchat.model import TransformerArgs
from torchchat.distributed.model_dist import TransformerStage

# Model config
def main():
    config = TransformerArgs.from_name("Transformer-2-7b-chat-hf")
    print(config)

    # Construct a device mesh with available devices (multi-host or single host)
    device_mesh = dist.init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "tp"))
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

    print(model)

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
    stage = PipelineStage(
        model, pp_rank, nstages, device,
        input_args=(example_args,),
        group=pp_mesh.get_group(),
    )

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

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
    load_safetensor_weights,
)

MODEL_NAME = "Transformer-2-7b-chat-hf"
NAME_TO_HF_MODEL_ID = {
    "Transformer-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
}


def init_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def create_device_mesh(mesh_dimensions):
    return dist.init_device_mesh("cuda", mesh_dimensions, mesh_dim_names=("pp", "tp"))


def load_model_weights(stage_module, hf_model_name, logger):
    weight_map, weight_path, key_map = get_hf_weight_map_and_path(hf_model_name)
    num_loaded_weights, num_missing_weights = load_safetensor_weights(
        stage_module, weight_map, weight_path, key_map
    )
    logger.info(
        f"Success - Loaded {num_loaded_weights} weights, {num_missing_weights} missing weights"
    )
    if num_missing_weights > 0:
        raise ValueError(f"Missing {num_missing_weights} weights")



def main():
    rank, world_size = init_distributed()
    logger = setup_logging(__name__)

    config = TransformerArgs.from_name(MODEL_NAME)
    logger.info(f"Chat Model Config: {config}")

    hf_model_name = NAME_TO_HF_MODEL_ID[MODEL_NAME]
    hf_config = get_hf_config_file(hf_model_name)
    if hf_config is None:
        raise ValueError(f"Config file not found for model id {hf_model_name}")
    logger.info(f"Using HF model weights from {hf_model_name}")

    mesh_dimensions = (2, 2)
    device_mesh = create_device_mesh(mesh_dimensions)

    tp_mesh = device_mesh["tp"]
    pp_mesh = device_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    nstages = pp_mesh.size()
    device = torch.device(f"cuda:{rank}")

    with device:
        with tp_mesh:
            model = TransformerStage(config, pp_rank, nstages)
            model.setup_caches(1, 4096)
    logger.info(f"Model: {model}")

    mbs = 2  # number of micro-batches
    mb_size = 1  # micro-batch size
    batch_size = mbs * mb_size  # total batch size
    seqlen = 4096  # sequence length
    dim = 4096  # embedding dimension

    mb_ids = torch.randint(0, config.vocab_size, (mb_size, seqlen), device=device)
    activation = torch.rand(mb_size, seqlen, dim, device=device)
    example_args = mb_ids if pp_rank == 0 else activation

    logger.info(f"Creating pipeline stage {pp_rank=}, {nstages=}")
    stage = PipelineStage(
        model,
        pp_rank,
        nstages,
        device,
        input_args=(example_args,),
        group=pp_mesh.get_group(),
    )

    
    logger.info(f"Loading weights for {pp_rank=}")
    load_model_weights(stage.submod, hf_model_name, logger)
    assert False, "103: check first tensor load"
    stage.rewrap_embeddings()

    schedule = ScheduleGPipe(stage, mbs)
    logger.info(f"Created schedule: {schedule}")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), device=device)
    logger.info(f"Input: {input_ids}")
    if pp_rank == 0:
        schedule.step(input_ids)
    else:
        output = schedule.step()
        logger.info(f"Output: {output}")

    dist.destroy_process_group()
    logger.info(f"Rank {rank} has completed.")


if __name__ == "__main__":
    main()

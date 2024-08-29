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
from distributed.dtensor_utils import find_cpu_tensors, record_module_dtypes
from distributed.utils import Color as color

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


def load_model_weights(stage_module, hf_model_name, device, logger):
    weight_map, weight_path, key_map = get_hf_weight_map_and_path(hf_model_name)
    num_loaded_weights, num_missing_weights = load_safetensor_weights(
        stage_module, weight_map, weight_path, key_map, device
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

    # TODO - remove this...just debugging issue
    cpu_tensors = find_cpu_tensors(stage.submod)
    logger.info(f"Found {len(cpu_tensors)} cpu tensors: {cpu_tensors}")
    if len(cpu_tensors) > 0:
        raise ValueError("Found cpu tensors in stage")

    # Load weights
    logger.info(f"Loading weights for {pp_rank=} on {device=}")
    # load_model_weights(stage.submod, hf_model_name, device=device, logger=logger)

    # this will set the kvcache layers to float16...
    stage.submod.eval()
    #stage.submod = stage.submod.to(torch.float16)
    #if stage.submod.stage_idx == 0:
        #tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
    #    stage.submod.tok_embeddings = stage.submod.tok_embeddings.to(torch.float32)
                
    

    # TODO - remove this...just debugging issue
    cpu_tensors = find_cpu_tensors(stage.submod)
    logger.info(f"Found {len(cpu_tensors)} cpu tensors: {cpu_tensors}")
    if len(cpu_tensors) > 0:
        raise ValueError("Found cpu tensors in stage")
    
    # verify dtypes
    #dtype_count, dtype_locations, fp32_locations = record_module_dtypes(stage.submod)
    #logger.info(f"Found {len(dtype_count)} dtypes: {dtype_count.items()}")
    #logger.info(f"checkme: Found fp32 {len(fp32_locations)} values: {fp32_locations.keys()}")
    # for name, param in stage.submod.named_parameters():
    #    logger.info(f"{name}: {param.dtype=}")
    #    if 'norm' in name:
    #        logger.info(f"**************   {name=}: {param.dtype=}")
    # logger.info(f"Found {len(dtype_locations)} dtypes: {dtype_locations.items()}")
    #assert False, "inspect dtypes"

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), device=device)

    #logger.info(f"Input: {input_ids.dtype=}, {input_ids.shape=}, {input_ids.device=}")
    # create real inputs
    '''full_batch_prompts = (
        "How do you",
        "I like to",
        "Can I help",
        "You need to",
        "The weather is",
        "I found a",
        "What is your",
        "You are so",
    )  # full batch size = 8

    inputs = tokenizer(full_batch_prompts, return_tensors="pt", padding=True).to(device)
    logger.info(f"check {inputs=}")
    
    # Attach to a schedule
    # number of microbatches = 8 // 2 = 4
    num_mbs = 4
    schedule = ScheduleGPipe(stage, num_mbs)
    '''
    schedule = ScheduleGPipe(stage, mbs)
    logger.info(f"Created schedule: {schedule}")

    with torch.no_grad(): # .inference_mode():
        if pp_rank == 0:
            schedule.step(input_ids)
        else:
            output = schedule.step()
            logger.info(f"Output: {output}")

    logger.info(f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}")

    dist.barrier()
    dist.destroy_process_group()
    


if __name__ == "__main__":
    main()

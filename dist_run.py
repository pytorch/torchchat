# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run command:
# torchrun --nproc-per-node 4 dist_run.py
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from build.model import TransformerArgs, ModelArgs
from build.model_dist import TransformerStage
from distributed.logging_utils import setup_logging

# TODO - these are not distributed specific, consider moving to new package
from distributed.safetensor_utils import (
    get_hf_config_file,
    get_hf_weight_map_and_path,
    load_safetensor_weights,
)
from distributed.dtensor_utils import find_cpu_tensors, record_module_dtypes
from distributed.utils import Color as color
from distributed.verification_utils import extract_and_save_weights
from build.utils import get_precision

MODEL_NAME = "Transformer-2-7b-chat-hf"
NAME_TO_HF_MODEL_ID_AND_DTYPE = {
    "Transformer-2-7b-chat-hf": ("meta-llama/Llama-2-7b-chat-hf", torch.float16),
    "Transformer-3-8b-chat-hf": ("meta-llama/Meta-Llama-3.1-8B-Instruct", torch.bfloat16),

}

def _get_tokenizer(hf_model_name, logger):
    """Load tokenizer from HF model id.  TODO - use torchchat tokenizer?"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    assert tokenizer is not None, f"Failed to load tokenizer for {hf_model_name}"
    logger.info(f"Loaded tokenizer for {hf_model_name}")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def _init_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def _create_device_mesh(mesh_dimensions):
    return dist.init_device_mesh("cuda", mesh_dimensions, mesh_dim_names=("pp", "tp"))


def _load_model_weights(stage_module, hf_model_name, device, logger, config):
    weight_map, weight_path, key_map = get_hf_weight_map_and_path(hf_model_name)
    num_loaded_weights, num_missing_weights = load_safetensor_weights(
        stage_module, weight_map, weight_path, key_map, device, model_config=config
    )
    logger.info(
        f"Success - Loaded {num_loaded_weights} weights, {num_missing_weights} missing weights"
    )
    if num_missing_weights > 0:
        raise ValueError(f"Missing {num_missing_weights} weights")


def _cleanup():
    dist.barrier()
    dist.destroy_process_group()


def main():
    rank, world_size = _init_distributed()
    logger = setup_logging(__name__)

    config = ModelArgs.from_name(MODEL_NAME).text_transformer_args
    logger.info(f"Chat Model Config: {config}")
    # TODO - should we make this work...atm returns float32
    # torchchat_precision = get_precision()

    hf_model_name, model_dtype = NAME_TO_HF_MODEL_ID_AND_DTYPE[MODEL_NAME]
    logger.info(f"Using HF model weights from {hf_model_name} and dtype {model_dtype}")

    hf_config = get_hf_config_file(hf_model_name)
    if hf_config is None:
        raise ValueError(f"Config file not found for model id {hf_model_name}")
    logger.info(f"Using HF model weights from {hf_model_name}")

    tokenizer = _get_tokenizer(hf_model_name, logger)
    
    mesh_dimensions = (2,2)
    device_mesh = _create_device_mesh(mesh_dimensions)

    tp_mesh = device_mesh["tp"]
    pp_mesh = device_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    nstages = pp_mesh.size()
    device = torch.device(f"cuda:{rank}")

    with device:
        with tp_mesh:
            model = TransformerStage(config, pp_rank, nstages)
            model.setup_caches(1, 4096)
            # TODO: refine this .to once we start using fp8 for KV cache
            model = model.to(model_dtype)
    logger.info(f"Model: {model}")

    mbs = 2  # number of micro-batches
    mb_size = 1  # micro-batch size
    batch_size = mbs * mb_size  # total batch size
    seqlen = 4096  # sequence length
    dim = 4096  # embedding dimension


    # Load weights
    logger.info(f"Loading weights for {pp_rank=} on {device=}")
    _load_model_weights(model, hf_model_name, device=device, logger=logger, config=config)

    model.eval()

    # verify weights by tracing
    #output_file = f"model_weights_rank{rank}.csv"
    #extract_and_save_weights(model, output_file)

    logger.info(f"Creating pipeline stage {pp_rank=}, {nstages=}")
    mb_prompts = (
    "How do you",# "I like to",
    )    # microbatch size = 1
    mb_ids = tokenizer(mb_prompts, return_tensors="pt", padding=True).to(device)

    mb_ids = torch.randint(0, config.vocab_size, (mb_size, seqlen), device=device)
    activation = torch.rand(mb_size, seqlen, dim, device=device, dtype=model_dtype)
    example_args = mb_ids if pp_rank == 0 else activation

    stage = PipelineStage(
        model,
        pp_rank,
        nstages,
        device,
        input_args=(example_args,),
        group=pp_mesh.get_group(),
    )

    # this check confirms that there are no cpu tensors in the model..we expect this to be true.
    cpu_tensors = find_cpu_tensors(stage.submod)
    # logger.info(f"Found {len(cpu_tensors)} cpu tensors: {cpu_tensors}")
    if len(cpu_tensors) > 0:
        raise ValueError("Found cpu tensors in stage")

    # TODO: this can likely be removed after we prove out a few more models
    # verify dtypes for model - expect all to be model_dtype except for bool causal_mask atm.
    # dtype_count, dtype_locations, fp32_locations = record_module_dtypes(stage.submod)
    # logger.info(
    #     f"Stage Dtypes - Found {len(dtype_count)} dtypes: {dtype_count.items()}"
    # )
    # assert (
    #     len(dtype_count) == 2
    # ), f"Expected 2 dtypes in model after checkpoint loading: {model_dtype} and {torch.bool}"

    # real input
    
    #mb_prompts = (
    #    "How do you", "I like to",
    #)  # microbatch size = 2
    # Run time inputs
    full_batch_prompts = (
        "What is Snow?","How do you", #"I like to", "Can I help", "You need to",
        #"The weather is", "I found a", "What is your", "You are so",
    )  # full batch size = 8
    inputs = tokenizer(full_batch_prompts,padding="max_length", max_length=seqlen, return_tensors="pt",).to(device)
    input_ids = inputs["input_ids"]
    logger.info(f"Input: {input_ids.dtype=}, {input_ids.shape=}, {input_ids=}")
    
    
    # Attach to a schedule
    # number of microbatches = 4 // 2 = 2
    num_mbs = 2
    #input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), device=device)
    #logger.info(f"Input: {input_ids.dtype=}, {input_ids.shape=}, {input_ids.device=}")

    schedule = ScheduleGPipe(stage, num_mbs)
    logger.info(f"Created schedule: {schedule}")
    output=None

    # Run the model
    with torch.no_grad():  # .inference_mode():
        if pp_rank == 0:
            schedule.step(input_ids)
        else:
            output = schedule.step()
            #logger.info(f"Output: {output.shape=}")
    
    # Decode
    if output is not None:
        logger.info(f"Output: {output.shape=}")
        next_token_logits = output[:, -1, :]
        full_batch_logits = output[:, 0:-1, :]
        logger.info(f"{next_token_logits.shape=}")
        next_token = torch.argmax(next_token_logits, dim=-1)
        next_full_batch = torch.argmax(full_batch_logits, dim=-1)
        logger.info(f"{next_token=}, {(tokenizer.batch_decode(next_token))}")
        logger.info(f"{full_batch_logits=}, {(tokenizer.batch_decode(next_full_batch))}")


    logger.info(
        f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}"
    )

    _cleanup()


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run command:
# torchrun --nproc-per-node 4 dist_run.py
import torch
import torch.distributed as dist

from distributed.verification_utils import find_cpu_tensors
from distributed.logging_utils import setup_logging

# TODO - these are not distributed specific, consider moving to new package
from distributed.safetensor_utils import (
    get_hf_config_file,
    get_hf_weight_map_and_path,
    load_safetensor_weights,
)
from distributed.utils import Color as color
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torchchat.model import ModelArgs, Transformer
from torchchat.utils.build_utils import set_precision

from torchchat.cli.builder import (
    _initialize_model,
    _initialize_tokenizer,
    BuilderArgs,
    TokenizerArgs,
)
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any
from tokenizer.tiktoken import Tokenizer as TiktokenTokenizer
from sentencepiece import SentencePieceProcessor



logger = setup_logging(__name__)

MODEL_NAME = "Meta-Llama-3-8B"
NAME_TO_HF_MODEL_ID_AND_DTYPE = {
    "Transformer-2-7b-chat-hf": ("meta-llama/Llama-2-7b-chat-hf", torch.float16),
    "Meta-Llama-3-8B": ("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16),
    #"Meta-Llama-3-8B":("meta-llama/Meta-Llama-3-8B", torch.bfloat16)
}
CACHE_PRECISION = torch.bfloat16


def _init_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    #torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.cuda.set_device(rank)
    return rank, world_size


def _create_device_mesh(mesh_dimensions):
    return dist.init_device_mesh("cuda", mesh_dimensions, mesh_dim_names=("pp", "tp"))

def dict_to_args(dictionary: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dictionary)

def _build_chat_tokenizer(model_base_name: str = "llama2") -> SentencePieceProcessor | TiktokenTokenizer:
    # Create base args for tokenizer
    default_model_dir = Path(os.getenv("TORCHCHAT_MODELDIR", "~/.torchchat/model-cache")).expanduser()
    
    tokenconfig = {
        'model_directory': default_model_dir,
        'model': model_base_name,
        'tokenizer_path': None
    }
    
    args = dict_to_args(tokenconfig)
    tokenizer_args = TokenizerArgs.from_args(args)
    tokenizer = _initialize_tokenizer(tokenizer_args)
    assert tokenizer is not None, f"Failed to get tokenizer using {tokenconfig=}"
    logger.info(f"using tokenizer = {tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}")
    assert False, "check tok"
    return tokenizer

def _get_hf_tokenizer(hf_model_name):
    """Load tokenizer from HF model id.  TODO - use torchchat tokenizer?"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    assert tokenizer is not None, f"Failed to load tokenizer for {hf_model_name}"
    logger.info(f"Loaded tokenizer for {hf_model_name}")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def _load_model_weights(stage_module, hf_model_name, device, model_config):
    """Load the weights from the safetensor file(s) into the model stage.
    Model config is needed b/c we permute wq and wk weights based on attn heads.
    """

    weight_map, weight_path, key_map = get_hf_weight_map_and_path(hf_model_name)

    num_loaded_weights, num_missing_weights = load_safetensor_weights(
        stage_module,
        weight_map,
        weight_path,
        key_map,
        device,
        model_config=model_config,
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



    config = ModelArgs.from_name(MODEL_NAME).text_transformer_args
    logger.info(f"Chat Model Config: {config}")

    tokenizer = _build_chat_tokenizer()
    logger.info(f"built tokenizer {tokenizer=}")

    hf_model_name, model_dtype = NAME_TO_HF_MODEL_ID_AND_DTYPE[MODEL_NAME]
    logger.info(f"Using HF model weights from {hf_model_name} and dtype {model_dtype}")

    set_precision(CACHE_PRECISION)
    logger.info(f"Using cache precision {CACHE_PRECISION}")

    #tokenizer = _get_tokenizer(hf_model_name)

    hf_config = get_hf_config_file(hf_model_name)
    if hf_config is None:
        raise ValueError(f"Config file not found for model id {hf_model_name}")
    logger.info(f"Using HF model weights from {hf_model_name}")

    # Assuming 2 pipeline stages, feel free to change this as long as the
    # asserts are satisfied
    pp_degree = 1
    assert world_size % pp_degree == 0
    assert config.n_layers % pp_degree == 0

    # Sequence parallel is enabled in this program
    # Sequence parallel = Tensor parallel + dividing sequence by tp_degree at layer boundary
    sp_degree = world_size // pp_degree

    # Create device mesh
    mesh_dimensions = (pp_degree, sp_degree)
    device_mesh = _create_device_mesh(mesh_dimensions)
    tp_mesh = device_mesh["tp"]
    pp_mesh = device_mesh["pp"]
    tp_rank = tp_mesh.get_local_rank()
    pp_rank = pp_mesh.get_local_rank()

    # Assuming same number of GPUs per node
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Fill in PP configs
    config.stage_idx = pp_rank
    config.n_stages = pp_degree

    with device:
        model = Transformer(config)

    model.setup_caches(1, 4096)

    # Distribute model on TP mesh
    #model.distribute(tp_mesh)
    #logger.info(f"Model: {model}")
    # 8 samples = 2 microbatches, each mb contains 4 samples.  
    mbs = 1  # number of micro-batches 
    mb_size = 1  # micro-batch size = number of samples in a batch. 
    batch_size = mbs * mb_size  # total batch size
    seqlen = 4096  # sequence length
    dim = 4096  # embedding dimension
    assert seqlen % sp_degree == 0

    mb_ids = torch.randint(0, config.vocab_size, (mb_size, seqlen), device=device)
    activation = torch.rand(
        mb_size, seqlen // sp_degree, dim, device=device, dtype=model_dtype
    )
    example_args = mb_ids if pp_rank == 0 else activation

    # Load weights
    logger.info(f"Loading weights for {pp_rank=} on {device=}")
    _load_model_weights(model, hf_model_name, device=device, model_config=config)
    
    # Setup input position
    # input_pos for prefill: a list of increasing integers from 0 to seqlen
    input_pos = torch.arange(seqlen, device=device)
    model.setup_input_pos(input_pos)
    model.to("cuda")
    model.eval()
    '''
    logger.info(f"Creating pipeline stage {pp_rank=}, {pp_degree=}")
    stage = PipelineStage(
        model,
        pp_rank,
        pp_degree,
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

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), device=device)
    logger.info(f"Input: {input_ids.dtype=}, {input_ids.shape=}, {input_ids.device=}")
    '''
    full_batch_prompts = (
        "What is snow?",#  "I like to", "Can I help", "You need to",
        #"The weather is", "I found a", "What is your", "You are so",
    )  # full batch size = 8
    #torch.set_printoptions(threshold=30, edgeitems=10)
    inputs = tokenizer(full_batch_prompts,padding="max_length", max_length=4096, return_tensors="pt",).to(device)
    
    input_ids = inputs["input_ids"].to(device)
    #logger.info(f"{input_ids=}")
    input_ids=torch.tensor([128000, 128006, 882, 128007, 271, 12840, 374, 12056, 30, 128009, 128006, 78191, 128007, 271], device="cuda", dtype=torch.int64)
    logger.info(f"Input: {input_ids.dtype=}, {input_ids[0:10]=}")
    pad_token = tokenizer(tokenizer.pad_token)
    logger.info(f"{pad_token=}")
    padded_input = torch.full((4096,), 128009, dtype=torch.int64, device=device)
    insert_size = len(input_ids)
    padded_input[:insert_size] = input_ids
    padded_input = padded_input.unsqueeze(0)
    #logger.info(f"{padded_input.shape=}")
    #logger.info(f"{padded_input[:20]=}")
    
    output = model(padded_input)
    logger.info(f"{output=}")
    logger.info(f"Output: {output.shape=}")
    next_token_logits = output[:, -1, :]
    full_batch_logits = output[:, 0:-1, :]
    logger.info(f"{next_token_logits.shape=}")
    next_token = torch.argmax(next_token_logits, dim=-1)
    next_full_batch = torch.argmax(full_batch_logits, dim=-1)
    logger.info(f"{next_token=}, {tokenizer.batch_decode(next_token, skip_special_tokens=False)}")
    logger.info(f"{full_batch_logits=}, {(tokenizer.batch_decode(next_full_batch))}")
    '''
    schedule = ScheduleGPipe(stage, mbs)
    logger.info(f"Created schedule: {schedule}")
    output=None
    with torch.no_grad():  # .inference_mode():
        if pp_rank == 0:
            output=schedule.step(input_ids)
        else:
            output = schedule.step()

    if pp_rank == pp_degree - 1 and tp_rank == 0:
        #logger.info(f"Output: {output}")
        # Decode
    #if output:
        logger.info(f"Output: {output.shape=}")
        next_token_logits = output[:, -1, :]
        full_batch_logits = output[:, 0:-1, :]
        logger.info(f"{next_token_logits.shape=}")
        next_token = torch.argmax(next_token_logits, dim=-1)
        next_full_batch = torch.argmax(full_batch_logits, dim=-1)
        logger.info(f"{next_token=}, {tokenizer.batch_decode(next_token, skip_special_tokens=True)}")
        #logger.info(f"{full_batch_logits=}, {(tokenizer.batch_decode(next_full_batch))}")

    '''
    logger.info(
        f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}"
    )

    _cleanup()


if __name__ == "__main__":
    main()

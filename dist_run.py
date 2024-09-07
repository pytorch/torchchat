# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

# Run command:
# torchrun --nproc-per-node 4 dist_run.py
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from distributed.logging_utils import setup_logging
# TODO - these are not distributed specific, consider moving to new package
from distributed.safetensor_utils import (get_hf_config_file,
                                          get_hf_weight_map_and_path,
                                          load_safetensor_weights)
from distributed.utils import Color as color, get_stage_size, build_gpu_memory_monitor
from distributed.verification_utils import find_cpu_tensors
from torchchat.cli.builder import TokenizerArgs, _initialize_tokenizer
from torchchat.model import ModelArgs, Transformer
from torchchat.utils.build_utils import set_precision

try:
    from tokenizer.tiktoken import Tokenizer as TiktokenTokenizer
except ImportError:
    TiktokenTokenizer = None
try:
    from sentencepiece import SentencePieceProcessor
except ImportError:
    SentencePieceProcessor = None


# logger = setup_logging(__name__)
from distributed.logging_utils import SingletonLogger
logger = SingletonLogger.get_logger(__name__)


NAME_TO_HF_MODEL_ID_AND_DTYPE = {
    "Transformer-2-7b-chat-hf": ("meta-llama/Llama-2-7b-chat-hf", torch.float16),
    "Meta-Llama-3-8B": ("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16),
}
CACHE_PRECISION = torch.bfloat16


def _init_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Assuming same number of GPUs per node
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def _create_device_mesh(mesh_dimensions):
    return dist.init_device_mesh("cuda", mesh_dimensions, mesh_dim_names=("pp", "tp"))


def dict_to_args(dictionary: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dictionary)


def _build_chat_tokenizer(
    model_base_name: str = "llama3",
) -> SentencePieceProcessor | TiktokenTokenizer:
    # Create base args for tokenizer
    default_model_dir = Path(
        os.getenv("TORCHCHAT_MODELDIR", "~/.torchchat/model-cache")
    ).expanduser()

    tokenconfig = {
        "model_directory": default_model_dir,
        "model": model_base_name,
        "tokenizer_path": None,
    }
    args = dict_to_args(tokenconfig)
    tokenizer_args = TokenizerArgs.from_args(args)
    tokenizer = _initialize_tokenizer(tokenizer_args)
    assert tokenizer is not None, f"Failed to get tokenizer using {tokenconfig=}"
    logger.info(
        f"using tokenizer = {tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    )
    return tokenizer

def _encode_tokens(string, tokenizer, bos=True, device="cuda", dtype=torch.int64):
        tokens = tokenizer.encode(string)
        if bos:
            tokens = [tokenizer.bos_id()] + tokens
        logger.info(f"***** encoding:  {tokens=}, {string=}")
        return torch.tensor(tokens, dtype=dtype, device=device)

def _logits_to_probs(
        logits,
        temperature=1.0,
    ):
        logits = logits / max(
            temperature, 1e-5 if logits.dtype != torch.float16 else 1e-3
        )
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

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

def _multinomial_sample_one_no_sync(
        probs_sort,
    ):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def _cleanup():
    dist.barrier()
    dist.destroy_process_group()

def _get_hf_tokenizer(hf_model_name):
    """Load tokenizer from HF model id. note - use torchchat tokenizer as default"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    assert tokenizer is not None, f"Failed to load tokenizer for {hf_model_name}"
    logger.info(f"Loaded tokenizer for {hf_model_name}")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    rank, world_size = _init_distributed()
    gpu_memory_monitor, device_info = build_gpu_memory_monitor()
    logger.info(f"{color.yellow} {device_info}{color.reset}")


    MODEL_NAME = "Meta-Llama-3-8B" # "Transformer-2-7b-chat-hf"

    config = ModelArgs.from_name(MODEL_NAME).text_transformer_args
    logger.info(f"Chat Model Config: {config}")
    

    tokenizer = _build_chat_tokenizer()
    logger.info(f"built tokenizer {tokenizer=}")

    hf_model_name, model_dtype = NAME_TO_HF_MODEL_ID_AND_DTYPE[MODEL_NAME]
    logger.info(f"Using HF model weights from {hf_model_name} and dtype {model_dtype}")


    hf_tokenizer = _get_hf_tokenizer(hf_model_name)

    set_precision(CACHE_PRECISION)
    logger.info(f"Using cache precision {CACHE_PRECISION}")

    hf_config = get_hf_config_file(hf_model_name)
    if hf_config is None:
        raise ValueError(f"Config file not found for model id {hf_model_name}")
    logger.info(f"Using HF model weights from {hf_model_name}")

    # Assuming 2 pipeline stages, feel free to change this as long as the
    # asserts are satisfied
    pp_degree = 4
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
    model.distribute(tp_mesh)
    logger.info(f"Model: {model}")

    mbs = 1  # number of micro-batches
    mb_size = 1  # micro-batch size
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
    stage_size = get_stage_size(model)
    logger.info(f"Stage for rank {rank} is size: {color.yellow}{stage_size}{color.reset}")

    # Setup input position
    # input_pos for prefill: a list of increasing integers from 0 to seqlen
    input_pos = torch.arange(seqlen, device=device)
    model.setup_input_pos(input_pos)
    model.eval()

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

    #input_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), device=device)
    #logger.info(f"Input: {input_ids.dtype=}, {input_ids.shape=}, {input_ids.device=}")

    prompt = "what is snow?"
    input_ids = _encode_tokens(prompt, tokenizer, device, dtype=torch.int64)

    prompt_len = input_ids.size(0)
    logger.info(f"{prompt_len=}")
    start_pos = 0
    max_new_tokens = min(seqlen, seqlen - start_pos - prompt_len)
    token_buffer_size = prompt_len + max_new_tokens
    
    empty = torch.empty(token_buffer_size, dtype=torch.int64, device=device)
    empty.fill_(tokenizer.eos_id())
    empty[:prompt_len] = input_ids
    empty = empty.unsqueeze(0)
    seq = empty
    #input_pos = torch.arange(
    #        start_pos, T + start_pos, device=device, dtype=torch.int
    #    )


    schedule = ScheduleGPipe(stage, mbs)
    logger.info(f"Created schedule: {schedule}")

    with torch.no_grad():  # .inference_mode():
        if pp_rank == 0:
            schedule.step(empty)
        else:
            output = schedule.step()

# Decoding
    if pp_rank == pp_degree - 1 and tp_rank == 0:
        
        next_token_logits = output[:,prompt_len-1, :]

        logger.info(f"{next_token_logits=}")
        logger.info(f"{next_token_logits.shape=}")

        next_token = torch.argmax(next_token_logits, dim=-1)

        # self.tokenizer.decode([period_id] + x.tolist())[1:]
        next_token_decoded = tokenizer.decode((next_token.tolist()))

        logger.info(f"\n\n{color.green}====>>>> {color.blue} {next_token_decoded=}, {next_token}\n{color.reset}")
        res_mem_gib, res_mem_pct = gpu_memory_monitor.get_peak_stats()
        logger.info(f"{color.blue} Memory used: {color.green}{res_mem_pct:.3f} %, {color.magenta}{res_mem_gib:.3f} GB{color.reset}")


    logger.info(
        f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}"
    )

    _cleanup()


if __name__ == "__main__":
    main()

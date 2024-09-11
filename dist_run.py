# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

# Run command:
# torchrun --nproc-per-node 4 dist_run.py
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe


from distributed.logging_utils import SingletonLogger

# TODO - these are not distributed specific, consider moving to new package
from distributed.safetensor_utils import (
    get_hf_config_file,
    get_hf_weight_map_and_path,
    load_safetensor_weights,
)

from distributed.utils import (
    Color as color,
    GPUMemoryMonitor,
    get_module_size,
    get_num_params,
    bytes_to_readable,
    TrackTime, 
    CUDATrackTime,
)

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


logger = SingletonLogger.get_logger()

MODEL_NAME = "Transformer-2-7b-chat-hf"

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

def _encode_string(string, tokenizer, bos=True, device="cuda", dtype=torch.int64)-> torch.Tensor:
    """Encode a prompt string into a tensor of token ids."""
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=dtype, device=device)

def _create_padded_prompt(input_ids, tokenizer, seqlen, start_pos, device) -> Tuple[torch.Tensor, int]:
    """Create a padded tensor for the encoded input prompt. Returns the padded tensor and the prompt length."""
    prompt_len = input_ids.size(0)
    max_new_tokens = min(seqlen, seqlen - start_pos - prompt_len)
    token_buffer_size = prompt_len + max_new_tokens
    seq = torch.full((1, token_buffer_size), tokenizer.eos_id(), dtype=torch.int64, device=device)
    seq[0, :prompt_len] = input_ids
    return seq, prompt_len

def _cleanup():
    dist.barrier()
    dist.destroy_process_group()


def main():
    rank, world_size = _init_distributed()

    gpu_memory_monitor = GPUMemoryMonitor("cuda")
    logger.info(f"{color.yellow} {gpu_memory_monitor.get_device_info()}{color.reset}")

    config = ModelArgs.from_name(MODEL_NAME).transformer_args['text']
    logger.info(f"Chat Model Name: {MODEL_NAME}\nModel Config: {config}")

    tokenizer = _build_chat_tokenizer()
    logger.info(f"built tokenizer {tokenizer=}")

    hf_model_name, model_dtype = NAME_TO_HF_MODEL_ID_AND_DTYPE[MODEL_NAME]
    logger.info(f"Using HF model weights from {hf_model_name} and dtype {model_dtype}")

    set_precision(CACHE_PRECISION)
    logger.info(f"Using cache precision {CACHE_PRECISION}")

    hf_config = get_hf_config_file(hf_model_name)
    if hf_config is None:
        raise ValueError(f"Config file not found for model id {hf_model_name}")
    logger.info(f"Using HF model weights from {hf_model_name}")

    # Assuming 2 pipeline stages, feel free to change this as long as the
    # asserts are satisfied
    pp_degree = 2
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
    # logger.info(f"Model: {model}")

    mbs = 1  # number of micro-batches TODO: move to multibatch
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
    with TrackTime("cuda") as timer:
        _load_model_weights(model, hf_model_name, device=device, model_config=config)
    logger.info(
        f"{color.green}Total weight loading time: {timer.get_time()} {timer.unit} for stage {rank}{color.reset}"
    )

    # info on stage size and params
    stage_size = get_module_size(model)
    stage_size_formatted = bytes_to_readable(stage_size)
    stage_num_params = get_num_params(model)
    logger.info(
        f"Stage {rank} has {color.blue}{stage_num_params} params{color.reset}, Size: {color.blue}{stage_size_formatted}{color.reset}\n"
    )
    
    # Setup input position
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

    prompt = "What is snow?"
    start_pos = 0

    # encode the prompt
    input_ids = _encode_string(prompt, tokenizer, bos=True, device=device, dtype=torch.int64)
    logger.info(f"{input_ids[0:8]=}")
    
    # create a padded tensor for the input prompt
    padded_sequence, prompt_len = _create_padded_prompt(input_ids, tokenizer, seqlen, start_pos, device)
    logger.info(f"{prompt_len=}")

    schedule = ScheduleGPipe(stage, mbs)
    logger.info(f"Created schedule: {schedule}")

    with torch.no_grad():  # .inference_mode():
        if pp_rank == 0:
            schedule.step(padded_sequence)
        else:
            output = schedule.step()

    # Decoding
    if pp_rank == pp_degree - 1 and tp_rank == 0:
        next_token_logits = output[:,prompt_len-1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        next_token_decoded = tokenizer.decode((next_token.tolist()))

        logger.info(f"\n\n{color.green} Prefill response ====>>>> {color.blue} {next_token_decoded=}, {next_token}\n{color.reset}")
        

    # show peak memory stats for this stage
    res_mem_gib, res_mem_pct = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"{color.blue} Memory used: {color.green}{res_mem_pct:.3f} %, {color.magenta}{res_mem_gib:.3f} GB{color.reset}"
    )

    logger.info(
        f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}"
    )

    _cleanup()


if __name__ == "__main__":
    main()

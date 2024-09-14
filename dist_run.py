# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Run command:
# torchrun --nproc-per-node 4 dist_run.py
import torch
import torch.distributed as dist

from distributed.logging_utils import SingletonLogger

# TODO - these are not distributed specific, consider moving to new package
from distributed.safetensor_utils import (
    get_hf_config_file,
    get_hf_weight_map_and_path,
    load_safetensor_weights,
)
from distributed.utils import (
    bytes_to_readable,
    Color as color,
    CUDATrackTime,
    get_module_size,
    get_num_params,
    GPUMemoryMonitor,
    TrackTime,
)
from distributed.verification_utils import find_cpu_tensors
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torchchat.cli.builder import _initialize_tokenizer, TokenizerArgs
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

MODEL_NAME = "Meta-Llama-3-8B"

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


def _encode_strings(
    strings: List[str],
    tokenizer,
    bos: bool = True,
    device: str = "cuda",
    dtype=torch.int64,
) -> List[torch.Tensor]:
    """Encode a list of prompt strings into a list of tensor token ids."""
    encoded_list = []
    for string in strings:
        tokens = tokenizer.encode(string)
        if bos:
            tokens = [tokenizer.bos_id()] + tokens
        encoded_list.append(torch.tensor(tokens, dtype=dtype, device=device))
    return encoded_list


def _create_padded_prompts(
    input_ids_list: List[torch.Tensor],
    tokenizer,
    seqlen: int,
    start_pos: int,
    device: str,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Create a padded tensor for multiple encoded input prompts.

    Returns:
        Tuple[torch.Tensor, List[int]]: A tuple containing the padded tensor and a list of prompt lengths.
    """
    pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.eos_id()

    # Find the maximum prompt length
    max_prompt_len = max(ids.size(0) for ids in input_ids_list)

    # Calculate the buffer size
    max_new_tokens = max(0, min(seqlen - start_pos, seqlen - max_prompt_len))
    token_buffer_size = max_prompt_len + max_new_tokens

    # Create the padded batch tensor
    batch_size = len(input_ids_list)
    batch_seq = torch.full(
        (batch_size, token_buffer_size), pad_token_id, dtype=torch.int64, device=device
    )

    prompt_lengths = []
    for i, input_ids in enumerate(input_ids_list):
        prompt_len = input_ids.size(0)
        batch_seq[i, :prompt_len] = input_ids
        prompt_lengths.append(prompt_len)

    return batch_seq, prompt_lengths


def _batch_decode_next_tokens(
    output: torch.Tensor,
    prompt_lengths: List[int],
    tokenizer,
) -> List[Tuple[int, str]]:
    """
    Decode the next token for each prompt in the batch.

    Returns:
        List[Tuple[int, str]]: List of tuples containing the next token id and its
        decoded string for each prompt in the batch.
    """
    batch_size = output.shape[0]
    results = []

    for i in range(batch_size):
        next_token_logits = output[i, prompt_lengths[i] - 1, :]

        # Argmax (deterministic) TODO: add temperature
        next_token = torch.argmax(next_token_logits, dim=-1)

        next_token_decoded = tokenizer.decode([next_token.item()])
        results.append((next_token.item(), next_token_decoded))

    return results


def _cleanup():
    dist.barrier()
    dist.destroy_process_group()


def main():
    rank, world_size = _init_distributed()

    gpu_memory_monitor = GPUMemoryMonitor("cuda")
    logger.info(f"{color.yellow} {gpu_memory_monitor.get_device_info()}{color.reset}")

    config = ModelArgs.from_name(MODEL_NAME).transformer_args["text"]
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
    tp_group = tp_mesh.get_group()
    pp_group = pp_mesh.get_group()

    logger.info(f"review: {pp_group=}, {tp_group= }")

    logger.info(f"Created device mesh: {device_mesh}\n {tp_mesh=}, {pp_mesh=}\n")
    # TODO - this assumes 1D mesh, need to update for 2D+ mesh
    pp_group_size = pp_mesh.size()
    tp_group_size = tp_mesh.size()

    logger.info(f"pp_group_size: {pp_group_size}, tp_group_size: {tp_group_size}")

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
    with TrackTime() as timer:
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

    prompt = [
        "What is snow?",
    ]

    """
    "What is the capital of France?",
        "What is your name?",
        "What is the capital of Japan?",
        "When is Christmas?",
        "Where does Santa Claus live?",
        "What is the capital of the United States?",
        "What is the capital of China?",
        "What is the capital of Russia?",
        "What is PyTorch?",
        "What is the capital of India?",
        "What is an LLM?",
        "What is the capital of Brazil?",
        "What is the capital of Mexico?",
        "What is the capital of Argentina?",
        "What is the capital of Canada?",
    ]
    """

    start_pos = 0

    # encode the prompt
    input_ids = _encode_strings(
        prompt, tokenizer, bos=True, device=device, dtype=torch.int64
    )
    logger.info(f"{input_ids[0:8]=}")

    # create a padded tensor for the input prompt
    padded_sequence, prompt_lengths = _create_padded_prompts(
        input_ids, tokenizer, seqlen, start_pos, device
    )
    logger.info(f"{prompt_lengths=}")
    logger.info(f"first prompt {padded_sequence[0, :prompt_lengths[0]+1]=}")
    if len(prompt_lengths) > 1:
        logger.info(f"second prompt {padded_sequence[1, :prompt_lengths[1]+1]=}")

    schedule = ScheduleGPipe(stage, mbs)
    logger.info(f"Created schedule: {schedule}")

    # with CUDATrackTime() as timer:
    first_pp_group = 0
    last_pp_group = pp_group_size - 1

    x_recv = torch.zeros(1, device=device, dtype=torch.int64)
    logger.info(f"{x_recv.shape=}")

    last_global_rank = world_size - 1
    res = []

    # if pp_rank == pp_group_size - 1:
    #    dst = dist.get_global_rank(pp_group, 0)
    #    dist.send(tensor, dst, pp_group)

    with torch.no_grad():  # .inference_mode():
        # for _ in range(1):
        # first
        if pp_rank == 0:
            schedule.step(padded_sequence)
            src = dist.get_global_rank(pp_group, pp_group_size - 1)
            dist.recv(
                x_recv,
                src,
                group=pp_group,
            )

        # last
        elif pp_rank == last_pp_group:
            output = schedule.step()
            # need to decode the output
            decode_results = _batch_decode_next_tokens(
                output=output, prompt_lengths=prompt_lengths, tokenizer=tokenizer
            )

            logger.info(
                f"\n\n{color.green} Prefill responses ====>>>> {color.blue} {decode_results=} \n{color.reset}"
            )
            next_token = torch.tensor([decode_results[0][0]], device=device)
            res.append(decode_results[0][1])
            dst = dist.get_global_rank(pp_group, 0)
            logger.info(f"SENDING back...from {rank=} to {dst=}")
            logger.info(f"SENDING data {next_token.shape=}, {next_token=}")

            dist.send(
                next_token,
                dst,
                pp_group,
            )

        # middle pp ranks
        else:
            schedule.step()

    def _update_padded_sequence(
        padded_sequence: torch.Tensor,
        x_recv: torch.Tensor,
        res,
        prompt_lengths: List[int],
    ) -> None:
        for i in range(len(prompt_lengths)):
            prompt_lengths[i] += 1
            padded_sequence[i, prompt_lengths[i] - 1] = x_recv

        logger.info(f"REVIEW {padded_sequence[0,:15]=}")

    # logger.info(f"{color.green}Total prefill time: {timer.get_time()} {timer.unit}{color.reset}")

    # decoding loop
    # append first token to the prompt from prefill
    logger.info(f"\npre update {padded_sequence[0,0:9]=}")
    _update_padded_sequence(padded_sequence, x_recv, res, prompt_lengths)
    logger.info(f"{prompt_lengths=}, {padded_sequence[0, prompt_lengths[0]-1]=}")
    logger.info(f"\n post update {padded_sequence[0,0:9]=}")

    num_tokens = 5
    with torch.no_grad():
        for step in range(num_tokens):
            if pp_rank == 0:
                logger.info(
                    f"about to send...{prompt_lengths=}, {padded_sequence[0, :prompt_lengths[0]+1]=}"
                )
                schedule.step(padded_sequence)

                src = dist.get_global_rank(pp_group, pp_group_size - 1)
                dist.recv(
                    x_recv,
                    src,
                    group=pp_group,
                )
                logger.info(f"RECEIVED {x_recv=}")
                assert x_recv != 128006, f"next_token is header id={x_recv}"
                _update_padded_sequence(padded_sequence, x_recv, res, prompt_lengths)
                logger.info(
                    f"about to send...{prompt_lengths=}, {padded_sequence[:, prompt_lengths[0]-1]=}"
                )
                schedule.step(padded_sequence)

            elif pp_rank == last_pp_group:
                output = schedule.step()
                # need to decode the output

                decode_results = _batch_decode_next_tokens(
                    output=output, prompt_lengths=prompt_lengths, tokenizer=tokenizer
                )

                for i in range(len(prompt_lengths)):
                    prompt_lengths[i] += 1
                    logger.info(
                        f"output review {prompt_lengths[i]=}, {padded_sequence[i, prompt_lengths[i]-1]=}"
                    )

                logger.info(
                    f"\n\n{color.green} * Decode * responses ====>>>> {color.blue} {decode_results=} \n{color.reset}"
                )
                res.append(decode_results[0][1])
                next_token = torch.tensor([decode_results[0][0]], device=device)
                dst = dist.get_global_rank(pp_group, 0)
                logger.info(
                    f"SENDING back...from {rank=} to {dst=}, data {next_token.shape=}, {next_token=}"
                )
                assert next_token != 128006, f"next_token is header id={next_token}"

                if step < num_tokens - 1:
                    dist.send(
                        next_token,
                        dst,
                        pp_group,
                    )

            # middle pp ranks
            else:
                schedule.step()

    # Decoding
    """
    if pp_rank == pp_degree - 1 and tp_rank == 0:
        decode_results = _batch_decode_next_tokens(
            output=output, prompt_lengths=prompt_lengths, tokenizer=tokenizer
        )

        logger.info(
            f"\n\n{color.green} Prefill responses ====>>>> {color.blue} {decode_results=} \n{color.reset}"
        )
    """
    """
    # show peak memory stats for this stage
    res_mem_gib, res_mem_pct = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"{color.blue} Memory used: {color.green}{res_mem_pct:.3f} %, {color.magenta}{res_mem_gib:.3f} GB{color.reset}"
    )

    logger.info(
        f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}"
    )
    """
    logger.info(f"$$$$$$ {color.red}{res=}{color.reset}  $$$$$")
    logger.info(
        f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}"
    )
    _cleanup()


if __name__ == "__main__":
    main()

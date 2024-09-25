# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
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
)
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torchchat.cli.builder import _initialize_tokenizer, TokenizerArgs
from torchchat.model import ModelArgs, Transformer, TransformerArgs
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

# Using model name to identify the model to load, for example "llama2-7b-chat".
# You can change it to other values listed below.
# For details on the name-to-distribution mapping, see README.md or models.json.
NAME_TO_DISTRIBUTION_AND_DTYPE = {
    "llama2-7b-chat": ("meta-llama/Llama-2-7b-chat-hf", torch.float16),
    "llama3": ("meta-llama/Meta-Llama-3-8B-Instruct", torch.bfloat16),
}


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
    model_name: str,
    model_base_name: Optional[str] = None,
) -> SentencePieceProcessor | TiktokenTokenizer:
    """Builds a tokenizer for the given model name."""
    # Try to infer the model base name from the model name:
    # e.g. "llama2-7b-chat" -> "llama2"
    if model_base_name is None:
        model_base_name = model_name.split("-")[0]
        logger.info(
            f"Using model base name '{model_base_name}' to build tokenizer. "
            "If not found, please specify it using the `model_base_name` argument."
        )

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


def _load_model_weights(stage_module, distribution, device, model_config):
    """Load the weights from the safetensor file(s) into the model stage.
    Model config is needed b/c we permute wq and wk weights based on attn heads.
    """

    weight_map, weight_path, key_map = get_hf_weight_map_and_path(distribution)

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
    bos: bool,
    device: torch.device,
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
    device: torch.device,
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
    pos: int,
) -> torch.Tensor:
    """
    Decode the next token for each prompt in the batch.
    Args:
        output (torch.Tensor): The output tensor to decode.
        pos: the position of the `output` to decode in the sequence length dimension.

    Returns:
        Decoded token ids.
    """
    # Take the next token logits for each prompt
    next_token_logits = output[:, pos, :]
    # Argmax (deterministic) TODO: add temperature
    next_token = torch.argmax(next_token_logits, dim=-1)
    # Token ids in int tensor form
    return next_token


def _update_padded_sequence(
    padded_sequence: torch.Tensor,
    new_token: torch.Tensor,
    prompt_lengths: List[int],
) -> None:
    for i in range(len(prompt_lengths)):
        padded_sequence[i, prompt_lengths[i]] = new_token[i, 0]
        # logger.info(f"updated prompt {i} with new token {new_token[i, 0]}")


def _cleanup():
    dist.barrier()
    dist.destroy_process_group()


def main(args):
    model_name = args.model_name
    pp_degree = args.pp

    rank, world_size = _init_distributed()

    gpu_memory_monitor = GPUMemoryMonitor("cuda")
    logger.info(f"{color.yellow} {gpu_memory_monitor.get_device_info()}{color.reset}")

    distribution, model_dtype = NAME_TO_DISTRIBUTION_AND_DTYPE[model_name]
    logger.info(f"Using HF model weights from {distribution} and dtype {model_dtype}")

    # Model-level config
    model_config = ModelArgs.from_name(distribution)
    # Transformer-level config
    config = TransformerArgs.from_params(model_config.transformer_args["text"])
    logger.info(f"Transformer Config: {config}")

    tokenizer = _build_chat_tokenizer(model_name)

    set_precision(model_dtype)
    logger.info(f"Using cache precision {model_dtype}")

    hf_config = get_hf_config_file(distribution)
    if hf_config is None:
        raise ValueError(f"Config file not found for model id {distribution}")

    # Validate pipeline degree
    assert world_size % pp_degree == 0
    assert config.n_layers % pp_degree == 0

    # Tensor parallel is enabled in this program
    tp_degree = world_size // pp_degree

    # Create device mesh
    mesh_dimensions = (pp_degree, tp_degree)
    device_mesh = _create_device_mesh(mesh_dimensions)
    tp_mesh = device_mesh["tp"]
    pp_mesh = device_mesh["pp"]
    logger.info(f"Created device mesh: {device_mesh}\n{tp_mesh=}, {pp_mesh=}")

    tp_rank = tp_mesh.get_local_rank()
    pp_rank = pp_mesh.get_local_rank()
    tp_group = tp_mesh.get_group()
    pp_group = pp_mesh.get_group()
    logger.info(f"{pp_degree=}, {tp_degree=}")

    # Convenience variables
    first_pp_rank = 0
    last_pp_rank = pp_degree - 1

    # Assuming same number of GPUs per node
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Fill in PP configs
    config.stage_idx = pp_rank
    config.n_stages = pp_degree

    with device:
        # TODO: we should create model instead of Transformer
        model = Transformer(config)

    # Distribute model on TP mesh
    model.distribute(tp_mesh)
    if rank == 0:
        logger.info(f"Model: {model}")

    # Batch size. Since we push batches dynamically through the pipeline rather
    # than chunking them, this is effectively micro-batch size in pipeline
    # sense. Thus it is interchangeable with micro-batch size below.
    batch_size = 4
    seqlen_prefill = 1024  # sequence length
    dim = 4096  # embedding dimension

    # Setup KV caches (after model distribution)
    # The number of cache lanes is the same as the maximum number of
    # micro-batches that can be "in flight" in parallel -- imagine each
    # micro-batch takes 1 "pipeline lane," they need distinct KV cache spaces.
    # When decoding is done for certain micro-batches, we can reuse the KV cache
    # lanes.
    # TODO: bump up the lane count
    pipeline_lanes = 1
    model.setup_caches(batch_size, seqlen_prefill, cache_lanes=pipeline_lanes)

    # Load weights
    logger.info(f"Loading weights for {pp_rank=} on {device=}")
    with CUDATrackTime() as timer:
        _load_model_weights(model, distribution, device=device, model_config=config)
        model.to(device)

    logger.info(
        f"{color.green}Total weight loading time: {timer.get_time()} {timer.unit} for rank {rank}{color.reset}"
    )

    # info on stage size and params
    stage_size = get_module_size(model)
    stage_size_formatted = bytes_to_readable(stage_size)
    stage_num_params = get_num_params(model)
    logger.info(
        f"Stage {rank} has {color.blue}{stage_num_params} params{color.reset}, Size: {color.blue}{stage_size_formatted}{color.reset}"
    )

    # Setup input position (input_pos) for prefill: a list of increasing integers from 0 to seqlen
    input_pos = torch.arange(seqlen_prefill, device=device)
    model.eval()

    # Helper function to get example inputs and outputs for the stages.
    def get_example_ins_outs(seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mb_ids = torch.randint(0, config.vocab_size, (batch_size, seqlen), device=device)
        activation = torch.rand(
            batch_size, seqlen, dim, device=device, dtype=model_dtype
        )
        logits = torch.rand(
            batch_size, seqlen, config.vocab_size, device=device, dtype=model_dtype
        )
        example_inputs = (mb_ids if pp_rank == first_pp_rank else activation,)
        example_outputs = (logits if pp_rank == last_pp_rank else activation,)
        return example_inputs, example_outputs

    # Create prefill stage
    logger.info(f"Creating pipeline stage for prefill {pp_rank=}, {pp_degree=}")
    example_inputs, example_outputs = get_example_ins_outs(seqlen_prefill)
    prefill_stage = PipelineStage(
        model,
        pp_rank,
        pp_degree,
        device,
        input_args=example_inputs,
        output_args=example_outputs,
        group=pp_group,
    )

    # Create schedule
    # Number of micro-batches for the schedule is 1, because each step() call we
    # only push 1 micro-batch into the pipeline. But we can continuously push
    # new micro-batches into the pipeline as they arrive, achieving same
    # pipelining effect.
    prefiller = ScheduleGPipe(prefill_stage, 1)

    prompt = [
        "What is a computer?",
        "Where does Santa live?",
        "Who is Abraham Lincoln?",
        "How are models trained?",
    ]

    start_pos = 0

    # Need these global ids due to the API definition of dist.send and recv
    first_pp_rank_global_id = dist.get_global_rank(pp_group, first_pp_rank)
    last_pp_rank_global_id = dist.get_global_rank(pp_group, last_pp_rank)

    # encode the prompt
    input_ids = _encode_strings(
        prompt, tokenizer, bos=True, device=device, dtype=torch.int64
    )

    # create a padded tensor for the input prompt
    padded_sequence, prompt_lengths = _create_padded_prompts(
        input_ids, tokenizer, seqlen_prefill, start_pos, device
    )
    # TODO: figure out how to set input_pos for each prompt in the batch then we
    # can remove this limitation.
    s = set(prompt_lengths)
    assert len(s) == 1, f"prompt_lengths should be the same, got {s}"

    # Need these global ids due to the API definition of dist.send and recv
    first_pp_rank_global_id = dist.get_global_rank(pp_group, first_pp_rank)
    last_pp_rank_global_id = dist.get_global_rank(pp_group, last_pp_rank)

    # New token generated each iteration
    # need a row dimension for each prompt in the batch
    new_token = torch.zeros(batch_size, 1, device=device, dtype=torch.int64)
    # Store the generated tokens
    res = []

    # Prefill phase
    # Run context input through pipeline
    # TODO: we need to pass `input_pos` and `cache_lane` to each stage.
    lane = 0
    kwargs = {"input_pos": input_pos, "cache_lane": lane}
    with torch.no_grad(), CUDATrackTime() as timer:
        if pp_rank == first_pp_rank:
            output = prefiller.step(padded_sequence, **kwargs)
        elif pp_rank == last_pp_rank:
            output = prefiller.step(**kwargs)
        else:  # middle pp ranks
            prefiller.step(**kwargs)

    logger.info(
        f"{color.green}Prefilling time: {timer.get_time()} {timer.unit} for rank {rank}{color.reset}"
    )

    # Decode token id into string and print it
    def decode_in_flight(token):
        # Make a 2D tensor with ids on row dimension
        unsqueezed = torch.unsqueeze(token, 1)
        token_str = tokenizer.decode(unsqueezed.tolist())
        if tp_rank == 0:
            logger.info(
                f"{color.green} responses ====>>>> "
                f"{color.blue} {token_str} {color.reset}"
            )

    # Decode the output -- first generated token
    if pp_rank == last_pp_rank:
        new_token = _batch_decode_next_tokens(output, prompt_lengths[0] - 1)
        res.append(new_token)
        if not args.disable_in_flight_decode:
            decode_in_flight(new_token)

    # seqlen = 1 now
    seqlen_decode = 1
    input_pos = torch.tensor([prompt_lengths[0]], device=device)

    # Create decode stage
    logger.info(f"Creating pipeline stage for decode {pp_rank=}, {pp_degree=}")
    example_inputs, example_outputs = get_example_ins_outs(seqlen_decode)
    decode_stage = PipelineStage(
        model,
        pp_rank,
        pp_degree,
        device,
        input_args=example_inputs,
        output_args=example_outputs,
        group=pp_group,
    )
    # create schedule
    decorder = ScheduleGPipe(decode_stage, 1)

    # Decoding
    with torch.no_grad(), CUDATrackTime() as timer:
        for step in range(args.ntokens - 1):
            kwargs = {"input_pos": input_pos, "cache_lane": lane}
            # sendrecv between last and first ranks, only if:
            # first_pp_rank != last_pp_rank.
            if pp_rank == last_pp_rank and pp_rank != first_pp_rank:
                dist.send(
                    new_token,
                    dst=first_pp_rank_global_id,
                    group=pp_group,
                )
            elif pp_rank == first_pp_rank and pp_rank != last_pp_rank:
                dist.recv(
                    new_token,
                    src=last_pp_rank_global_id,
                    group=pp_group,
                )

            # Run data through pipeline
            if pp_rank == first_pp_rank:
                output = decorder.step(new_token, **kwargs)
            elif pp_rank == last_pp_rank:
                output = decorder.step(**kwargs)
            else:  # middle pp ranks
                decorder.step(**kwargs)

            # Decode the output
            if pp_rank == last_pp_rank:
                new_token = _batch_decode_next_tokens(output, 0)
                res.append(new_token)
                if not args.disable_in_flight_decode:
                    decode_in_flight(new_token)

            # Increment input position
            input_pos += 1

    logger.info(
        f"{color.green}Decoding time: {timer.get_time()} {timer.unit} for rank {rank}{color.reset}"
    )

    # Display the decoding results

    # output formatted response via last pp group and tp rank 0
    if pp_rank == last_pp_rank and tp_rank == 0:
        # `res` is a list of tensors, each being a batch of generated token ids
        res = torch.stack(res, dim=1)
        res_list = res.tolist()
        response = tokenizer.decode(res_list)
        for i in range(len(response)):
            logger.info(f"Prompt: {color.green}{prompt[i]} {color.reset}")
            logger.info(f"Response: {color.red}{response[i]} {color.reset}")

    # Cleanup
    _cleanup()
    logger.info(
        f"{color.green}Success{color.white} - {color.blue}Rank {rank} has completed.{color.reset}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to load",
        choices=NAME_TO_DISTRIBUTION_AND_DTYPE.keys(),
    )
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel degree")
    parser.add_argument(
        "--ntokens",
        type=int,
        default=40,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--disable-in-flight-decode",
        action="store_true",
        default=False,
        help="Whether to decode token into string in flight",
    )
    args = parser.parse_args()

    main(args)

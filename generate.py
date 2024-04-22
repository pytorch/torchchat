# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import itertools

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

from build.builder import (
    _initialize_model,
    _initialize_tokenizer,
    BuilderArgs,
    TokenizerArgs,
)
from build.model import Transformer
from build.utils import device_sync, set_precision
from cli import add_arguments, add_arguments_for_generate, arg_init, check_args
from download import download_and_convert, is_model_downloaded

logger = logging.getLogger(__name__)

B_INST, E_INST = "[INST]", "[/INST]"


@dataclass
class GeneratorArgs:
    prompt: str = "torchchat is pronounced torch-chat and is so cool because"
    encoded_prompt: Optional[torch.Tensor] = None
    chat_mode: bool = False
    gui_mode: bool = False
    num_samples: int = 1
    max_new_tokens: int = 200
    top_k: int = 200
    temperature: int = 0  # deterministic argmax
    compile: bool = False
    compile_prefill: bool = False
    speculate_k: int = 5
    sequential_prefill: bool = True

    def __post_init__(self):
        if self.compile_prefill and self.sequential_prefill:
            raise RuntimeError("prefill compilation requires parallel prefill")

    def validate_model(self, model: Transformer, model_description: str = "model"):
        if model is None:
            return
        reason = ""
        model_type = ""
        if not self.sequential_prefill:
            reason = "parallel prefill"
        if self.prefill_compile:
            reason = "model compilation for prefill"
        if self.compile:
            reason = "model compilation"
        if model.config.dso_path:
            model_type = "DSO"
        if model.config.pte_path:
            model_type = "PTE"
        if model_type and reason:
            raise RuntimeError(
                f"cannot perform {reason} because a {model_type} {model_description} is used"
            )

    @classmethod
    def from_args(cls, args):  # -> GeneratorArgs:
        return cls(
            prompt=args.prompt,
            encoded_prompt=None,
            chat_mode=args.chat,
            gui_mode=args.gui,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            compile=args.compile,
            compile_prefill=args.compile_prefill,
            speculate_k=args.speculate_k,
            parallel_prefill=not args.parallel_prefill,
        )


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5 if logits.dtype != torch.float16 else 1e-3)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits, need_probs: bool, temperature: float = 1.0, top_k: Optional[int] = None
):
    if temperature == 0 and not need_probs:
        _, idx_next = torch.topk(logits, k=1, dim=-1)
        idx_next = idx_next.squeeze(dim=(0, 1))
        return (idx_next, None)
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    *,
    sequential_prefill=True,
    **sampling_kwargs,
) -> torch.Tensor:
    logging.debug(f"x: {x}, input_pos: {input_pos}")
    width = x.size(1)
    assert input_pos.size(0) == width

    if sequential_prefill:
        for i in range(width):
            x_sliced, ip_sliced = x[:, i].view(-1, 1), input_pos[i].view(-1)
            logging.debug(f"<sliced> x: {x_sliced}, input_pos: {ip_sliced}")
            logits = model(x_sliced, ip_sliced)  # (x[:, i], input_pos[i])
    else:
        # input_pos: [B, S]
        logits = model(x, input_pos)

    return sample(logits, need_probs=False, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    need_probs: bool,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, need_probs=need_probs, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    need_probs: bool,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
        # Actually better for Inductor to codegen attention here
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, need_probs=need_probs, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            if need_probs:
                new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs,
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor(
        [input_pos], dtype=torch.int64, device=cur_token.device
    )
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model,
        cur_token.view(1, -1),
        orig_input_pos.clone(),
        speculate_k,
        need_probs=True,
        **sampling_kwargs,
    )

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device),
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
    rejected_locations = (
        torch.rand_like(accept_draft_prob) > accept_draft_prob
    ).nonzero()

    if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    chat_mode: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    sequential_prefill=True,
    callback=lambda x: x,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if chat_mode:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = (
        max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    )
    model = model.to(device=device)
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device, dtype=torch.int)

    next_token = prefill(
        model,
        prompt.view(1, -1),
        input_pos,
        sequential_prefill=sequential_prefill,
        **sampling_kwargs,
    )
    if is_speculative:
        prefill(
            draft_model,
            prompt.view(1, -1),
            input_pos,
            sequential_prefill=sequential_prefill,
            **sampling_kwargs,
        )
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[:num_added]
            for i in next_tokens[:num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(
            model,
            next_token.view(1, -1),
            input_pos,
            max_new_tokens - 1,
            callback=callback,
            need_probs=False,
            **sampling_kwargs,
        )
        seq[T + 1 :] = torch.cat(generated_tokens)

    generate_stats = {"accept_counts": accept_counts}
    return seq, generate_stats


def encode_tokens(tokenizer, string, bos=True, device="cpu"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


B_INST, E_INST = "[INST]", "[/INST]"


def get_device_info(name: str) -> str:
    import platform
    from subprocess import check_output

    if name == "cpu":
        if platform.system() == "Darwin":
            return (
                check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .decode("utf-8")
                .strip()
            )
        if platform.system() == "Linux":
            return (
                check_output(
                    ["sed", "-nr", "s/^model name\\s+: (.*)$/\\1/p", "/proc/cpuinfo"]
                )
                .decode("utf-8")
                .split("\n")[0]
            )
    if name == "cuda":
        return torch.cuda.get_device_name(0)
    return ""


def _main(
    builder_args: BuilderArgs,
    speculative_builder_args: BuilderArgs,
    tokenizer_args: TokenizerArgs,
    generator_args: GeneratorArgs,
    profile: Optional[Path] = None,
    quantize=None,
    draft_quantize=None,
) -> None:
    """
    Generates text samples based on a pre-trained Transformer model and tokenizer.
    """

    # global print
    #    from tp import maybe_init_dist
    #    rank = maybe_init_dist()
    use_tp = False
    rank: Optional[int] = None
    #    if use_tp:
    #        if rank != 0:
    #            # only print on rank 0
    #            print = lambda *args, **kwargs: None

    print(f"Using device={builder_args.device} {get_device_info(builder_args.device)}")
    set_precision(builder_args.precision)
    is_speculative = speculative_builder_args.checkpoint_path is not None

    if generator_args.chat_mode and not builder_args.is_chat_model:
        # This is not a log message, it's a dangerous condition message
        # that we must ensure is displayed
        print(
            """
*******************************************************
 This model is not known to support the chat function.
 We will enable chat mode based on your instructions.
 If the model is not trained to support chat, it will
 produce nonsensical or false output.
*******************************************************
        """
        )
        # raise RuntimeError("You need to use --is-chat-model to indicate model has chat support.")

    tokenizer = _initialize_tokenizer(tokenizer_args)

    builder_args.setup_caches = False
    model = _initialize_model(builder_args, quantize, tokenizer)

    # will add a version of _initialize_model in future
    # (need additional args)
    if is_speculative:
        draft_model = _initialize_model(
            speculative_builder_args,
            quantize if draft_quantize == "quantize" else draft_quantize,
            tokenizer,
        )
    else:
        draft_model = None

    tokenizer_args.validate(model)
    tokenizer_args.validate(draft_model, "draft model")
    generator_args.validate(model)
    generator_args.validate(draft_model, "draft model")

    encoded = encode_tokens(
        tokenizer, generator_args.prompt, bos=True, device=builder_args.device
    )
    logging.debug(encoded)
    prompt_length = encoded.size(0)

    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )
    if generator_args.compile:
        if (
            is_speculative and builder_args.use_tp
        ):  # and ("cuda" in builder_args.device):
            torch._inductor.config.triton.cudagraph_trees = (
                False  # Bug with cudagraph trees in this case
            )

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(
                model_forward, mode="reduce-overhead", fullgraph=True
            )

        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

        # Uncomment to squeeze more perf out of prefill
        if generator_args.compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
        "accept_counts": [],
    }
    start = -1 if generator_args.compile else 0

    for i in range(start, generator_args.num_samples):
        device_sync(device=builder_args.device)
        if i >= 0 and generator_args.chat_mode:
            prompt = input("What is your prompt? \n")
            if builder_args.is_chat_model:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(
                tokenizer, prompt, bos=True, device=builder_args.device
            )

        if generator_args.chat_mode and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(
                x, buffer=buffer, period_id=period_id, done_generating=done_generating
            ):
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end='', flush=True)

        else:

            def callback(x):
                return x

        t0 = time.perf_counter()
        import contextlib

        if (i != generator_args.num_samples - 1 or not profile) or (
            use_tp and rank != 0
        ):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                generator_args.max_new_tokens,
                draft_model=draft_model,
                speculate_k=generator_args.speculate_k,
                chat_mode=generator_args.chat_mode,
                callback=callback,
                temperature=generator_args.temperature,
                top_k=generator_args.top_k,
                sequential_prefill=generator_args.sequential_prefill,
            )
            aggregate_metrics["accept_counts"].append(metrics["accept_counts"])
        if i == -1:
            logging.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=builder_args.device)
        t = time.perf_counter() - t0

        if not generator_args.chat_mode:
            print(tokenizer.decode(y.tolist()))
        else:
            print()
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        logging.info(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logging.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        logging.info(f"Acceptance probs: {acceptance_probs}")
        logging.info(
            f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
        )

    logging.info(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
    )
    logging.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


def main(args):
    # If a named model was provided and not downloaded, download it.
    if args.model and not is_model_downloaded(args.model, args.model_directory):
        download_and_convert(args.model, args.model_directory, args.hf_token)

    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)

    _main(
        builder_args,
        speculative_builder_args,
        tokenizer_args,
        generator_args,
        args.compile,
        args.compile_prefill,
        args.profile,
        args.quantize,
        args.draft_quantize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torchchat generate CLI")
    add_arguments(parser)
    add_arguments_for_generate(parser)
    args = parser.parse_args()
    check_args(args, "generate")
    args = arg_init(args)
    main(args)

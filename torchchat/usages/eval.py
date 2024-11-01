# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from typing import Callable, Optional

import torch
import torch._dynamo.config
import torch._inductor.config

from torchchat.cli.builder import (
    _initialize_model,
    _initialize_tokenizer,
    BuilderArgs,
    TokenizerArgs,
)
from torchchat.cli.cli import add_arguments_for_verb, arg_init

from torchchat.model import Model
from torchchat.utils.build_utils import set_precision
from torchchat.utils.measure_time import measure_time

torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.triton.cudagraphs = True
torch._dynamo.config.cache_size_limit = 100000

import lm_eval

from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict


def setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
    model: Model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: Optional[int] = None,
):
    """
    Sets up model cache and does some bookkeeping calculations for prompt, input_pos and max_seq_length
    that are needed for prefill or model_forward

    Args:
        model (LLaMA): The model whose cache gets set up
        prompt (torch.Tensor): Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens (int): The desired maximum number of new tokens that can be generated.
        max_seq_length (Optional[int], optional): The maximum sequence length allowed.

    Returns:
        seq (torch.Tensor): prompt but padded with zeros to size max_seq_length
        input_pos (torch.Tensor): tensor of integers in increasing order
        max_seq_length (int): The maximum sequence length allowed, updated based on other numbers
    """
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.text_transformer_args.block_size)

    device, dtype = prompt.device, prompt.dtype
    # create an empty tensor of the expected final shape and
    # fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    return seq, input_pos, max_seq_length


class GPTFastEvalWrapper(eval_wrapper):
    """
    A wrapper class for GPTFast, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: Model,
        tokenizer,
        model_forward: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        device="cpu",
        is_pte_model: bool = False,
    ):
        super().__init__(device=device)
        self._model = model
        self._model_forward = (
            model_forward
            if model_forward is not None
            else lambda x, input_pos: model(x, input_pos)
        )
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length
        self.times = []
        self.is_pte_model = is_pte_model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        bos_id = self._tokenizer.bos_id()
        encoded = [bos_id] + self._tokenizer.encode(string)
        return encoded

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        # TODO: make batches work
        inps = inps.squeeze(0)

        max_new_tokens = 1
        seq, input_pos, max_seq_length = (
            setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
                self._model,
                inps,
                max_new_tokens,
                self.max_length,
            )
        )
        x = seq.index_select(0, input_pos).view(1, -1)
        with measure_time(message=None) as measure:
            if (
                self.is_pte_model
            ):  # Sequential Prefill required for ExecuTorch (.pte) models since the prompt length can introduce dynamism
                width = x.size(1)
                assert input_pos.size(0) == width
                logits = torch.zeros(1, width, self._model.config.vocab_size).to(
                    x.device
                )
                for i in range(width):
                    x_sliced, ip_sliced = x[:, i].view(1, -1), input_pos[i].view(-1)
                    logits[0, i] = self._model_forward(
                        x_sliced, ip_sliced
                    )  # (x[:, i], input_pos[i])
            else:
                logits = self._model_forward(x, input_pos)
        self.times.append(measure.get_time())
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


@torch.no_grad()
def eval(
    model: Model,
    model_forward: Callable,
    tokenizer,
    tasks: Optional[list] = None,
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
    device: str = "cpu",
    is_pte_model: bool = False,
) -> dict:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library.

    Args:
        model (Model): The pre-trained language model to evaluate.
        tokenizer: The tokenizer to use for encoding/decoding text.
        tasks (Optional[list]): The names of the evaluation tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    Returns:
        eval_results (dict): A dictionary of evaluation results for the specified task(s).
    """
    if tasks is None:
        tasks = ["wikitext"]

    model_eval_wrapper = GPTFastEvalWrapper(
        model,
        tokenizer,
        model_forward=model_forward,
        max_seq_length=max_seq_length,
        device=device,
        is_pte_model=is_pte_model,
    )

    try:
        lm_eval.tasks.initialize_tasks()
    except:
        pass

    if "hendrycks_test" in tasks:
        tasks.remove("hendrycks_test")
        tasks += list(lm_eval.tasks.hendrycks_test.create_all_tasks().keys())
    task_dict = get_task_dict(tasks)

    eval_results = evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit,
    )
    eval_results["times"] = model_eval_wrapper.times
    return eval_results


def main(args) -> None:
    """Evaluates model on a task from the `lm-evaluation-harness` library.

    Args:
        checkpoint_path (Path): The path to the model checkpoint file to load.
        compile (bool): Whether or not to compile the model for optimization.
        tasks (Optional[list]): The names of the evaluation tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    """

    builder_args = BuilderArgs.from_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    quantize = args.quantize
    device = args.device
    tasks = args.tasks
    limit = args.limit
    compile = args.compile
    max_seq_length = args.max_seq_length

    print(f"Using device={device}")
    set_precision(builder_args.precision)

    tokenizer = _initialize_tokenizer(tokenizer_args)
    builder_args.setup_caches = False
    model = _initialize_model(
        builder_args,
        quantize,
        tokenizer,
    )
    tokenizer_args.validate_model(model)

    model_forward = lambda x, input_pos: model(x, input_pos)  # noqa

    if compile:
        assert not (
            builder_args.dso_path or builder_args.pte_path or builder_args.aoti_package_path
        ), "cannot compile exported model"
        model_forward = torch.compile(
            model_forward, mode="reduce-overhead", dynamic=True, fullgraph=True
        )
        torch._inductor.config.coordinate_descent_tuning = False if device == "cpu" else True

    with measure_time("Time to run eval: {time:.02f}s."):
        result = eval(
            model.to(device),
            model_forward,
            tokenizer,
            tasks,
            limit,
            max_seq_length,
            device=builder_args.device,
            is_pte_model=builder_args.pte_path is not None,
        )

    times = torch.tensor(result["times"])
    print(
        f"Time in model.forward: {times.sum():.02f}s, over {times.numel()} model evaluations"
    )
    print(
        f"forward run time stats - Median: {times.median():.02f}s Min: {times.min():.02f}s Max: {times.max():.02f}s"
    )
    if builder_args.dso_path:
        print(f"For model {builder_args.dso_path}")
    elif builder_args.aoti_package_path:
        print(f"For model {builder_args.aoti_package_path}")
    elif builder_args.pte_path:
        print(f"For model {builder_args.pte_path}")
    elif builder_args.checkpoint_path:
        print(f"For model {builder_args.checkpoint_path}")
    elif builder_args.checkpoint_dir:
        print(f"For model {builder_args.checkpoint_dir}")
    else:
        raise RuntimeError("Well That's Fine. How did we get here")

    for task, res in result["results"].items():
        print(f"{task}:")
        for metric, val in res.items():
            if val != "N/A":
                print(f" {metric}: {val if isinstance(val, str) else f'{val:0.4f}'}")

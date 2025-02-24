# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from typing import Callable, Dict, List, Optional

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

import PIL

from lm_eval.evaluator import evaluate
from lm_eval.models.hf_vlms import HFMultimodalLM
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict
from torchtune import utils
from torchtune.data import (
    format_content_with_images,
    left_pad_sequence,
    Message,
    padded_collate_tiled_images_and_mask,
)
from torchtune.generation import generate, sample

from torchtune.modules.common_utils import local_kv_cache
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.modules.transforms import Transform


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
        super().__init__(pretrained="gpt2", device=device)
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


class VLMEvalWrapper(HFMultimodalLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.

    Note:
        This is ONLY for vision-language models.

    Args:
        model (DeepFusionModel): The VLM to evaluate.
        transform (Transform): The transform (tokenizer) to use for preprocessing.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length.
        batch_size (int): The batch size.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
        image_tag (str): The string to use for the image token. Default is "<image>", which
            is the default used by the MMMU dataset.
        max_images_per_sample (int): The maximum number of images per sample. Defaults to
            the max number of images in MMMU.
    """

    def __init__(
        self,
        model: DeepFusionModel,
        transform: Transform,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        enable_kv_cache: bool = True,
        # TODO (@joecummings): Update these defaults once more multimodal
        # tasks are added to the eval harness
        image_tag: str = "<image>",
        max_images_per_sample: int = 7,
    ):
        self._model = model
        self._transform = transform
        self._device = device
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        # Defaulting KV cache to True for multimodal
        self._enable_kv_cache = True
        self._image_tag = image_tag
        self._max_images_per_sample = max_images_per_sample
        self.times = []

    @property
    def model(self):
        # Not actually changing the dtype here, just adding it as a
        # property on the model
        self._model.dtype = self._dtype
        return self._model

    @property
    def model_transform(self):
        return self._transform

    @property
    def device(self):
        return self._device

    @property
    def cache_hook(self):
        # Dummy class to appease the Harness
        class DummyCacheHook:
            def __init__(self):
                self.add_partial = lambda x, y, z: True

        return DummyCacheHook()

    @property
    def rank(self):
        # Hardcoded for now b/c we only support single GPU eval
        return 0

    @property
    def world_size(self):
        # Hardcoded for now b/c we only support single GPU eval
        return 1

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def eos_token_id(self):
        return self._transform.tokenizer.eos_id

    @property
    def eot_token_id(self):
        return self._transform.tokenizer.eot_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def truncation(self):
        return True

    def tok_encode(self, string, **kwargs) -> List[int]:
        # This is only used to get a number of tokens for use in sorting samples in dataset
        # These values will not actually be used for eval
        return self._transform.tokenizer.encode(string, add_bos=False, add_eos=False)

    def tok_decode(self, tokens, skip_special_tokens=True) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._transform.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens
        )

    def tok_batch_multimodal_encode(
        self,
        all_texts: List[str],
        all_images: List[List[PIL.Image.Image]],
        left_truncate_len: int = None,
        *args,
        **kwargs,
    ):
        # Eleuther already parses out the text and images, so we just need to get
        # it into a Message format for our tokenizer
        all_encoded_messages = []

        for text, images in zip(all_texts, all_images):
            # Ensure images are all RGB
            proper_images = []
            for image in images:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                proper_images.append(image)

            # Construct the messages
            messages = []
            content = format_content_with_images(
                text, image_tag=self._image_tag, images=proper_images
            )
            messages.append(Message(role="user", content=content))
            messages.append(Message(role="assistant", content=""))

            # Transform the messages
            tok_batch = self.model_transform({"messages": messages}, inference=True)
            all_encoded_messages.append(tok_batch)

        # Pad the encoded messages
        tok_batch = padded_collate_tiled_images_and_mask(
            all_encoded_messages,
            pad_direction="left",
            pad_max_images=self._max_images_per_sample,
            pad_max_tiles=self._transform.max_num_tiles,
        )
        utils.batch_to_device(tok_batch, self.device)

        # Convert the batch to the format expected by the HF
        tok_batch["input_ids"] = tok_batch.pop("tokens")

        # the harness will use left_truncate_len to indicate that the current batch
        # needs to be truncated to self.max_seq_len - self.max_gen_toks
        if left_truncate_len is not None:
            tok_batch["input_ids"] = tok_batch["input_ids"][:, -left_truncate_len:]

        return tok_batch

    @torch.inference_mode()
    def _model_multimodal_generate(
        self,
        batch: Dict[str, torch.Tensor],
        max_length: int,
        stop: List[str],
        **generation_kwargs,
    ):
        # 1. Validate inputs
        prompt = batch.pop("input_ids")
        bsz, seq_len = prompt.shape

        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )

        if bsz > 1:
            raise ValueError(
                f"Got a batch size of '{bsz}'. Batch size > 1 is not yet supported for "
                "multimodal generation."
            )

        encoder_max_seq_len = (
            self.model_transform.image_seq_len * self._max_images_per_sample
        )
        # Setup masks for bsz 1
        with self.device:
            causal_mask = torch.tril(
                torch.ones(
                    size=(self.max_length, self.max_length),
                    dtype=torch.bool,
                )
            )
            input_pos = torch.arange(self.max_length)

        batch["input_pos"] = input_pos[None, :seq_len]
        batch["mask"] = causal_mask[None, :seq_len]

        with measure_time(message=None) as measure:
            # 2. Setup KV cache
            with local_kv_cache(
                self.model,
                batch_size=self.batch_size,
                device=self.device,
                dtype=self._dtype,
                encoder_max_seq_len=encoder_max_seq_len,
                decoder_max_seq_len=self.max_length,
            ):
                # 3. Prefill step
                generated_tokens = []
                logits = self.model(prompt, **batch)[:, -1]
                token = sample(logits, temperature=0.0, top_k=None)
                generated_tokens.append(token.item())

                cache_mask = batch["encoder_mask"][:, -1:]

                # 4. Continue generating
                for _ in range(max_length):
                    if token.item() in self.model_transform.stop_tokens:
                        break
                    logits = self.model(
                        token,
                        mask=causal_mask[None, seq_len, None, :],
                        encoder_input=None,
                        encoder_mask=cache_mask,
                        input_pos=input_pos[None, seq_len],
                    )[:, -1]
                    token = sample(logits, temperature=0.0, top_k=None)
                    generated_tokens.append(token.item())
                    seq_len += 1
        self.times.append(measure.get_time())

        # 5. Return generated tokens
        return torch.tensor(generated_tokens, dtype=torch.int32).unsqueeze(0)


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


def multi_model_eval(
    model: Model,
    model_forward: Callable,
    tokenizer,
    tasks: Optional[list] = None,
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
    device: str = "cpu",
    is_pte_model: bool = False,
):
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
    max_seq_length = 4096 if max_seq_length is None else max_seq_length
    device = utils.get_device(device) if isinstance(device, str) else device

    model_eval_wrapper = VLMEvalWrapper(
        model,
        transform=tokenizer,  # tranform is the tokenizer for multimodal models
        max_seq_length=max_seq_length,
        device=device,
    )

    try:
        lm_eval.tasks.initialize_tasks()
    except:
        pass

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

    modality = builder_args.modality
    print(f"Modality of model={modality}")
    assert modality in [
        "text",
        "text-image",
    ], "Only text and text-image modality is supported for evaluation"

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
            builder_args.dso_path
            or builder_args.pte_path
            or builder_args.aoti_package_path
        ), "cannot compile exported model"
        model_forward = torch.compile(
            model_forward, mode="reduce-overhead", dynamic=True, fullgraph=True
        )
        torch._inductor.config.coordinate_descent_tuning = (
            False if device == "cpu" else True
        )

    with measure_time("Time to run eval: {time:.02f}s."):
        if modality == "text":
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
        elif modality == "text-image":
            result = multi_model_eval(
                model.to(device),
                model_forward,
                tokenizer,
                tasks,
                limit,
                max_seq_length,
                device=builder_args.device,
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

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

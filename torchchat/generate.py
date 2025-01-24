# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import base64
import contextlib
import itertools
import logging
import os
import textwrap
import time
from concurrent import futures
from functools import partial

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch._C import _SDPBackend as SDPBackend

from PIL import Image

# torchtune model definition dependencies
from torchtune.data import Message, padded_collate_tiled_images_and_mask

from torchtune.generation import sample as tune_sample

from torchtune.models.llama3_2_vision._model_builders import llama3_2_vision_transform
from torchtune.training import set_default_dtype

from torchchat.cli.builder import (
    _initialize_model,
    _initialize_tokenizer,
    BuilderArgs,
    TokenizerArgs,
)
from torchchat.distributed.utils import (
    Color as color,
    run_in_dist_env,
)
from torchchat.model import Model, ModelType
from torchchat.utils.build_utils import device_sync, set_precision
from torchchat.utils.device_info import get_device_info

logger = logging.getLogger(__name__)


# NOTE: Logging disabled by default here due to conflicts with torch._dynamo
class NoOpLogger:
    def __no_op(self, *_, **__):
        pass
    def __getattr__(self, name):
        return self.__no_op


logger = (
    NoOpLogger() if os.getenv("LOG_LEVEL") is None
    else logging.getLogger(__name__)
)

## Chat Formatters #############################################################

class _ChatFormatter(ABC):

    # Messages can arrive as a standard dict with "role" and "content" as
    # strings, or where "content" is a list of objects with "text" fields.
    MESSAGE_TYPE = Dict[str, Union[str, List[Dict[str, str]]]]

    # A dialog is a sequence of messages
    DIALOG_TYPE = List[MESSAGE_TYPE]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def encode_dialog_prompt(
        self,
        dialog: DIALOG_TYPE,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Encode a sequence of messages into a sequence of token IDs, including
        the chat template

        Args:
            dialog (DIALOG_TYPE): The sequence of dialog messages to encode.
                This will be the additional messages on top of those that have
                already been processed.
            add_generation_prompt (bool): Whether to include a generation prompt
                at the end of the encoded sequence.

        Returns:
            List[int]: A list of token IDs representing the encoded prompt.
        """


class Llama3ChatFormatter(_ChatFormatter):
    """Format a chat prompt using special tokens to demarcate roles and messages.

    Refer to the LLaMA3 documentation for more details https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    """

    def _encode_header(self, role) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def _encode_message(self, message: _ChatFormatter.MESSAGE_TYPE) -> List[int]:
        tokens = self._encode_header(message["role"])
        if isinstance(message["content"], str):
            tokens.extend(
                self.tokenizer.encode(message["content"], bos=False, eos=False)
            )
        elif isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "text":
                    tokens.extend(
                        self.tokenizer.encode(content["text"], bos=False, eos=False)
                    )

        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(
        self,
        dialog: _ChatFormatter.DIALOG_TYPE,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self._encode_message(message))
        # Add the start of an assistant message for the model to complete.
        if add_generation_prompt and dialog and dialog[-1]["role"] != "assistant":
            tokens.extend(self._encode_header("assistant")) # Pass role directly as a string
        return tokens


class Llama2ChatFormatter(_ChatFormatter):
    """
    Chat formatting for Llama2
    CITE: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/
    """

    B_INST, E_INST = "[INST] ", " [/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    @staticmethod
    def _get_content_str(message: _ChatFormatter.MESSAGE_TYPE) -> str:
        if isinstance(message["content"], list):
            return message["content"][0]["text"]
        return message["content"]

    def encode_dialog_prompt(
        self,
        dialog: _ChatFormatter.DIALOG_TYPE,
        add_generation_prompt: bool = True, # UNUSED
    ) -> List[int]:
        new_turn = True
        tokens = []
        for message in dialog:
            if new_turn:
                tokens += self.tokenizer.encode(f"{self.tokenizer.bos}{self.B_INST}")
            content = self._get_content_str(message).strip()
            role = message["role"]
            if role == "system":
                tokens += self.tokenizer.encode(f"{self.B_SYS}{content}{self.E_SYS}")
                new_turn = False
            elif role == "user":
                tokens += self.tokenizer.encode(f"{content}{self.E_INST}")
                new_turn = False
            elif role == "assistant":
                tokens += self.tokenizer.encode(f" {content} {self.tokenizer.eos}\n")
                new_turn = True
            else:
                raise ValueError("Invalid role in dialog.")
        return tokens



class HFTokenizerChatFormatter(_ChatFormatter):
    """Chat formatter that uses the built-in formatting capabilities of an HF
    tokenizer instance
    """
    def encode_dialog_prompt(
        self,
        dialog: _ChatFormatter.DIALOG_TYPE,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        rendered = self.tokenizer.apply_chat_template(
            dialog, add_generation_prompt=add_generation_prompt
        )
        logger.debug("Formatted chat prompt:\n%s", rendered)
        return self.tokenizer.encode(rendered)

## Generation ##################################################################

@dataclass
class GeneratorArgs:
    prompt: Optional[str] = (
        None  # When passed into the Generator, this will be used as the system prompt
    )
    encoded_prompt: Optional[torch.Tensor] = None
    image_prompts: Optional[Sequence[Union[str, PathLike, bytes]]] = (
        None  # string or Path to an image file or the raw base64 bytes of an image
    )
    chat_mode: bool = False
    gui_mode: bool = False
    num_samples: int = 1
    max_new_tokens: int = 200
    top_k: int = 200
    temperature: float = 0.0  # deterministic argmax if 0.0
    compile: bool = False
    compile_prefill: bool = False
    speculate_k: int = 5
    sequential_prefill: bool = False
    max_autotune: bool = False
    # (Misnomer) See Issue: https://github.com/pytorch/torchchat/issues/1273
    is_torchtune_model: bool = False

    def __post_init__(self):
        if self.compile_prefill and self.sequential_prefill:
            raise RuntimeError("prefill compilation requires parallel prefill")

    def validate_build(
        self, builder_args: BuilderArgs, model_description: str = "model"
    ):
        reason = ""
        model_type = ""
        if not self.sequential_prefill:
            reason = "parallel prefill"
        if self.compile_prefill:
            reason = "model compilation for prefill"
        if self.compile:
            reason = "model compilation"
        if builder_args.aoti_package_path:
            model_type = "PT2"
        if builder_args.dso_path:
            model_type = "DSO"
        if builder_args.pte_path:
            model_type = "PTE"
        if model_type and reason:
            raise RuntimeError(
                f"cannot perform {reason} because a {model_type} {model_description} is used"
            )

    @classmethod
    def from_args(cls, args):
        dso_path = getattr(args, "dso_path", None)
        pte_path = getattr(args, "pte_path", None)
        aoti_package_path = getattr(args, "aoti_package_path", None)
        sequential_prefill = (
            args.sequential_prefill or bool(aoti_package_path) or bool(pte_path) or bool(dso_path)
        )

        # Validate that all image prompts exist before expensive model load
        if image_prompts := getattr(args, "image_prompts", None):
            non_existent_image_prompts = [
                image_prompt
                for image_prompt in image_prompts
                if (not os.path.exists(image_prompt))
            ]
            if non_existent_image_prompts:
                raise RuntimeError(
                    f"Image prompt {non_existent_image_prompts} does not exist"
                )

        return cls(
            prompt=getattr(args, "prompt", ""),
            encoded_prompt=None,
            image_prompts=image_prompts,
            chat_mode=args.chat,
            gui_mode=args.gui,
            num_samples=getattr(args, "num_samples", 1),
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            compile=args.compile,
            compile_prefill=args.compile_prefill,
            speculate_k=args.speculate_k,
            sequential_prefill=sequential_prefill,
            max_autotune=args.max_autotune,
            is_torchtune_model=args.model and args.model.endswith("tune"),
        )


class LocalGenerator:
    """
    Generates text samples based on a pre-trained Transformer model and tokenizer.
    Args:
        builder_args: Defines the model configuration
        speculative_builder_args: Defines the speculative model configuration for speculative decode
        tokenizer_args: Defines the tokenizer configuration for both the model and speculative model
        generator_args: Controls the generation parameters
        profile: A Path to a directory where the profiling results will be stored, if enabled.
        quantize: If True, quantize the model. Please refer to docs/quantization.md for details.
        draft_quantize: If True, quantize the draft model.
    """

    def __init__(
        self,
        builder_args: BuilderArgs,
        speculative_builder_args: BuilderArgs,
        tokenizer_args: TokenizerArgs,
        generator_args: GeneratorArgs,
        profile: Optional[Path],
        quantize: bool,
        draft_quantize: bool,
    ):
        torch._inductor.config.coordinate_descent_tuning = (
            builder_args.device != "cpu"
        )
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

        self.builder_args = builder_args
        self.speculative_builder_args = speculative_builder_args
        self.tokenizer_args = tokenizer_args
        self.profile = profile
        self.quantize = quantize
        self.draft_quantize = draft_quantize
        self.is_torchtune_model = generator_args.is_torchtune_model
        self.dtype = builder_args.precision
        self.get_user_input : Callable = input

        self.rank: Optional[int] = None

        print(
            f"Using device={self.builder_args.device} {get_device_info(self.builder_args.device)}"
        )
        set_precision(self.builder_args.precision)

        self.is_speculative = self.speculative_builder_args.checkpoint_path is not None

        if generator_args.chat_mode and not self.builder_args.is_chat_model:
            # fmt: off
            print(textwrap.dedent(
                """
                *******************************************************
                This model is not known to support the chat function
                and may produce nonsensical or false output.
                *******************************************************
                """
            ))
            # fmt: on
        self.system_prompt = generator_args.prompt
        self.tokenizer = _initialize_tokenizer(self.tokenizer_args)

        # Right now the assumption is only llama3 uses tiktokenizer and it
        # must use tiktokenizer.
        # Piggy backing off of this flag then for now to identify llama3
        # without prompting user.
        self.is_llama3_model = self.tokenizer_args.is_tiktoken
        if self.is_llama3_model:
            self.chat_formatter = Llama3ChatFormatter(self.tokenizer)
            if generator_args.chat_mode:
                logger.debug(
                    "Llama3 model detected in chat mode. Using updated sentence schemas"
                )
        elif self.tokenizer_args.is_hf_tokenizer:
            if not self.tokenizer.has_chat_template():
                raise ValueError("Tokenizer must have a chat template")
            self.chat_formatter = HFTokenizerChatFormatter(self.tokenizer)
        else:
            self.chat_formatter = Llama2ChatFormatter(self.tokenizer)

        self.builder_args.setup_caches = False
        self.model = _initialize_model(self.builder_args, self.quantize, self.tokenizer)

        if self.is_speculative:
            self.draft_model = _initialize_model(
                self.speculative_builder_args,
                (
                    self.quantize
                    if self.draft_quantize == "quantize"
                    else self.draft_quantize
                ),
                self.tokenizer,
            )
        else:
            self.draft_model = None

        # torchtune model does not contain essential info for validation
        # TODO: refactor model config to be more generic
        if not self.is_torchtune_model:
            self.tokenizer_args.validate_model(self.model)
        self.tokenizer_args.validate_model(self.draft_model, "draft model")
        generator_args.validate_build(self.builder_args)
        generator_args.validate_build(self.speculative_builder_args, "draft model")

    def multinomial_sample_one_no_sync(
        self,
        probs_sort,
    ):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def logits_to_probs(
        self, logits, temperature: float = 1.0, top_k: Optional[int] = None
    ):
        logits = logits / max(
            temperature, 1e-5 if logits.dtype != torch.float16 else 1e-3
        )

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def sample(
        self,
        logits,
        need_probs: bool,
        temperature: float = 0,
        top_k: Optional[int] = None,
    ):
        logits = logits[0, -1]
        logger.debug("Logits: %s", logits)
        if temperature == 0 and not need_probs:
            _, idx_next = torch.topk(logits, k=1, dim=-1)
            return (idx_next, None)
        probs = self.logits_to_probs(logits, temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def prefill(
        self,
        model: Model,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        *,
        sequential_prefill=True,
        **sampling_kwargs,
    ) -> torch.Tensor:
        logger.debug("x: %s, input_pos: %s", x, input_pos)
        width = x.size(1)
        assert input_pos.size(0) == width

        if self.model.config.model_type == ModelType.Flamingo:
            assert batch is not None, "Flamingo requires batch"

            # TODO: Verify sequential prefill works with multimodal models
            is_multimodal = True
            if "encoder_input" in batch:
                encoder_input = batch["encoder_input"]
                encoder_mask = batch["encoder_mask"]
                is_multimodal = True
            else:
                encoder_input = None
                encoder_mask = None
                is_multimodal = False

            seq_len = x.size(1)
            mask = batch["causal_mask"][None, :seq_len]
            input_pos = input_pos.view(1, -1)
            logits = model(
                tokens=x,
                mask=mask,
                encoder_input=encoder_input,
                input_pos=input_pos,
                encoder_mask=encoder_mask,
            )[:, -1]

            if is_multimodal:
                batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

            return tune_sample(logits, temperature=0, top_k=500)
        elif sequential_prefill:
            for i in range(width):
                x_sliced, ip_sliced = x[:, i].view(-1, 1), input_pos[i].view(-1)
                logger.debug("<sliced> x: %s, input_pos: %s", x_sliced, ip_sliced)
                logits = model(x_sliced, ip_sliced)  # (x[:, i], input_pos[i])da
        else:
            # input_pos: [B, S]
            logits = model(x, input_pos)

        return self.sample(logits, need_probs=False, **sampling_kwargs)[0]

    def decode_one_token(
        self,
        model: Model,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        need_probs: bool,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        **sampling_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        x = x.view(1, -1)
        if model.config.model_type == ModelType.Flamingo:
            assert batch is not None, "Flamingo requires batch"
            mask = batch["causal_mask"][None, input_pos.item(), None, :]
            encoder_mask = batch["encoder_mask"] if "encoder_mask" in batch else None
            logits = model(
                x, encoder_mask=encoder_mask, mask=mask, input_pos=input_pos
            )[:, -1:]
        else:
            logits = model(x, input_pos)
        return self.sample(logits, need_probs=need_probs, **sampling_kwargs)

    """
    Decode the next n tokens.

    Yields a tuple of (token, prob) for each token.
    """

    def decode_n_tokens(
        self,
        model: Model,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        need_probs: bool,
        batch=Optional[Dict[str, Any]],  # Inputs for multimodal models
        callback=lambda _: _,
        eos_token_id: int = 2,
        eot_id: Optional[int] = None,
        attention_backend: SDPBackend = torch.nn.attention.SDPBackend.MATH,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        encountered_eos = False
        for _i in range(
            num_new_tokens - 1
        ):  # -1 to save space to run an EoS if dont generate it naturally
            # Actually better for Inductor to codegen attention here
            with torch.nn.attention.sdpa_kernel([attention_backend]):

                out_token = cur_token.clone()
                next_token, next_prob = self.decode_one_token(
                    model,
                    out_token,
                    input_pos,
                    batch=batch,
                    need_probs=need_probs,
                    **sampling_kwargs,
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                callback(new_tokens[-1], done_generating=_i == num_new_tokens - 2)
                if need_probs or next_prob is None:
                    yield out_token, None
                else:
                    new_probs.append(next_prob.clone())
                    yield out_token, next_prob.clone()
                cur_token = next_token

                # encountered eos
                if next_token.item() == eos_token_id or (
                    eot_id is not None and next_token.item() == eot_id
                ):
                    encountered_eos = True
                    final_token, next_prob = self.decode_one_token(
                        model,
                        cur_token,
                        input_pos,
                        need_probs,
                        batch=batch,
                        **sampling_kwargs,
                    )
                    input_pos += 1
                    break

        if not encountered_eos:
            eos_token = torch.tensor(
                [eos_token_id if eot_id is None else eot_id],
                dtype=cur_token.dtype,
                device=cur_token.device,
            )
            new_tokens.append(eos_token.clone())
            eos_token, next_prob = self.decode_one_token(
                model,
                eos_token.view(1, -1),
                input_pos,
                need_probs,
                batch=batch,
                **sampling_kwargs,
            )
            input_pos += 1
            yield eos_token.clone(), (
                next_prob.clone() if next_prob is not None else None
            )

    def model_forward(self, model, x, input_pos):
        return model(x, input_pos)

    def speculative_decode(
        self,
        model: Model,
        draft_model: Model,
        cur_token: torch.Tensor,
        input_pos: int,
        speculate_k: int,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        **sampling_kwargs,
    ) -> torch.Tensor:
        # draft model inference sequentially
        device = cur_token.device
        orig_input_pos = torch.tensor(
            [input_pos], dtype=torch.int64, device=cur_token.device
        )
        draft_tokens, draft_probs = self.decode_n_tokens(
            draft_model,
            cur_token,
            orig_input_pos.clone(),
            speculate_k,
            batch=batch,
            need_probs=True,
            **sampling_kwargs,
        )

        draft_tokens = torch.cat(draft_tokens)
        # parallel inference on target model using draft tokens
        target_logits = self.model_forward(
            model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(
                input_pos, input_pos + speculate_k + 1, device=cur_token.device
            ),
        )
        target_probs = self.logits_to_probs(target_logits[0], **sampling_kwargs)
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
            last_token = self.multinomial_sample_one_no_sync(target_probs[-1])
            # fill last token into draft model
            self.model_forward(
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
            next_token = self.multinomial_sample_one_no_sync(new)
            return torch.cat([draft_tokens[:accept_length], next_token])

    @torch.no_grad()
    def generate(
        self,
        model: Model,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        chat_mode: bool,
        batch: Optional[
            Dict[str, Any]
        ] = None,  # List of Image prompt tensors for multimodal models
        start_pos: int = 0,
        skip_cache_setup: bool = False,
        draft_model: Model,
        speculate_k: Optional[int] = 8,
        sequential_prefill=True,
        callback=lambda x: x,
        max_seq_length: int,
        attention_backend: SDPBackend = torch.nn.attention.SDPBackend.MATH,
        seed: Optional[int] = None,
        **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """
        if seed:
            torch.manual_seed(seed)

        is_speculative = draft_model is not None
        device, dtype = prompt.device, prompt.dtype

        if len(prompt.shape) > 1:
            prompt = prompt.squeeze(0)
        prompt_length = prompt.size(0)
        max_new_tokens = min(max_new_tokens, max_seq_length - start_pos - prompt_length)
        # set up caches only if first inference
        if start_pos == 0:
            if not skip_cache_setup:
                model = model.to(device=device)
                with torch.device(device):
                    if (
                        self.is_torchtune_model
                        or self.model.config.model_type == ModelType.Flamingo
                    ):
                        # 6404 is one-gpu affordable max_seq_length for single image input
                        model.setup_caches(
                            batch_size=1,
                            dtype=self.dtype,
                            encoder_max_seq_len=6404,
                            decoder_max_seq_len=max_seq_length,
                        )
                    else:
                        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
                    if is_speculative and draft_model is not model:
                        draft_model.setup_caches(
                            max_batch_size=1,
                            max_seq_length=max_seq_length,
                        )
            if model.config.model_type == ModelType.Flamingo:
                model.reset_caches()

        input_pos = torch.arange(
            start_pos, prompt_length + start_pos, device=device, dtype=torch.int
        )

        prefill_t0 = time.perf_counter()
        next_token = self.prefill(
            model,
            prompt.view(1, -1),
            input_pos,
            batch=batch,
            sequential_prefill=sequential_prefill,
            **sampling_kwargs,
        )
        if is_speculative:
            self.prefill(
                draft_model,
                prompt.view(1, -1),
                input_pos,
                sequential_prefill=sequential_prefill,
                **sampling_kwargs,
            )

        time_to_first_token = time.perf_counter() - prefill_t0
        yield None, {"time_to_first_token": time_to_first_token}
        # max_new_tokens <= 2 means we are effectively not calling decode_n_tokens().
        callback(next_token.clone().view(-1), done_generating=max_new_tokens <= 2)

        input_pos = torch.tensor(
            [start_pos + prompt_length], device=device, dtype=torch.int
        )
        accept_counts = [0] * (
            speculate_k + 1
        )  # creates array of [0, 0, 0, ...] that is speculate_k + 1 long

        if is_speculative:
            input_pos = (
                input_pos.item()
            )  # for speculative decoding easier to keep on host
            while input_pos < max_new_tokens - 1:
                cur_token = next_token.view(())

                next_tokens = self.speculative_decode(
                    model,
                    draft_model,
                    cur_token,
                    input_pos,
                    speculate_k,
                    batch=batch,
                    **sampling_kwargs,
                )

                accept_counts[len(next_tokens) - 1] += 1
                num_added = min(max_new_tokens - input_pos - 1, len(next_tokens))
                for token in next_tokens[:num_added,]:
                    callback(token)
                    yield token, None
                input_pos = input_pos + num_added
                next_token = next_tokens[-1]
        else:
            generated_tokens = []
            for generated_token, _ in self.decode_n_tokens(
                model,
                next_token,
                input_pos,
                max_new_tokens - 1,
                batch=batch,
                callback=callback,
                need_probs=False,
                eos_token_id=self.tokenizer.eos_id() if self.tokenizer else 2,
                eot_id=(
                    self.tokenizer.special_tokens["<|eot_id|>"]
                    if self.is_llama3_model
                    else None
                ),
                attention_backend=attention_backend,
                **sampling_kwargs,
            ):
                generated_tokens.append(generated_token.view(-1))
                yield generated_token, None

        generate_stats = {
            "accept_counts": accept_counts,
        }
        yield None, generate_stats

    def encode_tokens(self, string, bos=True, device="cpu"):
        tokens = self.tokenizer.encode(string)
        if bos:
            tokens = [self.tokenizer.bos_id()] + tokens
        logger.debug("Size after encode_tokens: %d", len(tokens))
        logger.debug("Token IDs: %s", tokens)
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def _callback(self, x, *, buffer, done_generating):
        # TODO: Refactor this callback to only include basic functionality & remove print statements
        period_id = self.tokenizer.encode(".")[0]
        buffer.append(
            self.tokenizer.decode([period_id] + x.tolist())[1:]
        )  # I think this results in the first output token being dropped from the display which is wrong.
        if x.item() == self.tokenizer.eos_id():
            done_generating = True
        if (
            self.is_llama3_model
            and x.item() == self.tokenizer.special_tokens["<|eot_id|>"]
        ):
            done_generating = True
            buffer = buffer[:-1]  # drop the eot_id from the output buffer
        if len(buffer) == 4 or done_generating:
            print("".join(buffer), end="", flush=True)
            buffer.clear()

    def _gen_model_input(
        self,
        prompt: Union[str | List[Any]],
        image_prompts: Optional[List[str | Image.Image]] = None,
        max_new_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 2048,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Convert prompt and image prompts into consumable model input args.

        When prompt is a list, the anticipated format is OpenAI API Inspired:
            [ ..., {"role": message["role"], "content": message["content"]}, ...]

        Args:
            prompt (Union[str, List[Any]]): Prompt or list of dialog.
            image_prompts (Optional[List[str | Image.Image]]): List of image prompts. Used only with Llama 3.2 11B.
            max_new_tokens (Optional[int]): Maximum number of new tokens to generate. Used only with Llama 3.2 11B.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, Any]]]: Encoded prompt and batch config for multimodal models.
        """

        # Text-Only model
        if self.model.config.model_type != ModelType.Flamingo:
            # Single String prompt
            if isinstance(prompt, str):
                encoded = self.encode_tokens(
                    prompt, bos=self.model.config.tokenizer_prepend_bos, device=self.builder_args.device
                )
            # List of dialog
            else:
                tokens = self.chat_formatter.encode_dialog_prompt(prompt)
                encoded = torch.tensor(
                    tokens, dtype=torch.int, device=self.builder_args.device
                )

            logger.debug(encoded)
            return encoded, None

        # Llama 3.2 11B
        assert (
            image_prompts is None or len(image_prompts) == 1
        ), "At most one image is supported at the moment"

        if image_prompts and isinstance(image_prompts[0], str):
            images = [Image.open(image_prompts[0])]
        else:
            images = None

        assert (
            max_new_tokens is not None
        ), "max_new_tokens must be specified for Flamingo models"

        # Wrap string prompts into a list
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        image_found = False
        messages = []
        for message in prompt:
            if isinstance(message["content"], str):
                if not image_found and image_prompts:
                    messages.append(
                        Message(
                            role=message["role"],
                            content=[
                                {"type": "image", "content": images[0]},
                                {"type": "text", "content": message["content"]},
                            ],
                        )
                    )
                    image_found = True
                else:
                    messages.append(Message(**message))

            elif isinstance(message["content"], list):
                images = None
                for content_dict in message["content"]:
                    if content_dict["type"] == "text":
                        prompt_arg = content_dict["text"]
                    elif content_dict["type"] == "image_url":
                        assert (
                            images is None
                        ), "At most one image is supported at the moment"

                        base64_decoded = base64.b64decode(
                            content_dict["image_url"].split(";base64,")[1]
                        )
                        images = [Image.open(BytesIO(base64_decoded))]
                        image_found = True

                is_multimodal = images is not None
                content = [{"type": "text", "content": prompt_arg}]

                if is_multimodal:
                    content = [{"type": "image", "content": images[0]}] + content

                messages.append(
                    Message(
                        role=message["role"],
                        content=content,
                    )
                )

        messages.append(
            Message(
                role="assistant",
                content="",
            )
        )

        transform = llama3_2_vision_transform(str(self.tokenizer_args.tokenizer_path))

        device = torch.device(device=self.builder_args.device)

        with device, set_default_dtype(self.dtype):
            data = transform({"messages": messages}, inference=True)

            if image_found:
                batch = padded_collate_tiled_images_and_mask(
                    [data], pad_direction="left", pad_max_images=1, pad_max_tiles=transform.max_num_tiles
                )
                encoded = batch.pop("tokens").to(device).view(-1)
                seq_len = encoded.size(0)
                batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
                batch["encoder_input"]["images"] = batch["encoder_input"]["images"].to(
                    self.dtype
                )

            else:
                encoded = torch.tensor(data["tokens"], device=device).view(-1)
                seq_len = encoded.size(0)
                batch = {}

            total_response_length = seq_len + max_new_tokens
            batch["causal_mask"] = torch.nn.functional.pad(
                torch.tril(
                    torch.ones(
                        size=(total_response_length, total_response_length),
                        dtype=torch.bool,
                    )
                ),
                (
                    0,
                    max_seq_len - total_response_length,
                    0,
                    max_seq_len - total_response_length,
                ),
                value=0,
            )

        logger.debug(encoded)
        return encoded, batch

    def chat(
        self,
        generator_args: GeneratorArgs,
    ):
        if generator_args.chat_mode:
            print("Starting Interactive Chat")

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )
        if self.builder_args.distributed:
            # During distributed inference the model gets sharded among the ranks
            # So we need to all reduce the model size to get the total model size
            model_size = torch.tensor(model_size, dtype=torch.int64, device=self.device)
            dist.all_reduce(model_size)
            model_size = model_size.item()

        if generator_args.compile:
            if self.builder_args.device == "cpu":
                if generator_args.max_autotune:
                    kwargs = {"mode": "max-autotune"}
                else:
                    kwargs = {}
            else:
                kwargs = {"mode": "reduce-overhead"}

            if self.is_speculative:
                self.model_forward = torch.compile(
                    self.model_forward, fullgraph=True, **kwargs
                )

            if self.model.config.model_type == ModelType.Flamingo:
                # Based on https://github.com/pytorch/torchtune/blob/57ab583c84c4a9dcacac23aeabc81f2a679670fe/torchtune/training/_compile.py#L42-L52
                from torchtune.modules import (
                    TransformerCrossAttentionLayer,
                    TransformerSelfAttentionLayer,
                )

                decoder = self.model.model.decoder
                for m in reversed(list(decoder.modules())):
                    if isinstance(m, TransformerSelfAttentionLayer) or isinstance(
                        m, TransformerCrossAttentionLayer
                    ):
                        m.compile()
            else:
                self.decode_one_token = torch.compile(
                    self.decode_one_token, fullgraph=True, **kwargs
                )

            if generator_args.compile_prefill:
                self.prefill = torch.compile(
                    self.prefill, fullgraph=True, dynamic=True, **kwargs
                )

        self.system_prompt = None
        # Set up our max_seq_length

        # This is a hack to get around the fact that different models have different ways to record their max_seq_length and might be wrong
        # TODO: unify the max_seq_length config representation.
        text_transformer_args = self.model.text_transformer_args
        max_seq_length = (
            text_transformer_args.max_seq_length if text_transformer_args else 2048
        )

        encoded, batch = self._gen_model_input(
            generator_args.prompt,
            generator_args.image_prompts,
            generator_args.max_new_tokens,
            max_seq_length,
        )

        if generator_args.chat_mode:
            print(
                f"Entering Chat Mode. Will continue chatting back and forth with the language model until the models max context length of {max_seq_length} tokens is hit or until the user says /bye"
            )
            get_system_prompt = self.get_user_input(
                "Do you want to enter a system prompt? Enter y for yes and anything else for no. \n"
            )
            if get_system_prompt == "y" or get_system_prompt == "Y":
                self.system_prompt = self.get_user_input("What is your system prompt? \n")

        # `is_torchtune_model` is a misnomer since it doesn't capture all
        # torchtune models (i.e. Flamingo)
        # See Issue: https://github.com/pytorch/torchchat/issues/1273
        elif (
            not generator_args.is_torchtune_model
            and self.model.config.model_type != ModelType.Flamingo
        ):
            max_seq_length = min(
                encoded.size(0) + generator_args.max_new_tokens,
                (
                    text_transformer_args.block_size
                    if text_transformer_args is not None
                    else 2048
                ),
                max_seq_length,
            )

        if self.draft_model is not None:
            max_seq_length += self.speculative_builder_args.speculate_k + 1

        aggregate_metrics = {
            "tokens_per_sec": [],
            "first_token_per_sec": [],
            "next_tokens_per_sec": [],
            "accept_counts": [],
        }
        start_pos = 0

        # arbitrarily large number as chat mode goes until max_seq length
        # or user exits
        num_samples = (
            generator_args.num_samples if not generator_args.chat_mode else 100000
        )
        for i in range(num_samples):
            device_sync(device=self.builder_args.device)
            is_first_sample: bool = i == 0
            if generator_args.chat_mode:
                prompt = self.get_user_input("User: ")
                if prompt == "/bye":
                    print("Exiting Chat.\n")
                    break

                # Encode the additional messages added in this dialog turn. If
                # this is the first turn, that includes any system prompt.
                messages_to_encode = []
                if is_first_sample and self.system_prompt:
                    messages_to_encode.append(
                        {"role": "system", "content": self.system_prompt}
                    )
                messages_to_encode.append({"role": "user", "content": prompt})
                encoded = self.chat_formatter.encode_dialog_prompt(
                    messages_to_encode, add_generation_prompt=True,
                )
                encoded = torch.tensor(
                    encoded, dtype=torch.int, device=self.builder_args.device
                )
                if encoded.size(0) + start_pos > max_seq_length:
                    print(
                        "This prompt would take us past the max_seq_length. Ending Conversation."
                    )
                    break

                print("Model: ", end="")

                buffer = []

                def callback(x, *, done_generating=False):
                    return self._callback(
                        x,
                        buffer=buffer,
                        done_generating=done_generating,
                    )

            else:
                assert not generator_args.chat_mode

                buffer = [generator_args.prompt]

                def callback(x, *, done_generating=False):
                    return self._callback(
                        x,
                        buffer=buffer,
                        done_generating=done_generating,
                    )

            if self.profile:
                from torch._inductor import config as inductor_config

                torch._inductor.config.profiler_mark_wrapper_call = True
                torch._inductor.config.cpp.enable_kernel_profile = True
            if i != generator_args.num_samples - 1 or not self.profile:
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            t0 = time.perf_counter()
            num_tokens_generated = 0
            with prof:
                generator_func = self.generate(
                    self.model,
                    encoded,
                    generator_args.max_new_tokens,
                    draft_model=self.draft_model,
                    speculate_k=generator_args.speculate_k,
                    chat_mode=generator_args.chat_mode,
                    batch=batch,
                    callback=callback,
                    temperature=generator_args.temperature,
                    top_k=generator_args.top_k,
                    sequential_prefill=generator_args.sequential_prefill,
                    start_pos=start_pos,
                    skip_cache_setup=not is_first_sample,
                    max_seq_length=max_seq_length,
                    attention_backend=self.builder_args.attention_backend,
                )
                if generator_args.chat_mode:
                    start_pos += encoded.size(0)
                for token_tensor, metrics in generator_func:
                    if token_tensor is not None:
                        start_pos += token_tensor.size(0)
                        num_tokens_generated += token_tensor.size(0)
                    if metrics is not None:
                        aggregate_metrics.update(metrics)
                    yield token_tensor, metrics
            jit_compile = is_first_sample and (
                generator_args.compile or generator_args.compile_prefill
            )
            compilation_time = time.perf_counter() - t0
            device_sync(device=self.builder_args.device)
            t = time.perf_counter() - t0
            if hasattr(prof, "export_chrome_trace"):
                if self.builder_args.device == "cpu":
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                elif self.builder_args.device == "cuda":
                    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                else:
                    print(prof.key_averages().table(sort_by="self_xpu_time_total"))
                prof.export_chrome_trace(f"{self.profile}.json")

            if start_pos >= max_seq_length:
                print(
                    f"[Max Sequence Length {max_seq_length} Reached. Ending Conversation.]"
                )
                print("---------------------------------------------------")

            tokens_sec = (num_tokens_generated + 1) / t
            first_token_sec = 1 / aggregate_metrics.get("time_to_first_token", 0)
            next_tokens_sec = num_tokens_generated / (
                t - aggregate_metrics.get("time_to_first_token", 0)
            )

            if jit_compile:
                print(
                    f"just-in-time compilation time (incl run time): {compilation_time:.2} seconds"
                )
            else:
                # aggregate_metrics will not append when is jit_compile, which will affect the average numbers.
                aggregate_metrics["tokens_per_sec"].append(tokens_sec)
                aggregate_metrics["first_token_per_sec"].append(first_token_sec)
                aggregate_metrics["next_tokens_per_sec"].append(next_tokens_sec)

            logging.info(
                f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\
                \nGenerated {num_tokens_generated} tokens \
                \nTime for inference {i + 1}: {t:.04f} sec total \
                \nTime to first token: {aggregate_metrics.get('time_to_first_token', 0):.04f} sec \
with {'sequential' if generator_args.sequential_prefill else 'parallel'} prefill.\
                \n\n      Total throughput: {tokens_sec:.04f} tokens/sec, {1 / tokens_sec:.04f} s/token \
                \nFirst token throughput: {first_token_sec:.04f} tokens/sec, {1 / first_token_sec:.04f} s/token \
                \n Next token throughput: {next_tokens_sec:.04f} tokens/sec, {1 / next_tokens_sec:.04f} s/token \
                    "
            )
            logging.info(
                f"\nBandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )
            if i == 0:
                logging.info(
                    f"*** This first iteration will include cold start effects for dynamic import, hardware caches{', JIT compilation' if jit_compile else ''}. ***"
                )
            print("\n========================================\n")
            if start_pos >= max_seq_length:
                if generator_args.chat_mode:
                    break

            if not generator_args.chat_mode:
                start_pos = 0

        if self.is_speculative:
            counts_aggregated = [
                sum(i) for i in zip(*aggregate_metrics["accept_counts"])
            ]
            acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
            print(f"Acceptance probs: {acceptance_probs}")
            print(
                f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
            )

        avg_tokens_sec = torch.mean(
            torch.tensor(aggregate_metrics["tokens_per_sec"])
        ).item()
        avg_first_token_sec = torch.mean(
            torch.tensor(aggregate_metrics["first_token_per_sec"])
        ).item()
        avg_next_tokens_sec = torch.mean(
            torch.tensor(aggregate_metrics["next_tokens_per_sec"])
        ).item()

        if not (
            torch.isnan(torch.tensor(avg_tokens_sec))
            or torch.isnan(torch.tensor(avg_first_token_sec))
            or torch.isnan(torch.tensor(avg_next_tokens_sec))
        ):
            print(
                f"\nWarning: Excluding compile in calculations \
                \n      Average tokens/sec (total): {avg_tokens_sec:.2f} \
                \nAverage tokens/sec (first token): {avg_first_token_sec:.2f} \
                \nAverage tokens/sec (next tokens): {avg_next_tokens_sec:.2f} \n\
                "
            )
        if torch.cuda.is_available():
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        if torch.xpu.is_available():
            print(f"Memory used: {torch.xpu.max_memory_reserved() / 1e9:.02f} GB")



class DistributedGenerator(LocalGenerator):
    def __init__(
        self,
        builder_args: BuilderArgs,
        speculative_builder_args: BuilderArgs,
        tokenizer_args: TokenizerArgs,
        generator_args: GeneratorArgs,
        profile: Optional[Path],
        quantize: bool,
        draft_quantize: bool,
        ):
        
        is_speculative = speculative_builder_args.checkpoint_path is not None
        assert is_speculative == False, "Distributed inference with pp > 1 does not support speculative inference yet."
        super().__init__(
            builder_args,
            speculative_builder_args,
            tokenizer_args,
            generator_args,
            profile,
            quantize,
            draft_quantize,
        )
        self.rank = dist.get_rank()
        # Assuming same number of GPUs per node
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        def distributed_input(prompt: str) -> str:
            if dist.get_rank() == 0:
                text = [input(prompt)]
            else:
                text = [None]
            
            dist.broadcast_object_list(text)
            return text[0]

        self.get_user_input: Callable = distributed_input

        if builder_args.pp > 1:
            self.seqlen_prefill = 1024  # sequence length for prefill stage

            logger.warn(f"{color.yellow}Pipeline parallelism is still experimental and might be slow{color.reset}")
            pp_mesh = self.model.device_mesh["pp"]

            self.pp_rank = pp_mesh.get_local_rank()
            self.pp_group = pp_mesh.get_group()

            self.pp_degree = pp_mesh.size()

            # Convenience variables
            self.first_pp_rank = 0
            self.last_pp_rank = self.pp_degree - 1


            self.first_pp_rank_global_id = dist.get_global_rank(self.pp_group, self.first_pp_rank)
            self.last_pp_rank_global_id = dist.get_global_rank(self.pp_group, self.last_pp_rank)

            self.prefiller = self.create_prefill_stage()
            self.decoder = self.create_decode_stage()

    def __del__(self):
        dist.destroy_process_group()

    # Helper function to get example inputs and outputs for the stages.
    def get_example_ins_outs(self, batch_size: int , seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function generates example inputs and outputs for the prefill and decode stages.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the example inputs and outputs.
        """
        model_dtype = torch.bfloat16
        mb_ids = torch.randint(
            0, self.model.config.vocab_size, (batch_size, seqlen), device=self.device
        )
        activation = torch.rand(
            batch_size, seqlen, self.model.config.dim, device=self.device, dtype=model_dtype
        )
        logits = torch.rand(
            batch_size, seqlen, self.model.config.vocab_size, device=self.device, dtype=model_dtype
        )
        example_inputs = (mb_ids if self.pp_rank == self.first_pp_rank else activation,)
        example_outputs = (logits if self.pp_rank == self.last_pp_rank else activation,)
        return example_inputs, example_outputs

    def create_prefill_stage(self):
        """
        Creates a pipeline stage for prefilling.

        Returns:
            PipelineStage: The created pipeline stage.
        """
        batch_size = 1

        # Create prefill stage
        logger.debug(f"Creating pipeline stage for prefill {self.pp_rank=}, {self.pp_degree=}")
        example_inputs, example_outputs = self.get_example_ins_outs(batch_size, self.seqlen_prefill)
        prefill_stage = PipelineStage(
            self.model,
            self.pp_rank,
            self.pp_degree,
            self.device,
            input_args=example_inputs,
            output_args=example_outputs,
            group=self.pp_group,
        )

        # Create schedule
        # Number of micro-batches for the schedule is 1, because each step() call we
        # only push 1 micro-batch into the pipeline. But we can continuously push
        # new micro-batches into the pipeline as they arrive, achieving same
        # pipelining effect.
        prefiller = ScheduleGPipe(prefill_stage, 1)
        return prefiller

    def create_decode_stage(self):
        """
        Creates a decode stage for the pipeline parallelism.

        Returns:
            ScheduleGPipe: The decode stage.
        """
        # seqlen = 1 now
        seqlen_decode = 1
        batch_size = 1

        # Create decode stage
        # logger.info(f"Creating pipeline stage for decode {self.pp_rank=}, {self.pp_degree=}")
        example_inputs, example_outputs = self.get_example_ins_outs(batch_size, seqlen_decode)
        decode_stage = PipelineStage(
            self.model,
            self.pp_rank,
            self.pp_degree,
            self.device,
            input_args=example_inputs,
            output_args=example_outputs,
            group=self.pp_group,
        )
        # create schedule
        decoder = ScheduleGPipe(decode_stage, 1)

        return decoder

    def prefill(
        self,
        model: Model,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        *,
        sequential_prefill=True,
        **sampling_kwargs,
    ) -> torch.Tensor:
        """
        This function is used to prefill the model with a given prompt. For pipeline parallelism we need to pad the input.

        Returns:
            torch.Tensor: The prefilled tensor.
        """
        if self.builder_args.pp == 1:
            return super().prefill(
                model,
                x,
                input_pos,
                batch,
                sequential_prefill=sequential_prefill,
                **sampling_kwargs,
            )

        pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.eos_id
        prompt_length = x.size(1)

        padded_seq = torch.full(
            (1, self.seqlen_prefill), pad_token_id, dtype=torch.int64, device=self.device
            )
        padded_seq[:,:prompt_length] = x
        input_pos = torch.arange(
            self.seqlen_prefill,
            device=self.device,
            dtype=torch.int,
            )

        # Prefill phase
        # Run context input through pipeline
        # TODO: we need to pass `input_pos` and `cache_lane` to each stage.
        lane = 0
        kwargs = {"input_pos": input_pos, "cache_lane": lane}
        
        if self.pp_rank == self.first_pp_rank:
            logits = self.prefiller.step(padded_seq, **kwargs)
        elif self.pp_rank == self.last_pp_rank:
            logits = self.prefiller.step(**kwargs)
        else:  # middle pp ranks
            self.prefiller.step(**kwargs)

        if self.pp_rank == self.last_pp_rank:
            new_token = self.sample(logits[:,:prompt_length], need_probs=False, **sampling_kwargs)[0]
            if self.pp_rank != self.first_pp_rank:
                dist.send(
                    new_token,
                    dst=self.first_pp_rank_global_id,
                    group=self.pp_group,
                )
        else:
            new_token = torch.zeros(1, 1, device=self.device, dtype=torch.int64)
            if self.pp_rank == self.first_pp_rank:
                dist.recv(
                    new_token,
                    src=self.last_pp_rank_global_id,
                    group=self.pp_group,
                )

        return new_token

    def decode_one_token(
        self,
        model: Model,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        need_probs: bool,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        **sampling_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decodes a single token.

        # TODO: implement speculative decoding with pp>1
        Returns:
            Tuple[torch.Tensor, None]: A tuple containing the decoded token and None.
        """
        if self.builder_args.pp == 1:
            return super().decode_one_token(
                model,
                x,
                input_pos,
                need_probs,
                batch=batch,
                **sampling_kwargs,
            )

        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1

        new_token = x.view(1, -1)

        lane = 0
        kwargs = {"input_pos": input_pos, "cache_lane": lane}
        # Run data through pipeline
        if self.pp_rank == self.first_pp_rank:
            logits = self.decoder.step(new_token, **kwargs)
        elif self.pp_rank == self.last_pp_rank:
            logits = self.decoder.step(**kwargs)
        else:  # middle pp ranks
            self.decoder.step(**kwargs)

        # Decode the output
        if self.pp_rank == self.last_pp_rank:
            new_token, _ = self.sample(logits, need_probs=need_probs, **sampling_kwargs)
            if self.pp_rank != self.first_pp_rank:
                dist.send(
                    new_token,
                    dst=self.first_pp_rank_global_id,
                    group=self.pp_group,
                )
        else:
            new_token = torch.zeros(1, 1, device=self.device, dtype=torch.int64)
            if self.pp_rank == self.first_pp_rank:
                dist.recv(
                    new_token,
                    src=self.last_pp_rank_global_id,
                    group=self.pp_group,
                )
                #TODO: Why do we get 2d tensor here?
                new_token=new_token[0]
        return new_token, None

    def sample(
        self,
        logits,
        need_probs: bool,
        temperature: float = 0,
        top_k: Optional[int] = None,
    ):
        if temperature == 0 and not need_probs:
            _, idx_next = torch.topk(logits[0, -1], k=1, dim=-1)
            return (idx_next, None)
        probs = self.logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        
        return idx_next, probs


def run_generator(
    args,
    rank: Optional[int] =None
    ):
    """
    This function creates and executes a generator 
    """
    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)    
    #Setup rank 1 and up to suppress log messages and print messages
    if builder_args.distributed and rank != 0:
        logger.setLevel(logging.CRITICAL)
        context = contextlib.redirect_stdout(None)
    else:
        context = contextlib.nullcontext()

    with context:
        Generator = DistributedGenerator if builder_args.distributed else LocalGenerator
        logger.debug("GeneratorArgs: %s", generator_args)
        gen = Generator(
            builder_args,
            speculative_builder_args,
            tokenizer_args,
            generator_args,
            args.profile,
            args.quantize,
            args.draft_quantize,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if torch.xpu.is_available():
            torch.xpu.reset_peak_memory_stats()

        for _ in gen.chat(generator_args):
            pass

def main(args):
    builder_args = BuilderArgs.from_args(args)
    
    if builder_args.distributed:
        world_size = builder_args.tp * builder_args.pp

        ctx = mp.get_context('spawn')
        with futures.ProcessPoolExecutor(max_workers=world_size-1, mp_context=ctx) as executor:
            for i in range(1,world_size):
                fn = partial(run_generator, args, i)
                executor.submit(run_in_dist_env, world_size, i, fn)
            #Starting rank 0
            fn = partial(run_generator, args, 0)
            run_in_dist_env(world_size, 0, fn)
    else:
        run_generator(args)

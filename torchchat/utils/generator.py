# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
from os import PathLike
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from torchchat.cli.builder import (
    _initialize_tokenizer,
    BuilderArgs,
    TokenizerArgs,
)

# torchtune model definition dependencies
from torchtune.data import Message, padded_collate_tiled_images_and_mask
from torchtune.models.llama3_2_vision._model_builders import llama3_2_vision_transform
from torchtune.training import set_default_dtype


class _ChatFormatter(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def encode_dialog_prompt(self, dialog) -> List[int]:
        raise NotImplementedError()


class Llama3ChatFormatter(_ChatFormatter):
    """Format a chat prompt using special tokens to demarcate roles and messages.

    Refer to the LLaMA3 documentation for more details https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    """

    def encode_header(self, role) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message) -> List[int]:
        tokens = self.encode_header(message["role"])
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

    def encode_dialog_prompt(self, dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header("assistant"))  # Pass role directly as a string
        return tokens


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"


class Llama2ChatFormatter(_ChatFormatter):
    def encode_dialog_prompt(self, dialog) -> List[int]:
        tokens = self.tokenizer.encode(f"{B_INST} ")
        first_message = True  # Bool to handle placing the B_INST token. Behavior is weird - the system prompt should have the B_INST, but not the first user message. All following user messages *should* have it. Also, if there is no system prompt, then the user message should have it.
        for message in dialog:
            if isinstance(message["content"], list):
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            content = content.strip()
            if message["role"] == "system":
                encoded = self.tokenizer.encode(f"{B_SYS}\n{content}\n{E_SYS}")
                first_message = False
            elif message["role"] == "user":
                encoded = [self.tokenizer.bos_id()] + self.tokenizer.encode(
                    f"{B_INST if first_message else ''} {content} {E_INST} "
                )
                first_message = True
            elif message["role"] == "assistant":
                encoded = self.tokenizer.encode(f"{content}\n\n") + [
                    self.tokenizer.eos_id()
                ]
            tokens += encoded
        return tokens


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


class Generator(object):
    """
    Base class for generators that can be used to generate text samples based on a pre-trained Transformer model and tokenizer.
    """

    def __init__(
        self,
        builder_args: BuilderArgs,
        tokenizer_args: TokenizerArgs,
        generator_args: GeneratorArgs,
    ):
        self.builder_args = builder_args
        self.tokenizer_args = tokenizer_args
        self.generate_args = generator_args

        self.dtype = builder_args.precision

        self.tokenizer = _initialize_tokenizer(self.tokenizer_args)

        # Right now the assumption is only llama3 uses tiktokenizer and it
        # must use tiktokenizer.
        # Piggy backing off of this flag then for now to identify llama3
        # without prompting user.
        self.is_llama3_model = self.tokenizer_args.is_tiktoken
        if self.is_llama3_model:
            self.chat_formatter = Llama3ChatFormatter(self.tokenizer)
            if generator_args.chat_mode:
                logging.debug(
                    "Llama3 model detected in chat mode. Using updated sentence schemas"
                )
        else:
            self.chat_formatter = Llama2ChatFormatter(self.tokenizer)

    @abstractmethod
    def is_text_only(self) -> bool:
        """
        Returns True if the model is text-only, False otherwise.
        """
        raise NotImplementedError()

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
        if self.is_text_only():
            # Single String prompt
            if isinstance(prompt, str):
                encoded = self.encode_tokens(
                    prompt, bos=True, device=self.builder_args.device
                )
            # List of dialog
            else:
                tokens = self.chat_formatter.encode_dialog_prompt(prompt)
                encoded = torch.tensor(
                    tokens, dtype=torch.int, device=self.builder_args.device
                )

            logging.debug(encoded)
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
                    [data], pad_direction="left", pad_max_images=1
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

        logging.debug(encoded)
        return encoded, batch

    def encode_tokens(self, string, bos=True, device="cpu"):
        tokens = self.tokenizer.encode(string)
        if bos:
            tokens = [self.tokenizer.bos_id()] + tokens
        logging.debug(f"Size after encode_tokens: {len(tokens)}")
        return torch.tensor(tokens, dtype=torch.int, device=device)

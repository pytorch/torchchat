# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import base64
import os
import time
import uuid

from abc import ABC
from dataclasses import dataclass
from io import BytesIO
from pwd import getpwuid
from typing import Any, Dict, List, Optional, Union, Type

import torch

from PIL import Image

from torchtune.data import Message, padded_collate_tiled_images_and_mask

from torchtune.models.llama3_2_vision._model_builders import llama3_2_vision_transform

from torchchat.cli.download import is_model_downloaded, load_model_configs
from torchchat.generate import LocalGenerator, DistributedGenerator, GeneratorArgs
from torchchat.model import FlamingoModel

from torchchat.utils.build_utils import device_sync


"""Dataclasses defined around the objects used the OpenAI API Chat specification.

See https://platform.openai.com/docs/api-reference/chat for the full specification and details.
"""

OPENAI_API_DEFAULT_MAX_TOKENS = 16

# Message classes and associated objects - see the types of Messages under "Create Chat Completion >>> Request body >>> messages"


@dataclass
class _ContentPart(ABC):
    """A single part of a message content field.

    See the "Assistants >>> Messages >>> Create Message >>> Request body >>> content >>> Show possible types" section of the OpenAI API docs for more details.
    """

    type: str


@dataclass
class ImageFile:
    file_id: str
    detail: Optional[str]


@dataclass
class ImageFileContentPart(_ContentPart):
    type: str = "image_file"
    image_file: Optional[ImageFile] = None


@dataclass
class ImageUrl:
    url: str
    detail: Optional[str]


@dataclass
class ImageUrlContentPart(_ContentPart):
    type: str = "image_url"
    image_url: Optional[ImageUrl] = None


@dataclass
class TextContentPart(_ContentPart):
    text: str = ""
    type: str = "text"


@dataclass
class _AbstractMessage(ABC):
    """Base class with common parameters for message types.

    Each message type is associated with a role (one of "system", "user", "assistant" or "tool") and contains an
    optional content field.

    See more details at https://platform.openai.com/docs/guides/text-generation/chat-completions-api .
    """

    role: str
    content: Optional[Union[List[_ContentPart], str]] = None


@dataclass
class SystemMessage(_AbstractMessage):
    role: str = "system"
    name: Optional[str] = None


@dataclass
class UserMessage(_AbstractMessage):
    role: str = "user"


@dataclass
class ToolMessage:
    tool_call_id: str
    type: str
    role: str = "tool"


@dataclass
class ToolCallFunction:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    type: str
    function: ToolCallFunction


@dataclass
class AssistantMessage(_AbstractMessage):
    role: str = "assistant"
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


# Completion request and response types.


@dataclass
class StreamOptions:
    """Parameters for streamed responses.

    Only set when `stream` is set to `true` in the request.
    """

    include_usage: bool = False


@dataclass
class ResponseFormat:
    type: Optional[str] = None


@dataclass
class CompletionRequest:
    """A full chat completion request.

    See the "Create Chat Completion >>> Request body" section of the OpenAI API docs for more details.
    """

    messages: List[_AbstractMessage]
    model: str
    frequency_penalty: float = 0.0  # unimplemented
    logit_bias: Optional[Dict[str, float]] = None  # unimplemented
    logprobs: Optional[bool] = None  # unimplemented
    top_logprobs: Optional[int] = None  # unimplemented
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0  # unimplemented
    response_format: Optional[ResponseFormat] = None  # unimplemented
    seed: Optional[int] = None
    service_tier: Optional[str] = None  # unimplemented
    stop: Optional[List[str]] = None  # unimplemented
    stream: bool = False
    stream_options: Optional[StreamOptions] = None  # unimplemented
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0  # unimplemented
    tools: Optional[List[Any]] = None  # unimplemented - Assistant features
    tool_choice: Optional[Union[str, Any]] = None  # unimplemented - Assistant features
    parallel_tool_calls: Optional[bool] = None  # unimplemented - Assistant features
    user: Optional[str] = None  # unimplemented

    def __post_init__(self):
        if isinstance(self.stream, str):
            self.stream = self.stream.lower() != "false"
        else:
            self.stream = bool(self.stream)


@dataclass
class CompletionChoice:
    """A single choice in a chat completion response.

    See the "The chat completion object >>> choices" section of the OpenAI API docs for more details.
    """

    index: int
    message: AssistantMessage
    finish_reason: str = None
    logprobs: Optional[List[Any]] = None


@dataclass
class UsageStats:
    """Object representing a single choice in a chat completion response.

    See the "The chat completion object >>> usage" section of the OpenAI API docs for more details.
    """

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    """A full chat completion response.

    See the "The chat completion object" section of the OpenAI API docs for more details.
    """

    id: str
    choices: List[CompletionChoice]
    created: int
    model: str
    system_fingerprint: str
    service_tier: Optional[str] = None
    usage: Optional[UsageStats] = None
    object: str = "chat.completion"


@dataclass
class ChunkDelta:
    """Changes between the previous chunk emitted for a chunked completion response.

    See the "The chat completion chunk object >>> choices >>> delta" section of the OpenAI API docs for more details.
    """

    tool_calls: Optional[List[ToolCall]]
    role: Optional[str]
    content: Optional[Union[List[_ContentPart], str]] = None


@dataclass
class CompletionChoiceChunk:
    """A single choice in a chat completion chunk response.

    See the "The chat completion chunk object >>> choices" section of the OpenAI API docs for more details.
    """

    delta: ChunkDelta
    index: int
    finish_reason: Optional[str] = None
    logprobs: Optional[List[Any]] = None


@dataclass
class CompletionResponseChunk:
    """Response chunk emitted during a chunked completion response.

    See the "The chat completion chunk object" section of the OpenAI API docs for more details.
    """

    id: str
    choices: List[CompletionChoiceChunk]
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    object: str = "chat.completion.chunk"
    usage: Optional[UsageStats] = None


class OpenAiApiGeneratorMixin:
    """A wrapper over the Generator class to interface with the OpenAI API.

    Implements endpoints for completion requests, both chunked and non-chunked using the dataclasses
    defined above.
    """

    def __init__(self, *args, **kwargs):
        """Initialize generator and parameters for maintaining context during generation.

        See the docstring for the Generator class in generate.py for argument details.
        """

        super().__init__(*args, **kwargs)
        try:
            self.max_seq_length = (
                self.model.text_transformer_args.max_seq_length
                + self.speculative_builder_args.speculate_k
                + 1
                if self.draft_model is not None
                else self.model.text_transformer_args.max_seq_length
            )
        except:
            self.max_seq_length = 2048
            print(
                f"can not find max_seq_length in model config, use default value: {self.max_seq_length}"
            )
        # The System fingerprint is a unique identifier for the model and its configuration.
        self.system_fingerprint = (
            f"{self.builder_args.device}_{self.builder_args.precision}"
        )

    def _gen_model_inputs_from_openai_completion_request(
        self, completion_request: CompletionRequest
    ) -> List[Message]:
        """Generate model inputs from an OpenAI completion request.

        Args:
            completion_request: Request object with prompt and other parameters.

        Returns:
            Modle inputs.
        """
        messages = completion_request.messages

        # Not Llama 3.2 11B
        if not isinstance(self.model, FlamingoModel):
            prompt = [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ]
            return self._gen_model_input(
                prompt=prompt, max_new_tokens=completion_request.max_tokens
            )

        # Llama 3.2 11B

        prompt = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]

        return self._gen_model_input(
            prompt=prompt, max_new_tokens=completion_request.max_tokens
        )

    def chunked_completion(self, completion_request: CompletionRequest):
        """Handle a chat completion request and yield a chunked response.

        ** Warning ** : Not all arguments of the CompletionRequest are consumed as the server isn't completely implemented.
        Current treatment of parameters is described below.

        - messages: The server consumes the final element of the array as the prompt.
        - model: This has no impact on the server state, i.e. changing the model in the request
        will not change which model is responding. Instead, use the --model flag to seelect the model when starting the server.
        - temperature: This is used to control the randomness of the response.
        - system_fingerprint: A unique identifier for the model and its configuration. Currently unimplemented - subject to change.

        See https://github.com/pytorch/torchchat/issues/973 for more details.


        Args:
            completion_request: Request object with prompt and other parameters.

        Yields:
            CompletionResponseChunk objects in response to completion_request as tokens are generated.

        """

        # Initialize counters for chunk responses and encode the prompt.
        id = str(uuid.uuid4())
        device_sync(device=self.builder_args.device)
        encoded, batch = self._gen_model_inputs_from_openai_completion_request(
            completion_request
        )

        idx = 0
        start_pos = 0

        generator_args = GeneratorArgs(
            None,
            max_new_tokens=(
                int(completion_request.max_tokens)
                if completion_request.max_tokens
                else OPENAI_API_DEFAULT_MAX_TOKENS
            ),
            encoded_prompt=encoded,
            temperature=float(completion_request.temperature),
            chat_mode=False,
            sequential_prefill=True,
        )

        def callback(x, *, done_generating=False):
            return self._callback(
                x,
                buffer=None,
                done_generating=done_generating,
            )

        device_sync(device=self.builder_args.device)

        buffer = []
        ILLEGAL_CHAR = '\ufffd'
        # Process each token, metrics tuple yielded by Generator.generate.
        for y, _ in self.generate(
            model=self.model,
            prompt=encoded,
            max_new_tokens=generator_args.max_new_tokens,
            draft_model=self.draft_model,
            speculate_k=generator_args.speculate_k,
            chat_mode=generator_args.chat_mode,
            batch=batch,
            callback=callback,
            temperature=generator_args.temperature,
            top_k=generator_args.top_k,
            sequential_prefill=generator_args.sequential_prefill,
            start_pos=start_pos,
            max_seq_length=self.max_seq_length,
            seed=int(completion_request.seed or 0),
        ):
            if y is None:
                continue

            elif y.item() == self.tokenizer.eos_id:
                # Stop generation if the EOS token is generated.
                break

            y = y.view(-1)
            buffer.append(y.item())
            # Decode the torch.Tensor token to a string and append to the buffer. Separate the sequences with a period token.
            content = "".join(
                self.tokenizer.decode([self.tokenizer.encode(".")[0]] + buffer)[1:]
            )
            # Skip content while illegal characters appear.
            if ILLEGAL_CHAR in content:
                continue
            buffer.clear()

            # Package the sequence into a CompletionChunkResponse and yield it.
            chunk_delta = ChunkDelta(
                role="assistant",
                content=content,
                tool_calls=None,
            )
            choice_chunk = CompletionChoiceChunk(
                delta=chunk_delta,
                index=idx,
                finish_reason=None,
            )
            chunk_response = CompletionResponseChunk(
                id="chatcmpl-" + str(id),
                choices=[choice_chunk],
                created=int(time.time()),
                model=completion_request.model,
                system_fingerprint=self.system_fingerprint,
            )
            yield chunk_response
            start_pos += y.size(0)
            idx += 1

        # Yield an ending chunk indicating the generation has completed.
        end_chunk = CompletionChoiceChunk(
            ChunkDelta(None, None, None), idx, finish_reason="stop"
        )

        yield CompletionResponseChunk(
            id="chatcmpl-" + str(id),
            choices=[end_chunk],
            created=int(time.time()),
            model=completion_request.model,
            system_fingerprint=self.system_fingerprint,
        )

    def sync_completion(self, request: CompletionRequest):
        """Handle a chat completion request and yield a single, non-chunked response"""
        output = ""
        for chunk in self.chunked_completion(request):
            if not chunk.choices[0].finish_reason:
                output += chunk.choices[0].delta.content

        message = AssistantMessage(content=output)
        return CompletionResponse(
            id="chatcmpl-" + str(uuid.uuid4()),
            choices=[
                CompletionChoice(
                    finish_reason="stop",
                    index=0,
                    message=message,
                )
            ],
            created=int(time.time()),
            model=request.model,
            system_fingerprint=self.system_fingerprint,
        )

    def _callback(self, x, *, buffer, done_generating):
        pass


def create_openai_api_generator(distributed: bool) -> Type:
    """
    Factory method to create an OpenAiApiGenerator
    """

    # Base class order matters to make sure OpenAiApiGeneratorMixin overrides methods in DistributedGenerator and Generator
    return type('OpenAiApiGenerator', (OpenAiApiGeneratorMixin, DistributedGenerator if distributed else LocalGenerator), {})


"""
Helper functions for the OpenAI API Models endpoint.

See https://platform.openai.com/docs/api-reference/models for the full specification and details.
Please create an issue if anything doesn't match the specification.
"""


@dataclass
class ModelInfo:
    """The Model object per the OpenAI API specification containing information about a model.

    See https://platform.openai.com/docs/api-reference/models/object for more details.
    """

    id: str
    created: int
    owned_by: str
    object: str = "model"


@dataclass
class ModelInfoList:
    """A list of ModelInfo objects."""

    data: List[ModelInfo]
    object: str = "list"


def retrieve_model_info(args, model_id: str) -> Union[ModelInfo, None]:
    """Implementation of the OpenAI API Retrieve Model endpoint.

    See https://platform.openai.com/docs/api-reference/models/retrieve

    Inputs:
        args: command line arguments
        model_id: the id of the model requested

    Returns:
        ModelInfo describing the specified if it is downloaded, None otherwise.
    """
    if model_config := load_model_configs().get(model_id):
        if is_model_downloaded(model_id, args.model_directory):
            path = args.model_directory / model_config.name
            created = int(os.path.getctime(path))
            owned_by = getpwuid(os.stat(path).st_uid).pw_name

            return ModelInfo(id=model_config.name, created=created, owned_by=owned_by)
        return None
    return None


def get_model_info_list(args) -> ModelInfo:
    """Implementation of the OpenAI API List Models endpoint.

    See https://platform.openai.com/docs/api-reference/models/list

    Inputs:
        args: command line arguments

    Returns:
        ModelInfoList describing all downloaded models.
    """
    data = []
    for model_id, model_config in load_model_configs().items():
        if is_model_downloaded(model_id, args.model_directory):
            path = args.model_directory / model_config.name
            created = int(os.path.getctime(path))
            owned_by = getpwuid(os.stat(path).st_uid).pw_name

            data.append(
                ModelInfo(id=model_config.name, created=created, owned_by=owned_by)
            )
    response = ModelInfoList(data=data)
    return response

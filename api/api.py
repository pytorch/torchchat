# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import uuid
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from build.utils import device_sync

from generate import Generator, GeneratorArgs

"""Dataclasses defined around the objects used the OpenAI API Chat specification.

See https://platform.openai.com/docs/api-reference/chat for the full specification and details.
"""

# Message classes and associated objects - see the types of Messages under "Create Chat Completion >>> Request body >>> messages"


@dataclass
class _AbstractMessage(ABC):
    """Base class with common parameters for message types.

    Each message type is associated with a role (one of "system", "user", "assistant" or "tool") and contains an
    optional content field.

    See more details at https://platform.openai.com/docs/guides/text-generation/chat-completions-api .
    """

    role: str
    content: Optional[str] = None


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
    max_tokens: Optional[int] = None  # unimplemented
    n: int = 1
    presence_penalty: float = 0  # unimplemented
    response_format: Optional[ResponseFormat] = None  # unimplemented
    seed: Optional[int] = None  # unimplemented
    service_tier: Optional[str] = None  # unimplemented
    stop: Optional[List[str]] = None  # unimplemented
    stream: bool = False
    stream_options: Optional[StreamOptions] = None  # unimplemented
    temperature: Optional[float] = 1.0  # unimplemented
    top_p: Optional[float] = 1.0  # unimplemented
    tools: Optional[List[Any]] = None  # unimplemented
    tool_choice: Optional[Union[str, Any]] = None  # unimplemented
    parallel_tool_calls: Optional[bool] = None  # unimplemented
    user: Optional[str] = None  # unimplemented


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
    content: Optional[str]


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
    system_fingerprint: str
    service_tier: Optional[str] = None
    object: str = "chat.completion.chunk"
    usage: Optional[UsageStats] = None


class OpenAiApiGenerator(Generator):
    """A wrapper over the Generator class to interface with the OpenAI API.

    Implements endpoints for completion requests, both chunked and non-chunked using the dataclasses
    defined above.
    """

    def __init__(self, *args, **kwargs):
        """Initialize generator and parameters for maintaining context during generation.

        See the docstring for the Generator class in generate.py for argument details.
        """

        super().__init__(*args, **kwargs)
        self.start_pos = 0
        self.max_seq_length = (
            self.model.config.max_seq_length
            + self.speculative_builder_args.speculate_k
            + 1
            if self.draft_model is not None
            else self.model.config.max_seq_length
        )
        # The System fingerprint is a unique identifier for the model and its configuration.
        # Currently, this is not implemented in a
        self.system_fingerprint = (
            self.builder_args.device + type(self.builder_args.precision).__name__
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
        device_sync(device=self.builder_args.device)

        # Initialize counters for chunk responses and encode the prompt.
        id = str(uuid.uuid4())

        idx = 0
        buffer = []
        encoded = self.encode_tokens(
            completion_request.messages[-1].get("content"),
            bos=True,
            device=self.builder_args.device,
        )
        generator_args = GeneratorArgs(
            completion_request.messages[-1].get("content"),
            encoded_prompt=encoded,
            chat_mode=False,
        )

        def callback(x, *, done_generating=False):
            return self._callback(
                x,
                buffer=buffer,
                done_generating=done_generating,
            )

        # Process each token, metrics tuple yielded by Generator.generate.
        for y, _ in self.generate(
            self.model,
            encoded,
            generator_args.max_new_tokens,
            draft_model=self.draft_model,
            speculate_k=generator_args.speculate_k,
            chat_mode=generator_args.chat_mode,
            callback=callback,
            temperature=generator_args.temperature,
            top_k=generator_args.top_k,
            sequential_prefill=generator_args.sequential_prefill,
            start_pos=self.start_pos,
            max_seq_length=self.max_seq_length,
        ):
            if y is None:
                continue

            # Decode the torch.Tensor token to a string and append to the buffer. Separate the sequences with a period token.
            content = "".join(
                self.tokenizer.decode([self.tokenizer.encode(".")[0]] + y.tolist())[1:]
            )

            # Package the sequence into a CompletionChunkResponse and yield it.
            chunk_delta = ChunkDelta(
                role="assistant",
                content=content,
                tool_calls=None,
            )
            choice_chunk = CompletionChoiceChunk(
                delta=chunk_delta,
                index=idx,
            )
            chunk_response = CompletionResponseChunk(
                id=str(id),
                choices=[choice_chunk],
                created=int(time.time()),
                model=completion_request.model,
                system_fingerprint=self.system_fingerprint,
            )
            yield chunk_response
            self.start_pos += y.size(0)
            idx += 1

        # Yield an ending chunk indicating the generation has completed.
        end_chunk = CompletionChoiceChunk(
            ChunkDelta(None, None, None), idx, finish_reason="stop"
        )

        yield CompletionResponseChunk(
            id=str(id),
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
            id=str(uuid.uuid4()),
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
        period_id = self.tokenizer.encode(".")[0]
        buffer.append(self.tokenizer.decode([period_id] + x.tolist())[1:])
        if (
            self.is_llama3_model
            and x.item() == self.tokenizer.special_tokens["<|eot_id|>"]
        ):
            buffer = buffer[:-1]  # drop the eot_id from the output buffer

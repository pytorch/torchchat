# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import uuid
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
class CompletionRequest:
    """A full chat completion request.

    See the "Create Chat Completion >>> Request body" section of the OpenAI API docs for more details.
    """

    model: str
    prompt: str
    messages: Optional[List[_AbstractMessage]]
    frequency_penalty: float = 0.0
    temperature: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    echo: bool = False
    frequency_penalty: float = 0.0
    guided_decode_json_schema: str = None
    guided_decode_json_schema_path: str = None
    n: int = 1
    presence_penalty: float = 0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None


@dataclass
class CompletionChoice:
    """A single choice in a chat completion response.

    See the "The chat completion object >>> choices" section of the OpenAI API docs for more details.
    """

    finish_reason: str
    index: int
    message: AssistantMessage
    logprobs: Optional[List[Any]]


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
    usage: UsageStats
    object: str = "chat.completion"
    service_tier: Optional[str] = None


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
    object: str = "chat.completion.chunk"
    service_tier: Optional[str] = None
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

    def completion(self, completion_request: CompletionRequest):
        """Handle a chat completion request and yield a chunked response.

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
            completion_request.prompt, bos=True, device=self.builder_args.device
        )
        generator_args = GeneratorArgs(
            completion_request.prompt,
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
                system_fingerprint=uuid.UUID(int=uuid.getnode()),
            )
            yield chunk_response
            self.start_pos += y.size(0)
            idx += 1

        # Yield an ending chunk indicating the generation has completed.
        end_chunk = CompletionChoiceChunk(ChunkDelta(None, None, None), idx, "eos")

        yield CompletionResponseChunk(
            id=str(id),
            choices=[end_chunk],
            created=int(time.time()),
            model=completion_request.model,
            system_fingerprint=uuid.UUID(int=uuid.getnode()),
        )

    def _callback(self, x, *, buffer, done_generating):
        period_id = self.tokenizer.encode(".")[0]
        buffer.append(self.tokenizer.decode([period_id] + x.tolist())[1:])
        if (
            self.is_llama3_model
            and x.item() == self.tokenizer.special_tokens["<|eot_id|>"]
        ):
            buffer = buffer[:-1]  # drop the eot_id from the output buffer

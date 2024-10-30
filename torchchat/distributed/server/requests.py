"""Module for text generation sampling parameters and request handling."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, List, Optional, Union

from torchchat.distributed.server.core_utils import _GB, _MB, Counter


class RequestBase(ABC):
    """Abstract base class for request handling."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Check if request is finished."""
        pass

    @abstractmethod
    def get_response(self) -> str:
        """Get generated response."""
        pass


class SamplingStrategy(Enum):
    """Enumeration of available sampling strategies."""

    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    SAMPLE = "sample"


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.

    Following OpenAI's text completion API parameters
    (https://platform.openai.com/docs/api-reference/completions/create).
    """

    # Sampling configuration
    """
    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        use_beam_search: Whether to use beam search instead of sampling.
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
    """
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    use_beam_search: bool = False

    # Generation control
    stop: List[str] = field(default_factory=list)
    ignore_eos: bool = False
    max_tokens: int = 16
    logprobs: Optional[int] = None

    # Constants
    _SAMPLING_EPS: Final[float] = 1e-5
    _PENALTY_RANGE: Final[tuple] = (-2.0, 2.0)

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.best_of = self.best_of if self.best_of is not None else self.n
        self._validate_all()

    @property
    def strategy(self) -> SamplingStrategy:
        """Determine the sampling strategy based on parameters.
        Zero temperature means greedy sampling.
        """
        if self.use_beam_search:
            return SamplingStrategy.BEAM_SEARCH
        elif self.temperature < self._SAMPLING_EPS:
            return SamplingStrategy.GREEDY
        return SamplingStrategy.SAMPLE

    def _validate_all(self) -> None:
        """Validate all parameters."""
        self._validate_basic_params()

        strategy = self.strategy
        if strategy == SamplingStrategy.BEAM_SEARCH:
            self._validate_beam_search()
        elif strategy == SamplingStrategy.GREEDY:
            self._validate_greedy_sampling()

    def _validate_basic_params(self) -> None:
        """Validate basic parameters."""
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")
        if self.best_of < self.n:
            raise ValueError(
                f"best_of must be >= n, got n={self.n}, best_of={self.best_of}"
            )

        for penalty in [self.presence_penalty, self.frequency_penalty]:
            if not (self._PENALTY_RANGE[0] <= penalty <= self._PENALTY_RANGE[1]):
                raise ValueError(
                    f"Penalty must be in [{self._PENALTY_RANGE[0]}, {self._PENALTY_RANGE[1]}]"
                )

        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}"
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable) or >= 1, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}")

    def _validate_beam_search(self) -> None:
        """Validate beam search specific parameters."""
        if self.best_of == 1:
            raise ValueError(f"best_of must be > 1 for beam search, got {self.best_of}")
        if self.temperature > self._SAMPLING_EPS:
            raise ValueError("temperature must be 0 for beam search")
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError("top_p must be 1 for beam search")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 for beam search")

    def _validate_greedy_sampling(self) -> None:
        """Validate greedy sampling specific parameters."""
        if self.best_of > 1:
            raise ValueError(
                f"best_of must be 1 for greedy sampling, got {self.best_of}"
            )
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError("top_p must be 1 for greedy sampling")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 for greedy sampling")


@dataclass
class Request(RequestBase):
    """A request containing prompt, generated tokens and related information."""

    # Static attributes
    arrival_time: float
    request_id: int
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    priority: int = 0

    # Dynamic state
    generated_tokens: List[str] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    _is_finished: bool = False
    is_running: bool = False
    process_time: float = 0.0
    last_step_time: float = 0.0

    def _check_finish_condition(self) -> None:
        """Check if generation should be finished."""
        if len(self.generated_tokens) >= self.sampling_params.max_tokens:
            self._is_finished = True
            return

        if (
            not self.sampling_params.ignore_eos
            and self.generated_tokens
            and self.generated_tokens[-1] in self.sampling_params.stop
        ):
            self._is_finished = True

    def add_generated_token(self, token: str, token_id: int) -> None:
        """Add a newly generated token."""
        if len(self.generated_tokens) >= self.sampling_params.max_tokens:
            raise ValueError(
                f"Generated tokens exceed max length {self.sampling_params.max_tokens} "
                f"for request {self.request_id}"
            )

        self.generated_tokens.append(token)
        self.generated_token_ids.append(token_id)
        self._check_finish_condition()

    def is_context_stage(self) -> bool:
        """Check if request is in context stage."""
        return not bool(self.generated_tokens)

    @property
    def input_length(self) -> int:
        """Get input length."""
        return len(self.prompt_token_ids)

    @property
    def output_length(self) -> int:
        """Get output length."""
        return len(self.generated_token_ids)

    def get_response(self) -> str:
        """Get generated response."""
        return "".join(self.generated_tokens)

    def get_input_token_ids(self) -> List[int]:
        """Get input token ids for next iteration."""
        if self.is_context_stage():
            return self.prompt_token_ids
        return [self.generated_token_ids[-1]]

    @property
    def num_input_tokens(self) -> int:
        """Get number of input tokens."""
        return len(self.get_input_token_ids())

    def get_first_new_token_index(self) -> int:
        """Get index of first newly generated token."""
        if self.is_context_stage():
            return 0
        return self.input_length + self.output_length - 1

    @property
    def kv_cache_slots(self) -> int:
        """Get number of KV cache slots needed."""
        return self.input_length + self.output_length

    @property
    def is_finished(self) -> bool:
        """Check if request is finished."""
        return self._is_finished


@dataclass
class Request(RequestBase):
    """A request containing prompt, generated tokens and related information."""

    # Static attributes
    arrival_time: float
    request_id: int
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    priority: int = 0

    # Dynamic state
    generated_tokens: List[str] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    _is_finished: bool = False
    is_running: bool = False
    process_time: float = 0.0
    last_step_time: float = 0.0

    def _check_finish_condition(self) -> None:
        """Check if generation should be finished."""
        if len(self.generated_tokens) >= self.sampling_params.max_tokens:
            self._is_finished = True
            return

        if (
            not self.sampling_params.ignore_eos
            and self.generated_tokens
            and self.generated_tokens[-1] in self.sampling_params.stop
        ):
            self._is_finished = True

    def add_generated_token(self, token: str, token_id: int) -> None:
        """Add a newly generated token."""
        if len(self.generated_tokens) >= self.sampling_params.max_tokens:
            raise ValueError(
                f"Generated tokens exceed max length {self.sampling_params.max_tokens} "
                f"for request {self.request_id}"
            )

        self.generated_tokens.append(token)
        self.generated_token_ids.append(token_id)
        self._check_finish_condition()

    def is_context_stage(self) -> bool:
        """Check if request is in context stage."""
        return not bool(self.generated_tokens)

    @property
    def input_length(self) -> int:
        """Get input length."""
        return len(self.prompt_token_ids)

    @property
    def output_length(self) -> int:
        """Get output length."""
        return len(self.generated_token_ids)

    def get_response(self) -> str:
        """Get generated response."""
        return "".join(self.generated_tokens)

    def get_input_token_ids(self) -> List[int]:
        """Get input token ids for next iteration."""
        if self.is_context_stage():
            return self.prompt_token_ids
        return [self.generated_token_ids[-1]]

    @property
    def num_input_tokens(self) -> int:
        """Get number of input tokens."""
        return len(self.get_input_token_ids())

    def get_first_new_token_index(self) -> int:
        """Get index of first newly generated token."""
        if self.is_context_stage():
            return 0
        return self.input_length + self.output_length - 1

    @property
    def kv_cache_slots(self) -> int:
        """Get number of KV cache slots needed."""
        return self.input_length + self.output_length

    @property
    def is_finished(self) -> bool:
        """Check if request is finished."""
        return self._is_finished


@dataclass
class BatchedRequests:
    """A batch of requests processed together."""

    requests: List[Request] = field(default_factory=list)
    start_time: Optional[float] = None
    is_running: bool = False

    def add_request(self, request: Request) -> None:
        """Add a new request to batch."""
        if request.request_id in self.request_ids:
            raise ValueError(f"Request {request.request_id} already exists")
        self.requests.append(request)

    def pop_finished_requests(self) -> List[Request]:
        """Remove and return finished requests."""
        finished, unfinished = [], []
        for request in self.requests:
            (finished if request.is_finished else unfinished).append(request)
        self.requests = unfinished
        return finished

    def start_iteration(self, start_time: float) -> None:
        """Start a new processing iteration."""
        if self.start_time is not None:
            raise RuntimeError("Batch already started iteration")
        self.start_time = start_time
        self.is_running = True

    def finish_iteration(
        self,
        generated_tokens: List[str],
        generated_token_ids: List[int],
        end_time: float,
    ) -> None:
        """Finish current processing iteration."""
        if self.start_time is None:
            raise RuntimeError("Batch not started")

        for request, token, token_id in zip(
            self.requests, generated_tokens, generated_token_ids
        ):
            request.last_step_time = end_time
            request.process_time += end_time - self.start_time
            request.add_generated_token(token, token_id)

        self.start_time = None
        self.is_running = False

    @property
    def request_ids(self) -> List[int]:
        """Get IDs of all requests in batch."""
        return [request.request_id for request in self.requests]

    @property
    def kv_cache_slots(self) -> int:
        """Get total KV cache slots needed."""
        return sum(request.kv_cache_slots for request in self.requests)

    @property
    def num_input_tokens(self) -> int:
        """Get total number of input tokens."""
        return sum(request.num_input_tokens for request in self.requests)

    def get_input_tokens_batched(self) -> List[List[int]]:
        """Get batched input tokens for all requests."""
        return [request.get_input_token_ids() for request in self.requests]

    def get_first_token_indexes(self) -> List[int]:
        """Get first token indexes for all requests."""
        return [request.get_first_new_token_index() for request in self.requests]

    def get_is_context_stage(self) -> List[bool]:
        """Get context stage status for all requests."""
        return [request.is_context_stage() for request in self.requests]


@dataclass
class MigratingRequest:
    """Request migrating between processing stages.

    Represents a request that:
    - Has finished context stage
    - Not yet accepted by decoding stage
    - Block still in context stage GPU memory
    """

    request: Request
    block_indexes: List[int]
    context_parallel_config: ParallelConfig


def create_request(
    prompt: Optional[str],
    prompt_token_ids: Optional[List[str]],
    sampling_params: SamplingParams,
    request_counter: Counter,
    tokenizer,
    arrival_time: Optional[float] = None,
    request_id: Optional[int] = None,
) -> Request:
    """Create a new request with the given parameters."""
    request_id = request_id if request_id is not None else next(request_counter)

    if prompt_token_ids is None:
        if prompt is None:
            raise ValueError("Either prompt or prompt_token_ids must be provided")
        prompt_token_ids = tokenizer.encode(prompt)

    if prompt is None:
        prompt = tokenizer.decode(prompt_token_ids)

    arrival_time = arrival_time if arrival_time is not None else time.time()

    return Request(
        arrival_time=arrival_time,
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

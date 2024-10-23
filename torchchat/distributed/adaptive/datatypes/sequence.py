from dataclasses import dataclass, field
from typing import List, Optional

from torchchat.distributed.adaptive.datatypes.block import LogicalTokenBlock
from torchchat.distributed.adaptive.datatypes.sequence_state import SequenceState
from torchchat.distributed.adaptive.datatypes.sequence_status import SequenceStatus
from torchchat.distributed.adaptive.datatypes.token_sampling import SamplingParams
from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()


@dataclass
class Sequence:
    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    block_size: int
    eos_token_id: int
    arrival_time: float
    sampling_params: SamplingParams

    output_token_ids: List[int] = field(default_factory=list)
    prompt_tokens_processed: int = 0
    prompt_tokens_stage_processed: int = 0
    prompt_processing_finished: bool = False
    prompt_stage_processing_finished: bool = False
    output_text: str = ""
    logical_token_blocks: List[LogicalTokenBlock] = field(default_factory=list)
    prefix_offset: int = 0
    read_offset: int = 0
    tokens: Optional[List[str]] = None

    def __post_init__(self):
        self.state = SequenceState(
            self.seq_id, self.arrival_time, len(self.prompt_token_ids)
        )
        self._append_tokens_to_blocks(self.prompt_token_ids)

    @property
    def status(self) -> SequenceStatus:
        return self.state  # ._status

    @status.setter
    def status(self, status: SequenceStatus) -> None:
        self.state.set_status(status)

    def _append_logical_block(self) -> None:
        self.logical_token_blocks.append(
            LogicalTokenBlock(
                block_number=len(self.logical_token_blocks),
                block_size=self.block_size,
            )
        )

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        for token_id in token_ids:
            logger.info(f"Appending token {token_id}")
            """
            if not self.logical_token_blocks or self.logical_token_blocks[-1].is_full():
                self._append_logical_block()
            self.logical_token_blocks[-1].append_tokens([token_id])
            """

    def update_prompt_tokens_processed(self, num_tokens: int) -> None:
        if self.prompt_processing_finished:
            raise ValueError("Prompt processing is already finished")
        if num_tokens <= 0:
            raise ValueError("Number of tokens must be positive")

        self.prompt_tokens_processed += num_tokens
        if self.prompt_tokens_processed > len(self.prompt_token_ids):
            raise ValueError("Processed tokens exceed prompt length")

        if self.prompt_tokens_processed == len(self.prompt_token_ids):
            self.prompt_processing_finished = True
            self.state.on_prompt_processing_completed()

    def update_prompt_tokens_stage_processed(self, num_tokens: int) -> None:
        if self.prompt_processing_finished or self.prompt_stage_processing_finished:
            raise ValueError("Prompt processing is already finished")
        if num_tokens <= 0:
            raise ValueError("Number of tokens must be positive")

        self.prompt_tokens_stage_processed += num_tokens
        if self.prompt_tokens_stage_processed > len(self.prompt_token_ids):
            raise ValueError("Stage processed tokens exceed prompt length")

        if self.prompt_tokens_stage_processed == len(self.prompt_token_ids):
            self.prompt_stage_processing_finished = True

    def append_token_id(self, token_id: int) -> None:
        if not self.prompt_processing_finished:
            raise ValueError("Cannot append token before prompt processing is finished")

        self.output_token_ids.append(token_id)
        self._append_tokens_to_blocks([token_id])
        self.state.on_token_generated()

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        return (
            self.output_token_ids[-1]
            if self.output_token_ids
            else self.prompt_token_ids[-1]
        )

    def get_next_prompt_chunk_token_ids(self, chunk_size: int) -> List[int]:
        start = self.prompt_tokens_stage_processed
        end = min(start + chunk_size, len(self.prompt_token_ids))
        return self.prompt_token_ids[start:end]

    def get_next_prompt_chunk_len(self, chunk_size: int) -> int:
        return min(
            chunk_size, len(self.prompt_token_ids) - self.prompt_tokens_stage_processed
        )

    def is_finished(self) -> bool:
        return self.status.is_finished

    def is_executing(self) -> bool:
        return self.status.is_executing

    def is_waiting(self) -> bool:
        return self.status.is_waiting

    def is_paused(self) -> bool:
        return self.status.is_paused

    def is_running(self) -> bool:
        return self.status.is_running

    def reset_for_recompute(self):
        self.status = SequenceStatus.WAITING
        self.prompt_tokens_processed = 0
        self.prompt_tokens_stage_processed = 0
        self.prompt_processing_finished = False
        self.prompt_stage_processing_finished = False
        self.prompt_token_ids.extend(self.output_token_ids)
        self.output_token_ids.clear()

    def check_stop(self) -> None:
        for stop_str in self.sampling_params.stop:
            if self.output_text.endswith(stop_str):
                self.output_text = self.output_text[: -len(stop_str)]
                self.status = SequenceStatus.FINISHED_STOPPED
                return

        if self.get_output_len() == self.sampling_params.max_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        if (
            not self.sampling_params.ignore_eos
            and self.get_last_token_id() == self.eos_token_id
        ):
            self.status = SequenceStatus.FINISHED_STOPPED

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, "
            f"status={self.status}, "
            f"num_blocks={len(self.logical_token_blocks)}, "
            f"num_prompt_tokens={len(self.prompt_token_ids)}, "
            f"num_output_tokens={len(self.output_token_ids)}, "
            f"prompt_processing_finished={self.prompt_processing_finished}, "
            f"num_prompt_tokens_processed={self.prompt_tokens_processed}, "
            f"num_prompt_tokens_stage_processed={self.prompt_tokens_stage_processed}, "
            f"prompt_stage_processing_finished={self.prompt_stage_processing_finished})"
        )


@dataclass
class SequenceScheduleMetadata:
    seq_id: str
    prompt_chunk_len: int

    @property
    def num_prompt_tokens(self) -> int:
        return self.prompt_chunk_len

    @property
    def is_prompt(self) -> bool:
        return self.prompt_chunk_len > 0

    @property
    def num_output_tokens(self) -> int:
        return 0 if self.is_prompt else 1

    @property
    def num_tokens(self) -> int:
        return max(self.prompt_chunk_len, 1)

    @classmethod
    def from_sequence(
        cls, seq: Sequence, prompt_chunk_len: Optional[int] = None
    ) -> "SequenceScheduleMetadata":
        if prompt_chunk_len is None:
            prompt_chunk_len = (
                0 if seq.prompt_stage_processing_finished else seq.get_prompt_len()
            )
        return cls(seq_id=seq.seq_id, prompt_chunk_len=prompt_chunk_len)


@dataclass
class SequenceMetadata:
    seq: Sequence
    block_table: Optional[List[int]]
    prompt_chunk_len: int

    @property
    def num_prompt_tokens(self) -> int:
        return self.prompt_chunk_len

    @property
    def is_prompt(self) -> bool:
        return self.prompt_chunk_len > 0

    @property
    def num_output_tokens(self) -> int:
        return 0 if self.is_prompt else 1

    @property
    def num_tokens(self) -> int:
        return max(self.prompt_chunk_len, 1)


@dataclass
class SamplerOutput:
    seq_id: str
    output_token: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SamplerOutput):
            return NotImplemented
        return self.seq_id == other.seq_id and self.output_token == other.output_token


SamplerOutputs = List[SamplerOutput]

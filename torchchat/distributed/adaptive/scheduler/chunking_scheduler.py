import time
from enum import Enum, auto
from typing import List, Tuple
from dataclasses import dataclass

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig, SimpleChunkingSchedulerConfig
from sarathi.core.block_space_manager.vllm_block_space_manager import VLLMBlockSpaceManager
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)

class Turn(Enum):
    PREFILL = auto()
    DECODE = auto()

@dataclass
class SchedulerState:
    running: List[Sequence]
    ignored_seq_ids: List[str]
    preempted_seq_ids: List[str]
    scheduled_seq_metadata_list: List[SequenceScheduleMetadata]
    num_batched_tokens: int

class SimpleChunkingScheduler(BaseScheduler):
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SimpleChunkingSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)
        self.chunk_size = self.scheduler_config.chunk_size
        self.whose_turn = Turn.PREFILL

    def get_block_space_manager_class(self):
        return VLLMBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(self, seq: Sequence, num_batched_tokens: int) -> int:
        assert not seq.is_finished()
        return min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            self.chunk_size - num_batched_tokens,
        )

    def _schedule(self) -> SchedulerOutputs:
        now = time.monotonic()
        state = SchedulerState([], [], [], [], 0)

        self._process_prefill_turn(state, now)
        
        if not state.scheduled_seq_metadata_list:
            self._process_decode_turn(state)

        self.whose_turn = Turn.PREFILL if self.whose_turn == Turn.DECODE else Turn.DECODE
        self.running = state.running

        return SchedulerOutputs(
            id=self.state.iteration_id,
            ignored_seq_ids=state.ignored_seq_ids,
            preempted_seq_ids=state.preempted_seq_ids,
            scheduled_seq_metadata_list=state.scheduled_seq_metadata_list,
        )

    def _process_prefill_turn(self, state: SchedulerState, now: float) -> None:
        if self.whose_turn == Turn.PREFILL:
            self._process_running_prefill(state)
            if not state.scheduled_seq_metadata_list:
                self._process_waiting_prefill(state, now)

    def _process_running_prefill(self, state: SchedulerState) -> None:
        self.running = self.policy.sort_by_priority(time.monotonic(), self.running)

        while self.running:
            seq = self.running.pop(0)
            if not seq.is_paused() or seq.prompt_stage_processing_finished:
                state.running.append(seq)
                continue

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(seq, state.num_batched_tokens)
            if next_num_prefill_tokens == 0:
                state.running.append(seq)
                continue

            state.num_batched_tokens += next_num_prefill_tokens
            state.running.append(seq)
            state.scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq, prompt_chunk_len=next_num_prefill_tokens)
            )

        self.running, state.running = state.running, []

    def _process_waiting_prefill(self, state: SchedulerState, now: float) -> None:
        while self.state.waiting:
            seq = self.state.waiting[0]
            if seq.arrival_time > now or not self._can_schedule_waiting_sequence(seq, state):
                break

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(seq, state.num_batched_tokens)
            if next_num_prefill_tokens == 0:
                break

            self.state.waiting.popleft()
            self._allocate(seq)
            self.running.append(seq)
            state.num_batched_tokens += next_num_prefill_tokens
            state.scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq, prompt_chunk_len=next_num_prefill_tokens)
            )

    def _can_schedule_waiting_sequence(self, seq: Sequence, state: SchedulerState) -> bool:
        if not self._check_request_prompt_length(seq):
            state.ignored_seq_ids.append(seq.seq_id)
            return False
        return (
            self.block_manager.can_allocate(seq)
            and len(self.running) + 1 <= self.scheduler_config.max_num_seqs
        )

    def _process_decode_turn(self, state: SchedulerState) -> None:
        while self.running:
            seq = self.running.pop(0)
            if not seq.is_paused() or not seq.prompt_stage_processing_finished:
                state.running.append(seq)
                continue

            if not self._try_append_slot(seq, state):
                break

    def _try_append_slot(self, seq: Sequence, state: SchedulerState) -> bool:
        while not self.block_manager.can_append_slot():
            if not self.running:
                self._preempt(seq)
                state.preempted_seq_ids.append(seq.seq_id)
                return False
            victim_seq = self.running.pop(-1)
            self._preempt(victim_seq)
            state.preempted_seq_ids.append(victim_seq.seq_id)

        self._append_slot(seq)
        state.running.append(seq)
        state.scheduled_seq_metadata_list.append(SequenceScheduleMetadata.from_sequence(seq))
        return True

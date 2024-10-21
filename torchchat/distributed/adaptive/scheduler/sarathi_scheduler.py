import time
from typing import List, Optional
from dataclasses import dataclass

import numpy as np

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig, SarathiSchedulerConfig
from sarathi.core.block_space_manager.sarathi_block_space_manager import SarathiBlockSpaceManager
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)

@dataclass
class ChunkSchedule:
    chunk_sizes: List[int]
    tokens_per_stage: int

class SarathiScheduler(BaseScheduler):
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size
        self.enable_dynamic_chunking_schedule = self.scheduler_config.enable_dynamic_chunking_schedule
        self.chunk_schedule: Optional[ChunkSchedule] = None

        if self.enable_dynamic_chunking_schedule:
            self._validate_dynamic_chunking_config()
            self.chunk_schedule = self._compute_chunk_size_schedule()

    def _validate_dynamic_chunking_config(self) -> None:
        assert self.scheduler_config.chunk_schedule_stages > 0, "Chunk schedule stages must be positive"
        assert self.scheduler_config.chunk_schedule_max_tokens > 0, "Chunk schedule max tokens must be positive"
        assert self.scheduler_config.low_chunk_size % 32 == 0, "Low chunk size must be a multiple of 32"
        assert self.scheduler_config.high_chunk_size % 32 == 0, "High chunk size must be a multiple of 32"

    def _compute_chunk_size_schedule(self) -> ChunkSchedule:
        chunk_sizes = np.linspace(
            self.scheduler_config.low_chunk_size,
            self.scheduler_config.high_chunk_size,
            self.scheduler_config.chunk_schedule_stages,
            dtype=np.int32,
        )[::-1]
        
        round_of_chunk_sizes = min(32, self.scheduler_config.low_chunk_size)
        chunk_sizes = np.round(chunk_sizes / round_of_chunk_sizes) * round_of_chunk_sizes
        chunk_sizes = chunk_sizes.astype(np.int64).tolist()

        tokens_per_stage = int(np.ceil(
            self.scheduler_config.chunk_schedule_max_tokens / self.scheduler_config.chunk_schedule_stages
        ))

        return ChunkSchedule(chunk_sizes, tokens_per_stage)

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(self, seq: Sequence, num_batched_tokens: int) -> int:
        assert not seq.is_finished()

        if self.enable_dynamic_chunking_schedule and self.chunk_schedule:
            request_stage_idx = min(
                int(np.ceil(seq.get_num_prompt_tokens_stage_processed() / self.chunk_schedule.tokens_per_stage)),
                len(self.chunk_schedule.chunk_sizes) - 1
            )
            chunk_size = self.chunk_schedule.chunk_sizes[request_stage_idx]
        else:
            chunk_size = self.chunk_size

        return min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            chunk_size - num_batched_tokens,
        )

    def _schedule(self) -> SchedulerOutputs:
        now = time.monotonic()

        running: List[Sequence] = []
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0

        self._process_running_sequences(now, running, preempted_seq_ids, scheduled_seq_metadata_list, num_batched_tokens)
        self._process_waiting_sequences(now, running, ignored_seq_ids, scheduled_seq_metadata_list, num_batched_tokens)

        self.running = running

        return SchedulerOutputs(
            id=self.state.iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )

    def _process_running_sequences(
        self,
        now: float,
        running: List[Sequence],
        preempted_seq_ids: List[str],
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
        num_batched_tokens: int,
    ) -> None:
        self.running = self.policy.sort_by_priority(now, self.running)
        running_prefills: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                running.append(seq)
                continue

            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
                continue

            self._process_completed_prefill_sequence(seq, running, preempted_seq_ids, scheduled_seq_metadata_list, num_batched_tokens)

        for seq in running_prefills:
            self._process_incomplete_prefill_sequence(seq, running, scheduled_seq_metadata_list, num_batched_tokens)

    def _process_completed_prefill_sequence(
        self,
        seq: Sequence,
        running: List[Sequence],
        preempted_seq_ids: List[str],
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
        num_batched_tokens: int,
    ) -> None:
        while not self.block_manager.can_append_slot():
            if self.running:
                victim_seq = self.running.pop(-1)
                self._preempt(victim_seq)
                preempted_seq_ids.append(victim_seq.seq_id)
            else:
                self._preempt(seq)
                preempted_seq_ids.append(seq.seq_id)
                return

        self._append_slot(seq)
        running.append(seq)
        num_batched_tokens += 1
        scheduled_seq_metadata_list.append(SequenceScheduleMetadata.from_sequence(seq))

    def _process_incomplete_prefill_sequence(
        self,
        seq: Sequence,
        running: List[Sequence],
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
        num_batched_tokens: int,
    ) -> None:
        assert not seq.prompt_stage_processing_finished

        next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(seq, num_batched_tokens)

        if next_num_prefill_tokens == 0:
            running.append(seq)
            return

        num_batched_tokens += next_num_prefill_tokens
        scheduled_seq_metadata_list.append(
            SequenceScheduleMetadata.from_sequence(seq, prompt_chunk_len=next_num_prefill_tokens)
        )
        running.append(seq)

    def _process_waiting_sequences(
        self,
        now: float,
        running: List[Sequence],
        ignored_seq_ids: List[str],
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
        num_batched_tokens: int,
    ) -> None:
        while self.state.waiting:
            seq = self.state.waiting[0]

            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                self.state.waiting.popleft()
                continue

            if not self.block_manager.can_allocate(seq):
                break

            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(seq, num_batched_tokens)

            if next_num_prefill_tokens == 0:
                break

            seq = self.state.waiting.popleft()
            self._allocate(seq)
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq, prompt_chunk_len=next_num_prefill_tokens)
            )
            running.append(seq)

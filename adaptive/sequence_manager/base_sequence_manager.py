from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from sarathi.config import SystemConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import (
    SamplerOutput,
    SamplerOutputs,
    Sequence,
    SequenceMetadata,
    SequenceScheduleMetadata,
)
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.utils.threading_utils import synchronized

@dataclass
class SequenceProcessingResult:
    ignored_seqs: List[Sequence]
    seq_metadata_list: List[SequenceMetadata]

class BaseSequenceManager(ABC):

    def __init__(self, config: SystemConfig):
        self.seq_map: Dict[str, Sequence] = {}

    @synchronized
    def add_seq(self, seq: Sequence) -> None:
        if seq.seq_id in self.seq_map:
            raise ValueError(f"Sequence with id {seq.seq_id} already exists")
        self.seq_map[seq.seq_id] = seq

    def _free_seq(self, seq_id: str) -> None:
        self.seq_map.pop(seq_id, None)

    def _preempt_seq(self, seq_id: str) -> None:
        seq = self._get_seq(seq_id)
        if not seq.is_executing():
            raise ValueError(f"Cannot preempt sequence {seq_id}: not executing")
        seq.reset_for_recompute()

    def _pause_seq(self, seq_id: str) -> None:
        seq = self._get_seq(seq_id)
        if not seq.is_running():
            raise ValueError(f"Cannot pause sequence {seq_id}: not running")
        seq.set_status(SequenceStatus.PAUSED)

    def _resume_seq(self, seq_id: str) -> None:
        seq = self._get_seq(seq_id)
        if not (seq.is_waiting() or seq.is_paused()):
            raise ValueError(f"Cannot resume sequence {seq_id}: neither waiting nor paused")
        seq.set_status(SequenceStatus.RUNNING)

    def _on_seq_scheduled(self, seq_sched_metadata: SequenceScheduleMetadata) -> None:
        self._resume_seq(seq_sched_metadata.seq_id)

    @abstractmethod
    def _get_block_table(self, seq: Sequence) -> List[int]:
        pass

    @synchronized
    def on_schedule(self, scheduler_outputs: SchedulerOutputs) -> SequenceProcessingResult:
        ignored_seqs = [self._free_and_get_seq(seq_id) for seq_id in scheduler_outputs.ignored_seq_ids]
        
        for seq_id in scheduler_outputs.preempted_seq_ids:
            self._preempt_seq(seq_id)

        seq_metadata_list = [
            self._create_sequence_metadata(seq_sched_metadata)
            for seq_sched_metadata in scheduler_outputs.scheduled_seq_metadata_list
        ]

        return SequenceProcessingResult(ignored_seqs, seq_metadata_list)

    def _create_sequence_metadata(self, seq_sched_metadata: SequenceScheduleMetadata) -> SequenceMetadata:
        self._on_seq_scheduled(seq_sched_metadata)
        seq = self._get_seq(seq_sched_metadata.seq_id)
        return SequenceMetadata(
            seq,
            self._get_block_table(seq),
            seq_sched_metadata.num_prompt_tokens,
        )

    @abstractmethod
    def _on_append_token(self, seq: Sequence) -> None:
        pass

    def _process_seq_output(self, seq: Sequence, sample: SamplerOutput) -> None:
        if seq.is_finished() or not seq.prompt_processing_finished:
            return

        seq.append_token_id(sample.output_token)
        self._on_append_token(seq)
        seq.check_stop()
        if seq.is_finished():
            self._free_seq(seq.seq_id)

    @synchronized
    def on_step_completed(self, scheduler_outputs: SchedulerOutputs, sampler_outputs: Optional[SamplerOutputs]) -> None:
        if sampler_outputs is None:
            return

        for scheduled_seq_metadata, sampler_output in zip(scheduler_outputs.scheduled_seq_metadata_list, sampler_outputs):
            if scheduled_seq_metadata.seq_id != sampler_output.seq_id:
                raise ValueError(f"Mismatch in sequence IDs: {scheduled_seq_metadata.seq_id} != {sampler_output.seq_id}")
            
            seq = self._get_seq(scheduled_seq_metadata.seq_id)
            if seq.is_waiting():
                continue

            if not seq.prompt_processing_finished:
                seq.update_prompt_tokens_stage_processed(scheduled_seq_metadata.prompt_chunk_len)
                seq.update_prompt_tokens_processed(scheduled_seq_metadata.prompt_chunk_len)

            self._pause_seq(scheduled_seq_metadata.seq_id)
            self._process_seq_output(seq, sampler_output)

    def generate_request_outputs(self, processing_result: SequenceProcessingResult) -> List[RequestOutput]:
        all_seqs = processing_result.ignored_seqs + [x.seq for x in processing_result.seq_metadata_list]
        return [RequestOutput.from_seq(seq) for seq in all_seqs]

    def _get_seq(self, seq_id: str) -> Sequence:
        if seq_id not in self.seq_map:
            raise KeyError(f"Sequence with id {seq_id} not found")
        return self.seq_map[seq_id]

    def _free_and_get_seq(self, seq_id: str) -> Sequence:
        seq = self._get_seq(seq_id)
        self._free_seq(seq_id)
        return seq

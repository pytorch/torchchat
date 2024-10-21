from dataclasses import dataclass, field
from typing import List
from sarathi.core.datatypes.sequence import SequenceScheduleMetadata

@dataclass
class SchedulerOutputs:
    """
    Represents the output of a scheduler in a language model system.

    Attributes:
        id: Unique identifier for this scheduler output.
        ignored_seq_ids: List of sequence IDs that were ignored.
        preempted_seq_ids: List of sequence IDs that were preempted.
        scheduled_seq_metadata_list: List of metadata for scheduled sequences.
    """
    id: int
    ignored_seq_ids: List[str]
    preempted_seq_ids: List[str]
    scheduled_seq_metadata_list: List[SequenceScheduleMetadata]

    prompt_chunk_lens: List[int] = field(init=False)
    num_batched_prompt_tokens: int = field(init=False)
    num_batched_output_tokens: int = field(init=False)
    num_batched_tokens: int = field(init=False)

    def __post_init__(self):
        self.scheduled_seq_metadata_list.sort(key=lambda x: not x.is_prompt)
        self.prompt_chunk_lens = [metadata.num_prompt_tokens for metadata in self.scheduled_seq_metadata_list]
        self.num_batched_prompt_tokens = sum(self.prompt_chunk_lens)
        self.num_batched_output_tokens = sum(metadata.num_output_tokens for metadata in self.scheduled_seq_metadata_list)
        self.num_batched_tokens = sum(metadata.num_tokens for metadata in self.scheduled_seq_metadata_list)

    @property
    def is_empty(self) -> bool:
        """Check if there are no scheduled sequences."""
        return not self.scheduled_seq_metadata_list

    @property
    def has_no_output(self) -> bool:
        """Check if there are no scheduled, ignored, or preempted sequences."""
        return not (self.scheduled_seq_metadata_list or self.ignored_seq_ids or self.preempted_seq_ids)

    @property
    def seq_ids(self) -> List[str]:
        """Get the list of sequence IDs for scheduled sequences."""
        return [metadata.seq_id for metadata in self.scheduled_seq_metadata_list]

    def __repr__(self) -> str:
        return (
            f"SchedulerOutputs(id={self.id}, "
            f"ignored_seq_ids={self.ignored_seq_ids}, "
            f"preempted_seq_ids={self.preempted_seq_ids}, "
            f"scheduled_seq_count={len(self.scheduled_seq_metadata_list)}, "
            f"num_batched_tokens={self.num_batched_tokens})"
        )

 from abc import ABC, abstractmethod
from typing import List, Optional
from collections import deque
from dataclasses import dataclass

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.block_space_manager.block_space_manager_registry import BlockSpaceManagerRegistry
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceStatus
from sarathi.core.policy import PolicyFactory

from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()

@dataclass
class SchedulerState:
    iteration_id: int = -1
    num_running_batches: int = 0
    waiting: deque[Sequence] = deque()
    running: List[Sequence] = []

class BaseScheduler(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.metrics_store = MetricsStore.get_instance()
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.state = SchedulerState()

        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.get_type(),
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            model_config.max_model_len,
        )
        self.prompt_limit = model_config.max_model_len

    def reset_state(self) -> None:
        """Reset the scheduler state."""
        self.state = SchedulerState()

    def add_seq(self, seq: Sequence) -> None:
        """Add a sequence to the waiting queue."""
        self.state.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:
        """Check if there are any unfinished sequences."""
        return bool(self.state.waiting or self.state.running)

    def get_num_unfinished_seqs(self) -> int:
        """Get the number of unfinished sequences."""
        return len(self.state.waiting) + len(self.state.running)

    @abstractmethod
    def _schedule(self) -> SchedulerOutputs:
        """Abstract method to be implemented by subclasses."""
        pass

    def schedule(self) -> SchedulerOutputs:
        """Schedule sequence groups."""
        self.state.iteration_id += 1

        if self.state.num_running_batches >= self.parallel_config.pipeline_parallel_size:
            return SchedulerOutputs(
                self.state.iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )

        scheduler_outputs = self._schedule()

        if not scheduler_outputs.is_empty():
            self.state.num_running_batches += 1

        return scheduler_outputs

    def free_finished_seqs(self) -> None:
        """Free finished sequences and update the running list."""
        for seq in self.state.running:
            if seq.is_finished():
                self._free_seq(seq)
        self.state.running = [seq for seq in self.state.running if not seq.is_finished()]

    def on_step_completed(self) -> None:
        """Perform actions when a step is completed."""
        self.free_finished_seqs()
        self.state.num_running_batches -= 1

    def _allocate(self, seq: Sequence) -> None:
        """Allocate resources for a sequence."""
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        """Free resources allocated to a sequence."""
        self.block_manager.free(seq)

    def _append_slot(self, seq: Sequence) -> None:
        """Append a slot to an executing sequence."""
        if not seq.is_executing():
            raise ValueError("Sequence must be in executing state to append a slot.")
        self.block_manager.append_slot(seq)

    def _preempt(self, seq: Sequence) -> None:
        """Preempt an executing sequence."""
        if not seq.is_executing():
            raise ValueError("Sequence must be in executing state to be preempted.")
        self._free_seq(seq)
        self.state.waiting.appendleft(seq)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        """Check if the prompt length is within the limit."""
        if seq.get_len() > self.prompt_limit:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long "
                f"and exceeds limit of {self.prompt_limit}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.state.waiting.popleft()
            return False
        return True

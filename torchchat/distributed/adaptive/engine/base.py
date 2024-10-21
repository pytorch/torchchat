from abc import ABC, abstractmethod
from typing import List, Optional
from collections import deque

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.block_space_manager.block_space_manager_registry import BlockSpaceManagerRegistry
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceStatus
from sarathi.core.policy import PolicyFactory
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore

logger = init_logger(__name__)

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

        self._iteration_id: int = -1
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.get_type(),
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            model_config.max_model_len,
        )
        self.prompt_limit: int = model_config.max_model_len
        self.num_running_batches: int = 0
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []

    def reset_state(self) -> None:
        self._iteration_id = -1

    def add_seq(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting or self.running)

    def get_num_unfinished_seqs(self) -> int:
        return len(self.waiting) + len(self.running)

    @abstractmethod
    def _schedule(self) -> SchedulerOutputs:
        pass

    def schedule(self) -> SchedulerOutputs:
        self._iteration_id += 1

        if self.num_running_batches >= self.parallel_config.pipeline_parallel_size:
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )

        scheduler_outputs = self._schedule()

        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1

        return scheduler_outputs

    def free_finished_seqs(self) -> None:
        finished_seqs = [seq for seq in self.running if seq.is_finished()]
        for seq in finished_seqs:
            self._free_seq(seq)
        self.running = [seq for seq in self.running if not seq.is_finished()]

    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.num_running_batches = max(0, self.num_running_batches - 1)

    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def _append_slot(self, seq: Sequence) -> None:
        if not seq.is_executing():
            raise ValueError(f"Sequence {seq.seq_id} is not in executing state")
        self.block_manager.append_slot(seq)

    def _preempt(self, seq: Sequence) -> None:
        if not seq.is_executing():
            raise ValueError(f"Sequence {seq.seq_id} is not in executing state")
        self._free_seq(seq)
        self.waiting.appendleft(seq)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.prompt_limit:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long "
                f"and exceeds limit of {self.prompt_limit}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.waiting.popleft()
            return False
        return True

"""Configuration system for distributed model serving."""

from dataclasses import dataclass, field
from enum import auto, Enum
from typing import ClassVar, Dict, Final, List, Optional

import torch

from torchchat.distributed.server.core_utils import _GB


@dataclass
class ParallelConfig:
    """Configuration for distributed execution."""

    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    pipeline_parallel_size: int = 1
    pipeline_parallel_rank: int = 0

    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size
        self.use_parallel = self.world_size > 1

    def to_list(self) -> List[int]:
        """Convert config to list format."""
        return [
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
        ]

    def is_last_stage(self) -> bool:
        """Check if this is the last pipeline stage."""
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1


@dataclass
class DisaggParallelConfig:
    """Configuration for disaggregated execution."""

    context: ParallelConfig
    decoding: ParallelConfig

    def get_num_workers(self) -> int:
        """Get total number of required workers."""
        return self.context.world_size + self.decoding.world_size


@dataclass
class SchedulerConfigBase:
    """Base configuration for schedulers."""

    policy: SchedulerPolicy
    max_batch_size: int
    max_tokens_per_batch: int

    def __post_init__(self):
        if isinstance(self.policy, str):
            try:
                self.policy = SchedulerPolicy(self.policy)
            except ValueError:
                raise ValueError(f"Unsupported policy: {self.policy}")


@dataclass
class ContextStageSchedConfig(SchedulerConfigBase):
    """Configuration for context stage scheduler."""

    parallel_config: Optional[ParallelConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.policy != SchedulerPolicy.FCFS:
            raise ValueError(
                f"Context stage only supports FCFS policy, got {self.policy}"
            )


@dataclass
class DecodingStageSchedConfig(SchedulerConfigBase):
    """Configuration for decoding stage scheduler."""

    model_name: Optional[str] = None
    waiting_block_prop_threshold: float = 0.05

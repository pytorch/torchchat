from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Type, Union
from enum import Enum, auto

class SchedulerType(Enum):
    SIMPLE_CHUNKING = auto()
    ADAPTIVE = auto()

PRIMITIVE_TYPES: Set[Type] = {int, str, float, bool, type(None)}

def get_all_subclasses(cls: Type) -> List[Type]:
    """Recursively get all subclasses of a given class."""
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in get_all_subclasses(s)]

def is_primitive_type(field_type: Type) -> bool:
    """Check if the type is a primitive type."""
    return field_type in PRIMITIVE_TYPES

@dataclass
class BaseConfig(ABC):
    @classmethod
    def create_from_type(cls: Type['BaseConfig'], type_: Any) -> 'BaseConfig':
        for subclass in get_all_subclasses(cls):
            if subclass.get_type() == type_:
                return subclass()
        raise ValueError(f"Invalid type: {type_}")

    @staticmethod
    @abstractmethod
    def get_type() -> Any:
        pass

@dataclass
class CacheConfig:
    block_size: int = field(
        default=16,
        metadata={"help": "Size of a cache block in number of tokens."}
    )
    num_gpu_blocks: Optional[int] = field(
        default=None,
        metadata={"help": "Number of GPU blocks for caching. This gets set after profiling."}
    )

@dataclass
class ParallelConfig:
    pipeline_parallel_size: int = field(
        default=2,
        metadata={"help": "Number of pipeline parallel groups."}
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel groups."}
    )

    def __post_init__(self) -> None:
        self.world_size: int = self.pipeline_parallel_size * self.tensor_parallel_size

@dataclass
class BaseSchedulerConfig(BaseConfig):
    max_num_seqs: int = field(
        default=128,
        metadata={"help": "Maximum number of sequences to be processed in a single iteration (batch size)."}
    )

    @abstractmethod
    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        pass

@dataclass
class SimpleChunkingSchedulerConfig(BaseSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Size of each chunk for simple chunking scheduler."}
    )

    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        return self.chunk_size

    @staticmethod
    def get_type() -> SchedulerType:
        return SchedulerType.SIMPLE_CHUNKING

@dataclass
class AdaptiveSchedulerConfig(BaseSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Size of each chunk for Adaptive scheduler."}
    )
    enable_dynamic_chunking_schedule: bool = field(
        default=False,
        metadata={"help": "Enable dynamic chunking schedule."}
    )
    low_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum chunk size for dynamic chunking."}
    )
    high_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum chunk size for dynamic chunking."}
    )
    chunk_schedule_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens for chunk scheduling."}
    )
    chunk_schedule_stages: Optional[int] = field(
        default=None,
        metadata={"help": "Number of stages for chunk scheduling."}
    )

    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        if self.enable_dynamic_chunking_schedule:
            return self.high_chunk_size or self.chunk_size
        return self.chunk_size

    @staticmethod
    def get_type() -> SchedulerType:
        return SchedulerType.ADAPTIVE

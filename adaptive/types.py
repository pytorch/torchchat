from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, NewType

# Custom types
GPULocation = NewType('GPULocation', Tuple[Optional[str], int])
ResourceMapping = NewType('ResourceMapping', List[GPULocation])
ReplicaResourceMapping = NewType('ReplicaResourceMapping', List[ResourceMapping])

class BaseEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class SchedulerType(BaseEnum):
    VLLM = auto()
    ORCA = auto()
    FASTER_TRANSFORMER = auto()
    ADAPTIVE = auto()
    SIMPLE_CHUNKING = auto()

class RequestGeneratorType(BaseEnum):
    SYNTHETIC = auto()
    TRACE = auto()

class RequestIntervalGeneratorType(BaseEnum):
    POISSON = auto()
    GAMMA = auto()
    STATIC = auto()
    TRACE = auto()

class RequestLengthGeneratorType(BaseEnum):
    UNIFORM = auto()
    ZIPF = auto()
    TRACE = auto()
    FIXED = auto()

class AttentionBackend(BaseEnum):
    FLASHINFER = auto()
    NO_OP = auto()

@dataclass
class GPUInfo:
    node_ip: Optional[str]
    gpu_id: int

    def to_tuple(self) -> GPULocation:
        return GPULocation((self.node_ip, self.gpu_id))

    @classmethod
    def from_tuple(cls, location: GPULocation) -> 'GPUInfo':
        return cls(node_ip=location[0], gpu_id=location[1])

def create_resource_mapping(gpu_infos: List[GPUInfo]) -> ResourceMapping:
    return ResourceMapping([gpu_info.to_tuple() for gpu_info in gpu_infos])

def create_replica_resource_mapping(resource_mappings: List[ResourceMapping]) -> ReplicaResourceMapping:
    return ReplicaResourceMapping(resource_mappings)

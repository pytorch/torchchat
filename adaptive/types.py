from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

GPULocation = Tuple[Optional[str], int]  # (node_ip, gpu_id)
ResourceMapping = List[GPULocation]
ReplicaResourceMapping = List[ResourceMapping]  # List ResourceMapping for each replica


class SchedulerType(Enum):
    VLLM = "VLLM"
    ORCA = "ORCA"
    FASTER_TRANSFORMER = "FASTER_TRANSFORMER"
    ADAPTIVE = "Adaptive"
    SIMPLE_CHUNKING = "SIMPLE_CHUNKING"


class RequestGeneratorType(Enum):
    SYNTHETIC = "SYNTHETIC"
    TRACE = "TRACE"


class RequestIntervalGeneratorType(Enum):
    POISSON = "POISSON"
    GAMMA = "GAMMA"
    STATIC = "STATIC"
    TRACE = "TRACE"


class RequestLengthGeneratorType(Enum):
    UNIFORM = "UNIFORM"
    ZIPF = "ZIPF"
    TRACE = "TRACE"
    FIXED = "FIXED"


class AttentionBackend(Enum):
    FLASHINFER = "FLASHINFER"
    NO_OP = "NO_OP"

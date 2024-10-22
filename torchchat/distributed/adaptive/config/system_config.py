from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Type, Union

from sarathi.config.base_poly_config import BasePolyConfig
from sarathi.config.flat_dataclass import create_flat_dataclass
from sarathi.logger import init_logger
from sarathi.transformers_utils.config import get_config
from sarathi.types import AttentionBackend, ResourceMapping, SchedulerType
from sarathi.utils.hf_utils import get_and_verify_dtype, get_and_verify_max_len

logger = init_logger(__name__)


class LoadFormat(str, Enum):
    """Supported model loading formats."""
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"


@dataclass
class ModelConfig:
    """Configuration for the model and its loading parameters."""
    
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Name or path of the huggingface model to use."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code when downloading model and tokenizer."},
    )
    download_dir: Optional[Path] = field(
        default=None,
        metadata={"help": "Directory for model weights, defaults to huggingface cache."},
    )
    load_format: LoadFormat = field(
        default=LoadFormat.AUTO,
        metadata={"help": "Format of model weights to load."},
    )
    dtype: str = field(
        default="float16",
        metadata={"help": "Data type for weights and activations."},
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed for reproducibility."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Specific model version to use."},
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length (derived from model if None)."},
    )

    def __post_init__(self) -> None:
        """Initialize and validate the configuration."""
        if isinstance(self.download_dir, str):
            self.download_dir = Path(self.download_dir)
            
        if isinstance(self.load_format, str):
            try:
                self.load_format = LoadFormat(self.load_format.lower())
            except ValueError as e:
                raise ValueError(
                    f"Invalid load format: {self.load_format}. "
                    f"Must be one of {', '.join(f.value for f in LoadFormat)}"
                ) from e

        self.hf_config = get_config(self.model, self.trust_remote_code, self.revision)
        self.dtype = get_and_verify_dtype(self.hf_config, self.dtype)
        self.hf_config.dtype = self.dtype
        self.max_model_len = get_and_verify_max_len(self.hf_config, self.max_model_len)

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        """Verify compatibility with parallel configuration."""
        self._verify_attention_heads(parallel_config.tensor_parallel_size)
        self._verify_hidden_layers(parallel_config.pipeline_parallel_size)

    def _verify_attention_heads(self, tensor_parallel_size: int) -> None:
        """Verify attention heads are compatible with tensor parallelism."""
        total_heads = self.hf_config.num_attention_heads
        if total_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total attention heads ({total_heads}) must be divisible by "
                f"tensor parallel size ({tensor_parallel_size})."
            )

    def _verify_hidden_layers(self, pipeline_parallel_size: int) -> None:
        """Verify hidden layers are compatible with pipeline parallelism."""
        total_layers = self.hf_config.num_hidden_layers
        if total_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total hidden layers ({total_layers}) must be divisible by "
                f"pipeline parallel size ({pipeline_parallel_size})."
            )

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Calculate number of key/value heads accounting for parallelism."""
        tensor_parallel_size = parallel_config.tensor_parallel_size
        
        if self._is_multi_query_attention():
            return 1
            
        for attr in ["n_head_kv", "num_kv_heads", "num_key_value_heads"]:
            if hasattr(self.hf_config, attr):
                return getattr(self.hf_config, attr) // tensor_parallel_size

        return self.hf_config.num_attention_heads // tensor_parallel_size

    def _is_multi_query_attention(self) -> bool:
        """Check if model uses multi-query attention architecture."""
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        return (
            not new_decoder_arch_falcon 
            and getattr(self.hf_config, "multi_query", False)
        )

    @property
    def hidden_size(self) -> int:
        """Get model hidden size."""
        return self.hf_config.hidden_size

    @property
    def head_size(self) -> int:
        """Calculate size per attention head."""
        return self.hidden_size // self.hf_config.num_attention_heads

    def get_num_q_heads(self, parallel_config: "ParallelConfig") -> int:
        """Calculate number of query heads accounting for parallelism."""
        if not hasattr(self.hf_config, "num_attention_heads"):
            raise ValueError("num_attention_heads not defined in model config")
        return self.hf_config.num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        """Calculate number of layers per pipeline stage."""
        return self.hf_config.num_hidden_layers // parallel_config.pipeline_parallel_size

    @property
    def total_num_layers(self) -> int:
        """Get total number of model layers."""
        return self.hf_config.num_hidden_layers


@dataclass
class CacheConfig:
    """Configuration for model caching behavior."""

    block_size: int = field(
        default=16,
        metadata={"help": "Size of cache block in tokens."},
    )
    num_gpu_blocks: Optional[int] = field(
        default=None,
        metadata={"help": "Number of GPU cache blocks (set after profiling)."},
    )

    def __post_init__(self) -> None:
        """Validate cache configuration."""
        if self.block_size <= 0:
            raise ValueError(f"Block size must be positive, got {self.block_size}")


@dataclass
class ParallelConfig:
    """Configuration for model parallelization."""

    pipeline_parallel_size: int = field(
        default=2,
        metadata={"help": "Number of pipeline parallel groups."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel groups."},
    )

    def __post_init__(self) -> None:
        """Calculate total world size and validate configuration."""
        if self.pipeline_parallel_size <= 0 or self.tensor_parallel_size <= 0:
            raise ValueError("Parallel sizes must be positive integers")
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size


@dataclass
class BaseSchedulerConfig(BasePolyConfig):
    """Base configuration for scheduling strategies."""

    max_num_seqs: int = field(
        default=128,
        metadata={"help": "Maximum sequences per iteration (batch size)."},
    )

    def __post_init__(self) -> None:
        """Validate base scheduler configuration."""
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")

    @abstractmethod
    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        """Calculate maximum number of tokens that can be batched."""
        pass


@dataclass
class VllmSchedulerConfig(BaseSchedulerConfig):
    """Configuration for VLLM scheduler."""

    max_batched_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of batched tokens."},
    )

    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        """Calculate maximum batched tokens, considering model length."""
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type() -> SchedulerType:
        """Get scheduler type identifier."""
        return SchedulerType.VLLM


@dataclass
class ChunkingSchedulerConfig(BaseSchedulerConfig):
    """Base configuration for chunk-based schedulers."""

    chunk_size: int = field(
        default=512,
        metadata={"help": "Size of each processing chunk."},
    )

    def __post_init__(self) -> None:
        """Validate chunking configuration."""
        super().__post_init__()
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")


@dataclass
class SimpleChunkingSchedulerConfig(ChunkingSchedulerConfig):
    """Configuration for simple chunking scheduler."""

    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        """Return chunk size as maximum batched tokens."""
        return self.chunk_size

    @staticmethod
    def get_type() -> SchedulerType:
        """Get scheduler type identifier."""
        return SchedulerType.SIMPLE_CHUNKING


@dataclass
class OrcaSchedulerConfig(BaseSchedulerConfig):
    """Configuration for Orca scheduler."""

    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        """Calculate maximum batched tokens based on sequences and model length."""
        return self.max_num_seqs * max_model_len

    @staticmethod
    def get_type() -> SchedulerType:
        """Get scheduler type identifier."""
        return SchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseSchedulerConfig):
    """Configuration for FasterTransformer scheduler."""

    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        """Calculate maximum batched tokens based on sequences and model length."""
        return self.max_num_seqs * max_model_len

    @staticmethod
    def get_type() -> SchedulerType:
        """Get scheduler type identifier."""
        return SchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(ChunkingSchedulerConfig):
    """Configuration for Sarathi scheduler with dynamic chunking support."""

    enable_dynamic_chunking_schedule: bool = field(
        default=False,
        metadata={"help": "Enable dynamic chunking schedule."},
    )
    low_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum chunk size for dynamic chunking."},
    )
    high_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum chunk size for dynamic chunking."},
    )
    chunk_schedule_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum tokens for chunk scheduling."},
    )
    chunk_schedule_stages: Optional[int] = field(
        default=None,
        metadata={"help": "Number of stages for chunk scheduling."},
    )

    def __post_init__(self) -> None:
        """Validate Sarathi scheduler configuration."""
        super().__post_init__()
        if self.enable_dynamic_chunking_schedule:
            self._validate_dynamic_chunking()

    def _validate_dynamic_chunking(self) -> None:
        """Validate dynamic chunking parameters."""
        if None in (self.low_chunk_size, self.high_chunk_size,
                   self.chunk_schedule_max_tokens, self.chunk_schedule_stages):
            raise ValueError(
                "All dynamic chunking parameters must be set when enabled"
            )
        if self.low_chunk_size > self.high_chunk_size:
            raise ValueError(
                f"low_chunk_size ({self.low_chunk_size}) must be <= "
                f"high_chunk_size ({self.high_chunk_size})"
            )
        if self.chunk_schedule_stages <= 0:
            raise ValueError(
                f"chunk_schedule_stages must be positive, got {self.chunk_schedule_stages}"
            )

    def get_max_num_batched_tokens(self, max_model_len: int) -> int:
        """Get maximum batched tokens based on chunking configuration."""
        if self.enable_dynamic_chunking_schedule:
            return self.high_chunk_size
        return self.chunk_size

    @staticmethod
    def get_type() -> SchedulerType:
        """Get scheduler type identifier."""
        return SchedulerType.SARATHI


@dataclass
class MetricsConfig:
    """Configuration for metrics collection and reporting."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Enable metrics writing."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep ID."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run ID."},
    )
    enable_op_level_metrics: bool = field(
        default=False,
        metadata={"help": "Enable operation-level metrics."},
    )
    enable_cpu_op_level_metrics: bool = field(
        default=False,
        metadata={"help": "Enable CPU operation-level metrics."},
    )
    enable_chrome_trace: bool = field(
        default=True,
        metadata={"help": "Enable Chrome tracing."},
    )
    enable_request_outputs: bool = field(
        default=False,
        metadata={"help": "Enable request outputs."},
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={"help": "Keep individual batch metrics."},
    )

    def __post_init__(self) -> None:
        """Validate metrics configuration."""
        if self.write_metrics and self.wandb_project:
            self._validate_wandb_config()

    def _validate_wandb_config(self) -> None:
        """Validate Weights & Biases configuration."""
        if self.wandb_sweep_id and not self.wandb_group:
            raise ValueError("wandb_group must be set when using wandb_sweep_id")


@dataclass
class ReplicaConfig:
    """Configuration for replica instances."""

    replica_id: int = field(
        default=0,
        metadata={"help": "ID of the replica."},
    )
    output_dir: Path = field(
        default=Path("."),
        metadata={"help": "Output directory for the replica."},
    )
    resource_mapping: Optional[List[Tuple[Optional[str], int]]] = field(
        default=None,
        metadata={"help": "Resource mapping for the replica."},
    )

    def __post_init__(self) -> None:
        """Initialize and validate replica configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        if self.replica_id < 0:
            raise ValueError(f"replica_id must be non-negative, got {self.replica_id}")
            
        self.output_dir = self.output_dir / f"replica_{self.replica_id}"

    def get_resource_mapping(self, world_size: int) -> List[Tuple[Optional[str], int]]:
        """Get resource mapping, creating default if none exists."""
        if not self.resource_mapping:
            self.resource_mapping = [(None, i) for i in range(world_size)]
        return self.resource_mapping


@dataclass
class WorkerConfig:
    """Configuration for worker processes."""

    gpu_memory_utilization: float = field(
        default=0.8,
        metadata={"help": "GPU memory utilization fraction (0.0 to 1.0)."},
    )
    attention_backend: AttentionBackend = field(
        default=AttentionBackend.FLASHINFER,
        metadata={"help": "Backend for attention computation."},
    )

    def __post_init__(self) -> None:
        """Validate worker configuration."""
        self._verify_gpu_memory_utilization()
        self._verify_attention_backend()

    def _verify_gpu_memory_utilization(self) -> None:
        """Verify GPU memory utilization is within valid range."""
        if not 0.0 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                "GPU memory utilization must be between 0.0 and 1.0. "
                f"Got {self.gpu_memory_utilization}"
            )

    def _verify_attention_backend(self) -> None:
        """Verify attention backend is valid."""
        if isinstance(self.attention_backend, str):
            try:
                self.attention_backend = AttentionBackend[self.attention_backend.upper()]
            except KeyError as e:
                valid_backends = ", ".join(b.name for b in AttentionBackend)
                raise ValueError(
                    f"Invalid attention backend: {self.attention_backend}. "
                    f"Must be one of: {valid_backends}"
                ) from e


@dataclass
class SystemConfig:
    """Complete system configuration combining all components."""

    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig) 
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: BaseSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self) -> None:
        """Validate complete system configuration."""
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self._verify_compatibility()

    def _verify_compatibility(self) -> None:
        """Verify compatibility between different configuration components."""
        # Add any cross-component validation here
        pass


@dataclass
class BaseEndpointConfig(ABC):
    """Base configuration for endpoints."""

    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},
    )
    output_dir: Path = field(
        default=Path("output"),
        metadata={"help": "Output directory."},
    )
    model_config: ModelConfig = field(default_factory=ModelConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: BaseSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self) -> None:
        """Initialize endpoint configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        self.output_dir = self.output_dir / timestamp
        
        self._validate_log_level()

    def _validate_log_level(self) -> None:
        """Validate logging level."""
        valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
        if self.log_level.lower() not in valid_levels:
            raise ValueError(
                f"Invalid log level: {self.log_level}. "
                f"Must be one of: {', '.join(valid_levels)}"
            )

    @classmethod
    def create_from_cli_args(cls) -> 'BaseEndpointConfig':
        """Create configuration from command line arguments."""
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__
        return self.__flat_config__.__dict__

    def create_system_config(self, replica_config: ReplicaConfig) -> SystemConfig:
        """Create system configuration from endpoint configuration."""
        return SystemConfig(
            replica_config=replica_config,
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            metrics_config=self.metrics_config,
            worker_config=self.worker_config,
        )

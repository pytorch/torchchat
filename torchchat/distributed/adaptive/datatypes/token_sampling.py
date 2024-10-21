 from dataclasses import dataclass
from enum import IntEnum, auto
from functools import cached_property
from typing import List, Union, Optional

# Constants
SAMPLING_EPS = 1e-5
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = -1

class SamplingType(IntEnum):
    """Enumeration of sampling types for text generation."""
    GREEDY = auto()
    RANDOM = auto()

@dataclass
class SamplingParams:
    """Parameters for controlling text generation sampling.
    
    Attributes:
        temperature: Controls randomness in sampling. Lower values make output more
            deterministic, higher values make it more random. Zero means greedy sampling.
        top_p: Controls cumulative probability of top tokens to consider.
            Must be in (0, 1]. Default is 1.0 (consider all tokens).
        top_k: Number of top tokens to consider. -1 means consider all tokens.
        stop: Strings that stop generation when produced. Output excludes stop strings.
        ignore_eos: Whether to continue generation after EOS token.
        max_tokens: Maximum tokens to generate per output sequence.
    
    Raises:
        ValueError: If any parameters are invalid.
    """
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    stop: Optional[Union[str, List[str]]] = None
    ignore_eos: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        """Validates parameters after initialization."""
        self.stop = self._normalize_stop_tokens(self.stop)
        self._validate_parameters()
        
        if self.is_greedy_sampling:
            self._validate_greedy_sampling()

    @property
    def is_greedy_sampling(self) -> bool:
        """Returns True if using greedy sampling (temperature near zero)."""
        return self.temperature < SAMPLING_EPS

    @cached_property
    def sampling_type(self) -> SamplingType:
        """Returns the type of sampling being used."""
        return SamplingType.GREEDY if self.is_greedy_sampling else SamplingType.RANDOM

    def _normalize_stop_tokens(self, stop: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalizes stop tokens into a list format.
        
        Args:
            stop: Stop tokens in string or list format.
            
        Returns:
            List of stop tokens.
        """
        if stop is None:
            return []
        if isinstance(stop, str):
            return [stop]
        return list(stop)

    def _validate_parameters(self) -> None:
        """Validates all sampling parameters."""
        if self.temperature < 0.0:
            raise ValueError(f"Temperature must be non-negative, got {self.temperature}")
        
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"Top-p must be in (0, 1], got {self.top_p}")
        
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"Top-k must be -1 (disabled) or at least 1, got {self.top_k}"
            )
        
        if self.max_tokens < 1:
            raise ValueError(f"Max tokens must be at least 1, got {self.max_tokens}")

    def _validate_greedy_sampling(self) -> None:
        """Validates parameters specific to greedy sampling."""
        if self.top_p < 1.0 - SAMPLING_EPS:
            raise ValueError("Top-p must be 1.0 when using greedy sampling")
        
        if self.top_k != -1:
            raise ValueError("Top-k must be -1 when using greedy sampling")

    def __repr__(self) -> str:
        """Returns a string representation of the sampling parameters."""
        return (
            f"{self.__class__.__name__}("
            f"temperature={self.temperature:.2f}, "
            f"top_p={self.top_p:.2f}, "
            f"top_k={self.top_k}, "
            f"stop={self.stop}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens})"
        )

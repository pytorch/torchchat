from dataclasses import dataclass, field
from typing import List, Tuple, Union

from torch import Tensor

BLANK_TOKEN_ID: int = -1
KVCache = Union[Tuple[Tensor, Tensor], Tensor]
from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()


@dataclass
class LogicalTokenBlock:
    """
    A block that stores a contiguous chunk of tokens from left to right.
    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    block_number: int
    block_size: int
    token_ids: List[int] = field(init=False)
    num_tokens: int = field(default=0, init=False)

    def __post_init__(self):
        self.token_ids = [BLANK_TOKEN_ID] * self.block_size

    @property
    def is_empty(self) -> bool:
        return self.num_tokens == 0

    @property
    def num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    @property
    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append_tokens(self, new_token_ids: List[int]) -> None:
        if len(new_token_ids) > self.num_empty_slots:
            raise ValueError(
                f"Cannot append {len(new_token_ids)} tokens. Only {self.num_empty_slots} slots available."
            )

        self.token_ids[self.num_tokens : self.num_tokens + len(new_token_ids)] = (
            new_token_ids
        )
        self.num_tokens += len(new_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.token_ids[: self.num_tokens]

    def get_last_token_id(self) -> int:
        if self.is_empty:
            raise IndexError("Cannot get last token from an empty block.")
        return self.token_ids[self.num_tokens - 1]


@dataclass
class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    block_number: int
    block_size: int
    device: str = field(default="cpu", init=False)

    def __repr__(self) -> str:
        return f"PhysicalTokenBlock(device={self.device}, block_number={self.block_number})"

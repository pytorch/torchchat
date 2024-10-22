from dataclasses import dataclass
from typing import List, Optional

from torchchat.distributed.adaptive.datatypes.sequence import Sequence
from torchchat.distributed.adaptive.datatypes.sequence_status import SequenceStatus


@dataclass
class RequestOutput:
    """
    The output data of a request to the Language Model.

    Attributes:
        seq_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        text: The generated output text.
        token_ids: The token IDs of the generated output.
        finished: Whether the whole request is finished.
        finish_reason: The reason for finishing, if applicable.
    """

    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    text: str
    token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None

    @classmethod
    def from_sequence(cls, seq: Sequence) -> "RequestOutput":
        """
        Create a RequestOutput instance from a Sequence object.

        Args:
            seq: The Sequence object to convert.

        Returns:
            A new RequestOutput instance.
        """
        return cls(
            seq_id=seq.seq_id,
            prompt=seq.prompt,
            prompt_token_ids=seq.prompt_token_ids,
            text=seq.output_text,
            token_ids=seq.output_token_ids,
            finished=seq.is_finished(),
            finish_reason=seq.status.finish_reason,
        )

    @property
    def total_tokens(self) -> int:
        """
        Calculate the total number of tokens (prompt + output).

        Returns:
            The total number of tokens.
        """
        return len(self.prompt_token_ids) + len(self.token_ids)

    def __str__(self) -> str:
        return (
            f"RequestOutput(seq_id={self.seq_id}, "
            f"prompt='{self.prompt[:20]}...', "
            f"text='{self.text[:20]}...', "
            f"finished={self.finished}, "
            f"finish_reason={self.finish_reason})"
        )

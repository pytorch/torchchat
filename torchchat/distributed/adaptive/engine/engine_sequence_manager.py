from typing import List, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from sarathi.config import SystemConfig
from sarathi.core.datatypes.sequence import Sequence
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from sarathi.transformers_utils.tokenizer import detokenize_incrementally

class EngineSequenceManager(BaseSequenceManager):
    """
    Manages sequences for the engine, including decoding and token appending.
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: SystemConfig,
    ):
        """
        Initialize the EngineSequenceManager.

        Args:
            tokenizer: The tokenizer to use for decoding.
            config: The system configuration.
        """
        super().__init__(config)
        self.tokenizer = tokenizer

    def _decode_seq(self, seq: Sequence) -> None:
        """
        Decodes the new token for a sequence.

        Args:
            seq: The sequence to decode.
        """
        new_tokens, new_output_text, prefix_offset, read_offset = detokenize_incrementally(
            self.tokenizer,
            all_input_ids=seq.get_token_ids(),
            prev_tokens=seq.tokens,
            prefix_offset=seq.prefix_offset,
            read_offset=seq.read_offset,
            skip_special_tokens=True,
        )

        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)

        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _on_append_token(self, seq: Sequence) -> None:
        """
        Called when a token is appended to a sequence.

        Args:
            seq: The sequence that had a token appended.
        """
        self._decode_seq(seq)

    def _get_block_table(self, seq: Sequence) -> List[int]:
        """
        Get the block table for a sequence.

        Args:
            seq: The sequence to get the block table for.

        Returns:
            An empty list, as this implementation doesn't use block tables.
        """
        return []

    def process_new_sequence(self, seq: Sequence) -> None:
        """
        Process a new sequence.

        Args:
            seq: The new sequence to process.
        """
        # Add any initialization or processing steps for new sequences here
        pass

    def update_sequence(self, seq: Sequence) -> None:
        """
        Update an existing sequence.

        Args:
            seq: The sequence to update.
        """
        # Add any update logic for existing sequences here
        self._decode_seq(seq)

    def __repr__(self) -> str:
        return f"EngineSequenceManager(tokenizer={self.tokenizer.__class__.__name__})"

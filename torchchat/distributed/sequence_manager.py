from abc import ABC, abstractmethod

import torch


class BaseSequenceManager(ABC):
    def __init__(self, config=None):
        self.seq_dict = {}  # (str, Seq)

    def add_sequence(self, seq):
        self.seq_dict[seq.id] = seq

    def _free_seq(self, seq_id):
        if seq_id in self.seq_dict:
            del self.seq_dict[seq_id]
        else:
            raise ValueError(f"Sequence {seq_id} not found.")


class SequenceManager(BaseSequenceManager):
    def __init__(self, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config

    def __decode_sequence(self, sequence):
        """Decode a sequence of tokens into a string."""
        new_tokens, new_text, prefix_offset, read_offset = detokenize_chunk(
            self.tokenizer,
            all_input_ids=sequence.get_token_ids(),
            prev_tokens=seq.tokens,
            prefix_offset=seq.prefix_offset,
            read_offset=seq.read_offset,
            skip_special_tokens=True,
        )
        if not seq.tokens:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text = new_text

    def _on_append_token(self, seq):
        self.__decode_sequence(seq)

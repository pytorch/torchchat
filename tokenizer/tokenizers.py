# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard
from typing import List
import json

# Third Party
from tokenizers import Tokenizer

# Local
from .base import TokenizerBase


class TokenizersTokenizer(TokenizerBase):
    """
    Wrapper around the `tokenizers` library for API compatibility
    """

    def __init__(self, file_path: str):
        self._tokenizer = Tokenizer.from_file(file_path)
        # The BOS and EOS tokens are not easily visible from the tokenizer
        # object itself, so we extract them at construction with a sample call
        self._bos_token = self._tokenizer.encode("Test", add_special_tokens=True).ids[0]
        # There is no explicit BOS token in many tokenizers, so we look for a
        # single special token that most resembles the BOS token.
        self._eos_token = None
        tok_content = json.loads(self._tokenizer.to_str())
        end_toks = [
            tok for tok in tok_content['added_tokens']
            if tok["special"] and "end" in tok["content"]
        ]
        assert end_toks, "Unable to find an EOS token in the added tokens"
        if len(end_toks) > 1:
            end_text_toks = [
                tok for tok in end_toks if "text" in tok["content"]
            ]
            if len(end_text_toks) == 1:
                self._eos_token = end_text_toks[0]["id"]
        assert self._eos_token is not None, "Unable to find an EOS token in the added tokens"

    def encode(
        self,
        s: str,
        *,
        bos: bool = False,
        eos: bool = False,
    ) -> List[int]:
        res = self._tokenizer.encode(s, add_special_tokens=bos).ids
        if eos and (not res or res[-1] != self._eos_token):
            res.append(self._eos_token)
        return res

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)

    def bos_id(self) -> int:
        return self._bos_token

    def eos_id(self) -> int:
        return self._eos_token

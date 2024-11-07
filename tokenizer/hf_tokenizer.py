# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard
from typing import Dict, List, Optional
import json
import os

# Third Party
import jinja2
from tokenizers import Tokenizer

# Local
from .base import TokenizerBase


class HFTokenizer(TokenizerBase):
    """
    Wrapper around the Huggingface `tokenizers` library for API compatibility
    """

    def __init__(self, file_path: str):
        # If the path is a directory, look for "tokenizer.json" which is
        # standard for transformers checkpoints and also look for the
        # "tokenizer_config.json" file to parse eos/bos tokens
        if os.path.isdir(file_path):
            tokenizer_path = os.path.join(file_path, "tokenizer.json")
            tokenizer_config_path = os.path.join(file_path, "tokenizer_config.json")
        else:
            tokenizer_path = file_path
            tokenizer_config_path = os.path.join(os.path.dirname(file_path), "tokenizer_config.json")
        if not os.path.isfile(tokenizer_path):
            tokenizer_config_path = None

        # Load the tokenizer itself
        self._tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load the chat template if we have a config path
        self._chat_template: Optional[jinja2.Template] = None

        # If available, parse bos/eos tokens from the tokenizer config
        self._bos_id, self._eos_id = None, None
        if tokenizer_config_path is not None:
            with open(tokenizer_config_path, "r") as handle:
                tok_config = json.load(handle)
            bos_token = tok_config.get("bos_token")
            eos_token = tok_config.get("eos_token")
            if bos_token is not None:
                self._bos_id = self._tokenizer.token_to_id(bos_token)
            if eos_token is not None:
                self._eos_id = self._tokenizer.token_to_id(eos_token)
            if chat_template_str := tok_config.get("chat_template"):
                self._chat_template = jinja2.Template(chat_template_str)

        # If no eos/bos tokens found, go looking for them!
        if None in [self._bos_id, self._eos_id]:
            tok_content = json.loads(self._tokenizer.to_str())
            if self._bos_id is None:
                self._bos_id = self._look_for_special_token(tok_content, ["begin", "text"])
            if self._eos_id is None:
                self._eos_id = self._look_for_special_token(tok_content, ["end", "text"])

        assert None not in [self._bos_id, self._eos_id], "Unable to find an BOS/EOS tokens"

    @staticmethod
    def _look_for_special_token(added_tokens: dict, search_strs: List[str]) -> Optional[int]:
        candidate_toks = added_tokens
        for search_str in search_strs:
            candidate_toks = [
                tok for tok in candidate_toks
                if tok["special"] and search_str in tok["content"]
            ]
            if len(candidate_toks) == 1:
                return candidate_toks[0]["id"]

    ## Interface ##

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
        return self._bos_id

    def eos_id(self) -> int:
        return self._eos_id

    ## Additional Public Methods ##

    def has_chat_template(self) -> bool:
        return bool(self._chat_template)

    def apply_chat_template(
        self,
        dialog: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """If configured with a chat template, apply it to the list of messages
        """
        if not self._chat_template:
            raise ValueError("No chat template configured!")
        return self._chat_template.render(
            messages=dialog, add_generation_prompt=add_generation_prompt
        )

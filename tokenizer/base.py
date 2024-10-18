# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Abstract base class for all tokenizer classes in python matching c++ interface.
"""

# Standard
from abc import ABC, abstractmethod
from typing import List


class TokenizerBase(ABC):
    __doc__ = __doc__

    @abstractmethod
    def encode(self, s: str, *, bos: bool = False, eos: bool = False) -> List[int]:
        """Encode the given string and optionally include bos/eos tokens"""

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode the given token ids into a string"""

    @abstractmethod
    def bos_id(self) -> int:
        """The id of the begin-of-string token"""

    @abstractmethod
    def eos_id(self) -> int:
        """The id of the end-of-string token"""

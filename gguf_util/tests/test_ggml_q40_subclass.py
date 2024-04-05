# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from ..ggml_quantization_type import Q4_0


class TestGGMLQ40TensorSubclass(unittest.TestCase):
    def test_from_float_and_to_float(self) -> None:
        weight = torch.randn([1, 32], dtype=torch.float16)

        packed = Q4_0.from_float(weight)

        self.assertEqual(packed.dtype, torch.uint8)
        # expected size = (1 * sizeof(uint16_t)) + (32 * sizeof(uint8_t)/2) = 18
        self.assertEqual(packed.numel(), 18)

        unpacked = Q4_0.to_float(packed).reshape(weight.shape)

        tolerance = (torch.max(weight) - torch.min(weight)) / 16

        self.assertTrue(torch.allclose(weight, unpacked, atol=tolerance.item()))

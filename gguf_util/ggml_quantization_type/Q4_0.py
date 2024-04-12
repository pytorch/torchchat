# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ggml q4_0 tensor layout
# ┌─────────────────────┬───────────────────────────┐
# │                     │                           │
# │                     │                           │
# │  2 bytes (1xfp16)   │    16 bytes (32xint4)     │
# │  group-wise scale   │    group-wise weights     │
# │                     │                           │
# │                     │                           │
# └─────────────────────┴───────────────────────────┘
#
# Notice that the 16 bytes (32 int4) are interleved:
# [0th value, 16th value, 1st value, 17th value, ..., 15th, 31st]
#

import torch
import gguf


def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor, dtype: torch.dtype = torch.float16):
    """
    Unpacks GGUF Q4_0 matrix to int32_data, scales, and zeros.

    Inputs:
    * gguf_tensor (gguf.gguf_reader.ReaderTensor): a Q4_0 gguf matrix with shape (nr, nc).
    * dtype (torch.dtype): float type to cast scales and zeros to.

    Returns:

    * int32_data: torch.Tensor
        A torch.int32 matrix with values in the uint4 range [0, 15], with the same shape as gguf_tensor (nr, nc).
    * scales: torch.Tensor
        A tensor with scales.  It has shape (nr, nc / 32).  Note 32 is the group_size.
    * zeros: torch.Tensor
        A tensor with zeros.  It has shape (nr, nc / 32).  Note 32 is the group_size.
        Also note that Q4_0 is a scale transformation and the zeros are not data-dependent.
    """
    assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0
    assert len(gguf_tensor.shape) == 2
    nr, nc = gguf_tensor.shape # number rows/cols in matrix

    packed = torch.from_numpy(gguf_tensor.data.reshape(-1, 18))
    assert packed.dtype == torch.uint8
    ng = packed.shape[0] # number of groups
    group_size = 32
    assert ng * group_size == nr * nc # sanity check


    # Extract scales (first 2 bytes of packed)
    scales = torch.tensor(packed[:,:2].contiguous().untyped_storage(), dtype=torch.float16).reshape(ng, 1)
    assert scales.dtype == torch.float16

    # Define zero-points
    zeros = 8 * torch.ones(ng, 1).to(torch.float16)

    # Extract quantized int values (last 16 bytes of packed)

    # De-interleave
    int32_data = packed[:,2:]
    int32_data = torch.cat([int32_data[:, ::2], int32_data[:, 1::2]], dim=1).contiguous()
    assert int32_data.dtype == torch.uint8
    assert int32_data.shape == (ng, 16)

    # Each of the 16 uint8 represent 32 uint4
    low = (int32_data & 0b1111).to(torch.int32)
    high = (int32_data >> 4).to(torch.int32)
    int32_data = torch.stack([low, high], dim=2).reshape(ng, 32)
    assert int32_data.dtype == torch.int32
    assert int32_data.min().item() == 0
    assert int32_data.max().item() == 15
    assert int32_data.shape == (ng, 32)

    # Reshape int_data, scales, and zeros to right size
    int32_data = int32_data.reshape(nr, nc)
    scales = scales.reshape(nr, -1)
    assert scales.shape == (nr, nc / group_size)

    zeros = zeros.reshape(nr, -1)
    assert zeros.shape == (nr, nc / group_size)

    return int32_data, scales.to(dtype), zeros.to(dtype)

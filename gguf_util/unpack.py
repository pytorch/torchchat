# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import gguf

def unpack_q40(gguf_tensor: gguf.gguf_reader.ReaderTensor, dtype: torch.dtype = torch.float16):
    """
    Unpacks GGUF Q4_0 matrix to int32_data, scales, and zeros.

    See https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h for definition of block_q4_0:

    #define QK4_0 32
    typedef struct {
        ggml_half d;           // delta
        uint8_t qs[QK4_0 / 2]; // nibbles / quants
    } block_q4_0;

    Also see dequantize_row_q4_0 in https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
    for how the block should be interpreted.
    """

    assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0
    assert len(gguf_tensor.shape) == 2
    QK4_0 = 32 # groupsize

    # Parse block_q4_0
    block_q4_0_size = int(2 + QK4_0 / 2)
    packed = torch.from_numpy(gguf_tensor.data.reshape(-1, block_q4_0_size))
    assert packed.dtype == torch.uint8
    ng = packed.shape[0] # number of groups/blocks

    curr = 0
    size = 2 # half size
    d = packed[:,curr:(curr+size)].contiguous()
    d = torch.tensor(d.untyped_storage(), dtype=torch.float16).reshape(ng, 1)
    curr += size

    size = int(QK4_0 / 2)
    qs = packed[:,curr:(curr+size)].contiguous()
    curr += size

    # Check we finished parsing
    assert curr == block_q4_0_size


    # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract the
    # zero-points
    x0 = qs & 0x0F
    x1 = qs >> 4

    int32_data = torch.cat([x0, x1], dim=1).to(torch.int32).reshape(ng, QK4_0)
    assert int32_data.dtype == torch.int32
    assert int32_data.min().item() == 0
    assert int32_data.max().item() == 2**4-1
    assert int32_data.shape == (ng, QK4_0)


    scales = d
    zeros = 8 * torch.ones(scales.shape).to(torch.float16)

    return int32_data, scales, zeros


def unpack_q6k(gguf_tensor: gguf.gguf_reader.ReaderTensor):
    """
    Unpacks GGUF Q6_k matrix to int32_data, scales, and zeros.

    See https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h for definition of block_q6_K:

    // 6-bit quantization
    // weight is represented as x = a * q
    // 16 blocks of 16 elements each
    // Effectively 6.5625 bits per weight
    typedef struct {
        uint8_t ql[QK_K/2];      // quants, lower 4 bits
        uint8_t qh[QK_K/4];      // quants, upper 2 bits
        int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
        ggml_half d;             // super-block scale
    } block_q6_K;

    QK_K is 64 or 256 by compile flag.

    Also see dequantize_row_q6_K in https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
    for how this block should be interpreted.
    """
    assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
    assert len(gguf_tensor.shape) == 2
    QK_K = 64 # TODO: QK_K = 256


    # Parse block_q6_K
    block_q6_K_size = int(QK_K/2 + QK_K/4 + QK_K/16 + 2 + 2) # TODO: check if there is padding.  Seem to be off by 2 bytes
    packed = torch.from_numpy(gguf_tensor.data.reshape(-1, block_q6_K_size))
    assert packed.dtype == torch.uint8
    ng = packed.shape[0] # number of groups/blocks

    curr = 0
    curr += 2 # TODO: look into missing bytes

    size = int(QK_K/2)
    ql = packed[:,curr:(curr+size)].contiguous()
    assert ql.shape == (ng, 32)
    curr += size

    size = int(QK_K/4)
    qh = packed[:,curr:(curr+size)].contiguous()
    assert qh.shape == (ng, 16)
    curr += size

    size = int(QK_K/16)
    scales = packed[:,curr:(curr+size)].contiguous()
    scales = torch.tensor(scales.untyped_storage(), dtype=torch.int8).reshape(ng, int(QK_K/16))
    curr += size

    size = 2 # half size
    d = packed[:,curr:(curr+size)].contiguous()
    d = torch.tensor(d.untyped_storage(), dtype=torch.float16).reshape(ng, 1)
    curr += size

    # Check we finished parsing
    assert curr == block_q6_K_size

    # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract the
    # zero-points
    q1 = ((ql[:,0:16] & 0xF) | (((qh[:,0:16] >> 0) & 3) << 4))
    q2 = ((ql[:,16:32] & 0xF) | (((qh[:,0:16] >> 2) & 3) << 4))
    q3 = ((ql[:,0:16] >> 4) | (((qh[:,0:16] >> 4) & 3) << 4))
    q4 = ((ql[:,16:32] >> 4) | (((qh[:,0:16] >> 6) & 3) << 4))
    int32_data = torch.cat([q1, q2, q3, q4], dim=1).to(torch.int32)
    assert int32_data.shape == (ng, QK_K)
    assert int32_data.min().item() == 0
    assert int32_data.max().item() == 2**6-1

    # scales = d * scales
    zeros = 32 * torch.ones(scales.shape)

    return int32_data, d, scales, zeros

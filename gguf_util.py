# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import gguf

class F16:
    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF F16 tensor.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.F16
        reversed_shape = gguf_tensor.shape[::-1] # TODO: GGUF tensors are reversed
        new_tensor = gguf_tensor.data.reshape(reversed_shape)
        return torch.from_numpy(new_tensor).to(torch.float16)

class F32:
    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF F32 tensor.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.F32
        reversed_shape = gguf_tensor.shape[::-1] # TODO: GGUF tensors are reversed
        new_tensor = gguf_tensor.data.reshape(reversed_shape)
        return torch.from_numpy(new_tensor).to(torch.float32)

class Q4_0:
    group_size = 32
    n_bit = 4

    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF Q4_0 matrix of size (nr, nc) to q, s, and z that can be dequantized by:

        x = s(q - 8) + z (roughly, reshape is needed),

        where
        * q is an int4-valued tensor of shape (nr, nc) and type torch.int32
        * s is a torch.float32 tensor of shape (nr, -1) with one scale per group
        * z is a torch.float32 tensor of shape (nr, -1) with one zero per group

        Note that z is always zero because Q4_0 is a scale-only scheme.

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
        nc, nr = gguf_tensor.shape # TODO: CHECK THIS.  GGUF TENSOR REVERSED?

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

        # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract 16
        x0 = qs & 0x0F
        x1 = qs >> 4

        int32_data = torch.cat([x0, x1], dim=1).to(torch.int32).reshape(ng, QK4_0)
        assert int32_data.dtype == torch.int32
        assert int32_data.min().item() >= 0
        assert int32_data.max().item() <= 2**4-1
        assert int32_data.shape == (ng, QK4_0)

        # Prepare for return
        q = int32_data.to(torch.int32).reshape(nr, nc)
        s = d.to(torch.float32).reshape(nr, -1)
        z = torch.zeros(s.shape).to(torch.float32)
        return q, s, z


class Q6_K:
    group_size = 16
    n_bit = 6

    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF Q6_k matrix of size (nr, nc) to q, s, and z that can be dequantized by:

        x = s(q - 32) + z (roughly, reshape is needed),

        where
        * q is an int6-valued tensor of shape (nr, nc) and type torch.int32
        * s is a torch.float32 tensor of shape (nr, -1) with one scale per group
        * z is a torch.float32 tensor of shape (nr, -1) with one zero per group

        There is one element of s/z per group of 32 elements of 4.

        Note that z is always zero because Q6_k is a scale-only scheme.

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

        QK_K is 64 or 256 by compile flag in the code, but in GGUF it looks like only the 256-variant
        is used, see "import gguf; gguf.GGML_QUANT_SIZES".

        Also see dequantize_row_q6_K in https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
        for how this block should be interpreted.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
        assert len(gguf_tensor.shape) == 2
        nc, nr = gguf_tensor.shape # TODO: CHECK THIS.  GGUF TENSOR REVERSED?
        QK_K = 256

        # Parse block_q6_K
        block_q6_K_size = int(QK_K/2 + QK_K/4 + QK_K/16 + 2)
        packed = torch.from_numpy(gguf_tensor.data.reshape(-1, block_q6_K_size))
        assert packed.dtype == torch.uint8
        ng = packed.shape[0] # number of groups/blocks

        curr = 0

        size = int(QK_K/2)
        ql = packed[:,curr:(curr+size)].contiguous()
        assert ql.shape == (ng, 128)
        curr += size

        size = int(QK_K/4)
        qh = packed[:,curr:(curr+size)].contiguous()
        assert qh.shape == (ng, 64)
        curr += size

        size = int(QK_K/16)
        scales = packed[:,curr:(curr+size)].contiguous()
        scales = torch.tensor(scales.untyped_storage(), dtype=torch.int8).reshape(ng, int(QK_K/16)).to(torch.float32)
        curr += size

        size = 2 # half size
        d = packed[:,curr:(curr+size)].contiguous()
        d = torch.tensor(d.untyped_storage(), dtype=torch.float16).reshape(ng, 1).to(torch.float32)
        curr += size

        # Check we finished parsing
        assert curr == block_q6_K_size

        # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract 32
        q1 = ((ql[:,0:32] & 0xF) | (((qh[:,0:32] >> 0) & 3) << 4))
        q2 = ((ql[:,32:64] & 0xF) | (((qh[:,0:32] >> 2) & 3) << 4))
        q3 = ((ql[:,0:32] >> 4) | (((qh[:,0:32] >> 4) & 3) << 4))
        q4 = ((ql[:,32:64] >> 4) | (((qh[:,0:32] >> 6) & 3) << 4))

        q5 = ((ql[:,64:96] & 0xF) | (((qh[:,32:64] >> 0) & 3) << 4))
        q6 = ((ql[:,96:128] & 0xF) | (((qh[:,32:64] >> 2) & 3) << 4))
        q7 = ((ql[:,64:96] >> 4) | (((qh[:,32:64] >> 4) & 3) << 4))
        q8 = ((ql[:,96:128] >> 4) | (((qh[:,32:64] >> 6) & 3) << 4))

        q = torch.cat([q1, q2, q3, q4, q5, q6, q7, q8], dim=1).to(torch.int32)
        assert q.shape == (ng, QK_K)
        assert q.min().item() >= 0
        assert q.max().item() <= 2**6-1

        # Unpack scales
        s1 = d * torch.cat([scales[:,0].reshape(-1,1), scales[:,1].reshape(-1,1)], dim=1)
        s2 = d * torch.cat([scales[:,2].reshape(-1,1), scales[:,3].reshape(-1,1)], dim=1)
        s3 = d * torch.cat([scales[:,4].reshape(-1,1), scales[:,5].reshape(-1,1)], dim=1)
        s4 = d * torch.cat([scales[:,6].reshape(-1,1), scales[:,7].reshape(-1,1)], dim=1)

        s5 = d * torch.cat([scales[:,8].reshape(-1,1), scales[:,9].reshape(-1,1)], dim=1)
        s6 = d * torch.cat([scales[:,10].reshape(-1,1), scales[:,11].reshape(-1,1)], dim=1)
        s7 = d * torch.cat([scales[:,12].reshape(-1,1), scales[:,13].reshape(-1,1)], dim=1)
        s8 = d * torch.cat([scales[:,14].reshape(-1,1), scales[:,15].reshape(-1,1)], dim=1)
        s = torch.cat([s1, s2, s3, s4, s5, s6, s7, s8], dim=1)
        assert s.shape == (ng, 16)

        # Prepare for return
        q = q.to(torch.int32).reshape(nr, nc)
        s = s.reshape(nr, -1)
        z = torch.zeros(s.shape).to(torch.float32)

        return q, s, z

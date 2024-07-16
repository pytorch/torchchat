# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from build.model import apply_rotary_emb, Attention

# from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache
from torch import nn


class CustomKVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype):
        super().__init__()

        dtype = torch.float

        # This is flipped around from what is in build.model's KVCache
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(self, input_pos, k_val, v_val):
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.float()
        v_out[:, :, input_pos] = v_val.float()

        return k_out, v_out


class CustomSDPAAttention(nn.Module):
    def __init__(self, attention: Attention):
        super().__init__()

        self.wq = attention.wq
        self.wk = attention.wk
        self.wv = attention.wv

        self.wo = attention.wo

        max_batch_size, n_heads, max_seq_length, head_dim = (
            attention.kv_cache.k_cache.shape
        )
        cache_dtype = attention.kv_cache.k_cache.dtype
        self.kv_cache = CustomKVCache(
            max_batch_size, max_seq_length, n_heads, head_dim, cache_dtype
        )

        self.n_heads = attention.n_heads
        self.head_dim = attention.head_dim
        self.n_local_heads = attention.n_local_heads
        self.dim = attention.dim

    def forward(self, x, freqs_cis, mask, input_pos=None):
        bsz, seqlen, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis).to(dtype=torch.float)
        k = apply_rotary_emb(k, freqs_cis).to(dtype=torch.float)
        v = v.to(dtype=torch.float)

        # KV cache should always be enabled
        assert self.kv_cache is not None
        output = torch.ops.llama.sdpa_with_kv_cache(
            q,
            k,
            v,
            self.kv_cache.k_cache,
            self.kv_cache.v_cache,
            input_pos[-1].item(),
            seqlen,
        )
        output = output.view(bsz, seqlen, self.dim).to(dtype=q.dtype)
        return self.wo(output)


def replace_attention_with_custom_sdpa_attention(module: nn.Module):
    from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache  # noqa

    for name, child in module.named_children():
        if isinstance(child, Attention):
            setattr(module, name, CustomSDPAAttention(child))
        else:
            replace_attention_with_custom_sdpa_attention(child)

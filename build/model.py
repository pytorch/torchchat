# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F

from build.utils import find_multiple, get_precision

config_path = Path(f"{str(Path(__file__).parent)}/known_model_params")


@dataclass
class TransformerArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layers: int = 32
    # n_head in gpt-fast
    n_heads: int = 32
    dim: int = 4096
    # hidden dim is intermediate_size in gpt-fast
    hidden_dim: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = None
    use_tiktoken: bool = False
    max_seq_length: int = 8192
    use_scaled_rope: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_heads
        if self.hidden_dim is None:
            # If hidden_dim is not explicitly set in the TransformerArgs,
            # then calculate implicitly based on dim and
            # also multiple of `args.multiple_of`
            multiple_of = self.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
            self.hidden_dim = find_multiple(hidden_dim, multiple_of)
        self.head_dim = self.dim // self.n_heads
        if isinstance(self.use_tiktoken, str):
            self.use_tiktoken = self.use_tiktoken == "True"

    @classmethod
    def from_params(cls, params_path):
        replace = [("rope_theta", "rope_base"), ("n_kv_heads", "n_local_heads")]
        with open(params_path, "r") as f:
            params = json.loads(f.read())
            # Patch for llama3
            for _from, _to in replace:
                if _from in params:
                    params[_to] = params.pop(_from)
        return cls(**params)

    @classmethod
    def from_table(cls, name: str):
        json_path = config_path / f"{name}.json"
        if json_path.is_file():
            return TransformerArgs.from_params(json_path)
        else:
            known_model_params = [
                config.replace(".json", "") for config in os.listdir(config_path)
            ]
            raise RuntimeError(
                f"unknown table index {name} for transformer config, must be from {known_model_params}"
            )

    @classmethod
    def from_name(cls, name: str):
        json_path = config_path / f"{name}.json"
        if Path(json_path).is_file():
            return TransformerArgs.from_params(json_path)

        known_model_params = [
            config.replace(".json", "") for config in os.listdir(config_path)
        ]

        print(f"known configs: {known_model_params}")
        # Fuzzy search by name (e.g. "7B" and "Mistral-7B")
        config = [
            config
            for config in known_model_params
            if config in str(name).upper() or config in str(name)
        ]

        # We may have two or more configs matched (e.g., "7B" and
        # "Mistral-7B"). Find the best config match:  take longer
        # name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(
                config[1]
            ), name  # make sure only one 'best' match
        elif len(config) == 0:
            raise ValueError(
                f"Unknown model directory name {name}. Must be one of {known_model_params}."
            )

        return TransformerArgs.from_params(config_path / f"{config[0]}.json")


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size,
        max_seq_length,
        n_heads,
        head_dim,
        dtype=None,
    ):
        super().__init__()
        # print(f"dtype on entry {dtype}")
        if not dtype:
            dtype = get_precision()
        # print(f"dtype on get_prec {dtype}")
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = torch.ops.aten.index_put_(self.k_cache, [None, None, input_pos], k_val)
        v_out = torch.ops.aten.index_put_(self.v_cache, [None, None, input_pos], v_val)

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: TransformerArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # self.freqs_cis: Optional[Tensor] = None
        # self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_heads
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads,
            self.config.block_size * 2,
            self.config.rope_base,
            use_scaled = self.config.use_scaled_rope,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)
        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=True)

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for _, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        # print(f"logits shape: {logits.shape}")
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(TransformerArgs.from_name(name))

    @classmethod
    def from_table(cls, name: str):
        return cls(TransformerArgs.from_table(name))

    @classmethod
    def from_params(cls, params_path: str):
        return cls(TransformerArgs.from_params(params_path))

    @classmethod
    def from_gguf(cls, gguf_path: str, **kwargs):
        from build.gguf_loader import load_model_and_state_dict

        model, state_dict = load_model_and_state_dict(gguf_path, **kwargs)
        if state_dict != {}:
            model.load_state_dict(state_dict, assign=True)
        return model


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: TransformerArgs):
        super().__init__()
        assert config.dim % config.n_heads == 0

        # key, query, value projections for all heads, but in a batch
        # total_head_dim = (config.n_heads + 2 * config.n_local_heads) * config.head_dim
        # self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(
            config.dim, config.n_local_heads * config.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.dim, config.n_local_heads * config.head_dim, bias=False
        )

        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        # if prefix + "wq.weight" in state_dict:
        #     wq = state_dict.pop(prefix + "wq.weight")
        #     wk = state_dict.pop(prefix + "wk.weight")
        #     wv = state_dict.pop(prefix + "wv.weight")
        #     state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

        if prefix + "wqkv.weight" in state_dict:
            wqkv = state_dict.pop(prefix + "wqkv.weight")
            q_size = self.n_heads * self.head_dim
            kv_size = self.n_local_heads * self.head_dim
            wq, wk, wv = torch.split(wqkv, (q_size, kv_size, kv_size), dim=0)
            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv

        return

        def _unfuse_wqkv_state_dict(
            state_dict: Dict[str, torch.Tensor],
            dim: int,
        ):
            for key in list(state_dict):
                if key.endswith("wqkv.weight"):
                    tensor = state_dict[key]
                    wq_key = key.replace("wqkv.weight", "wq.weight")
                    state_dict[wq_key] = tensor[:dim]
                    wk_key = key.replace("wqkv.weight", "wk.weight")
                    wv_key = key.replace("wqkv.weight", "wv.weight")
                    wk, wv = tensor[dim:].chunk(2, 0)
                    state_dict[wk_key] = wk
                    state_dict[wv_key] = wv
                    state_dict.pop(key)
                else:
                    continue

        _unfuse_wqkv_state_dict(state_dict, self.dim)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # kv_size = self.n_local_heads * self.head_dim
        # q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: TransformerArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * torch.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    n_elem: int, seq_len: int, base: int = 10000, dtype=None, use_scaled: bool = False
) -> Tensor:
    if not dtype:
        dtype = get_precision()
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ExecuTorch model components
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

try: 
    from executorch.extension.pybindings import portable_lib as exec_lib

    # ET changed the way it's loading the custom ops so it's not included in portable_lib but has to be loaded separately.
    from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache # no-qa

    class PTEModel(nn.Module):
        def __init__(self, config, path) -> None:
            super().__init__()
            self.config = config
            self.model_ = exec_lib._load_for_executorch(str(path))

        def forward(self, x, input_pos):
            # model_.forward expects inputs to be wrapped in a tuple
            forward_inputs = (x.to(torch.long), input_pos.to(torch.long))
            logits = self.model_.forward(forward_inputs)

            # After wrapping in a tuple, we get a list back, so we need to grab
            # the first element to get the tensor
            assert len(logits) == 1
            logits = logits[0]
            return logits

        def setup_caches(self, max_batch_size, max_seq_length):
            pass
except:
    pass

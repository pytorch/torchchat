# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from torch import Tensor
from torch.distributed._tensor import Replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.nn import functional as F

from torchchat.model import (
    apply_rotary_emb,
    KVCache,
    precompute_freqs_cis,
    TransformerArgs,
)

from torchchat.utils.build_utils import find_multiple

config_path = Path(f"{str(Path(__file__).parent)}/known_model_params")


# Use DTensor as output, by default
Colwise = ColwiseParallel(use_local_output=False)
Rowwise = RowwiseParallel(use_local_output=False)

class Transformer(nn.Module):
    def __init__(self, config: TransformerArgs) -> None:
        super().__init__()
        self.config = config
        layers_per_stage = config.n_layers // config.n_stages

        self.tok_embeddings = (
            nn.Embedding(config.vocab_size, config.dim)
            if config.stage_idx == 0 else None
        )

        # Use ModuleDict so that each layer can be assigned its layer ID in the original model
        self.layers = nn.ModuleDict()

        for layer_id in range(
            layers_per_stage * config.stage_idx, layers_per_stage * (config.stage_idx + 1)
        ):
            self.layers[str(layer_id)] = TransformerBlock(config)

        self.norm = (
            RMSNorm(config.dim, eps=config.norm_eps)
            if config.stage_idx == config.n_stages - 1 else None
        )
        self.output = (
            nn.Linear(config.dim, config.vocab_size, bias=False)
            if config.stage_idx == config.n_stages - 1 else None
        )

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
        for b in self.layers.values():
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads,
            self.config.block_size * 2,
            self.config.rope_base,
            use_scaled=self.config.use_scaled_rope,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)
        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=True)

    def distribute(self, device_mesh: DeviceMesh):
        if self.tok_embeddings:
            parallelize_module(
                self.tok_embeddings,
                device_mesh,
                RowwiseParallel(input_layouts=Replicate()),
            )
        for layer in self.layers.values():
            layer.distribute(device_mesh)
        # TODO (kwen2501) : parallelize these
        """
        if self.norm:
            parallelize_module(self.norm, device_mesh, ...)
        if self.output:
            parallelize_module(self.output, device_mesh, ...)
        """

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if input_pos is None:
            input_pos = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]

        if self.tok_embeddings:
            x = self.tok_embeddings(x)

        for _, layer in self.layers.items():
            x = layer(x, input_pos, freqs_cis, mask)

        if self.norm:
            x = self.norm(x)

        if self.output:
            x = self.output(x)

        # print(f"stage output shape: {x.shape}")
        return x

    # temporary disable them due to miss essential input
    # @classmethod
    # def from_name(cls, name: str):
    #     return cls(TransformerArgs.from_name(name))

    # @classmethod
    # def from_table(cls, name: str):
    #     return cls(TransformerArgs.from_table(name))

    # @classmethod
    # def from_params(cls, params_path: str):
    #     return cls(TransformerArgs.from_params(params_path))

    @classmethod
    def from_gguf(cls, gguf_path: str, **kwargs):
        from torchchat.utils.gguf_loader import load_model_and_state_dict

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

    def distribute(self, device_mesh: DeviceMesh):
        self.attention.distribute(device_mesh)
        self.feed_forward.distribute(device_mesh)
        self.ffn_norm.distribute(device_mesh)
        self.attention_norm.distribute(device_mesh)

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
        self.wk = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)
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

    def distribute(self, device_mesh: DeviceMesh):
        parallelize_module(self.wq, device_mesh, Colwise)
        parallelize_module(self.wk, device_mesh, Colwise)
        parallelize_module(self.wv, device_mesh, Colwise)
        parallelize_module(self.wo, device_mesh, Rowwise)

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
        # We use `to_local()` to convert DTensor back to regular Tensor
        q, k, v = q.to_local(), k.to_local(), v.to_local()
        # kv_size = self.n_local_heads * self.head_dim
        # q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, -1, self.head_dim)
        k = k.view(bsz, seqlen, -1, self.head_dim)
        v = v.view(bsz, seqlen, -1, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        # TODO: enable kv cache
        # if self.kv_cache is not None:
        #    k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        y = self.wo(y)
        # TODO: sequence parallelize this
        return y.full_tensor()


class FeedForward(nn.Module):
    def __init__(self, config: TransformerArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def distribute(self, device_mesh: DeviceMesh):
        parallelize_module(self.w1, device_mesh, Colwise)
        parallelize_module(self.w2, device_mesh, Rowwise)
        parallelize_module(self.w3, device_mesh, Colwise)

    def forward(self, x: Tensor) -> Tensor:
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        # y is a DTensor with Partial placement;
        # we convert its placement to Replicate and convert it back to a regular
        # Tensor. `full_tensor` is the API that does both.
        # TODO: sequence parallelize this
        return y.full_tensor()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def distribute(self, device_mesh: DeviceMesh):
        # TODO (kwen2501): parallelize this
        """
        parallelize_module(self.weight, device_mesh, Colwise)
        """
        pass

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

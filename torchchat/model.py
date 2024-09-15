# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import warnings
from abc import ABC, abstractmethod

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from torch import Tensor
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)
from torch.nn import functional as F

from torchtune.models.flamingo import flamingo_decoder, flamingo_vision_encoder
from torchtune.models.llama3_1._component_builders import llama3_1 as llama3_1_builder
from torchtune.modules.model_fusion import DeepFusionModel

from torchchat.utils.build_utils import find_multiple, get_precision

config_path = Path(f"{str(Path(__file__).parent)}/model_params")


def identity(**kwargs):
    if len(kwargs) != 1:
        raise ValueError("Only one argument is expected")
    return list(kwargs.values())[0]


class ModelType(Enum):
    TextOnly = "text_only"
    Llama3_1 = "llama3_1"
    Flamingo = "flamingo"


# Type for objects that can generate nn.Module instance
ModuleLike = Union[nn.Module, Callable[..., nn.Module]]


@dataclass
class ModelRecipe:
    """
    The class describes and contains all supported model structures in torchchat.

    ModelRecipe represents a model as a collection of Transformer modules and a fusion module,
    providing a standardized and centralized way to define and build models in torchchat.
    Attributes:
        model_type (ModelType):
            The type of the model.
        modules (Dict[str, ModuleLike]):
            A dictionary of ModuleLike modules, where each key is the module name and each
            value is a ModuleLike object that generates the transformer.
            The names of the Transformer modules should match the corresponding names in the
            fusion class and the JSON file holding model hyperparameters.
        fusion_class (ModuleLike):
            A ModuleLike object that generates a fusion module by taking the constructed modules above.
    """

    model_type: ModelType
    modules: Dict[str, ModuleLike]
    fusion_class: ModuleLike

    @classmethod
    def _text_only(cls):
        return cls(
            model_type=ModelType.TextOnly,
            modules={"text": Transformer},
            fusion_class=identity,
        )

    @classmethod
    def _llama3_1(cls):
        return cls(
            model_type=ModelType.Llama3_1,
            modules={"text": llama3_1_builder},
            fusion_class=identity,
        )

    @classmethod
    def _flamingo(cls):
        return cls(
            model_type=ModelType.Flamingo,
            modules={"encoder": flamingo_vision_encoder, "decoder": flamingo_decoder},
            fusion_class=DeepFusionModel,
        )

    @classmethod
    def get_recipe(cls, model_type):
        if model_type == ModelType.TextOnly:
            return cls._text_only()
        elif model_type == ModelType.Flamingo:
            return cls._flamingo()
        elif model_type == ModelType.Llama3_1:
            return cls._llama3_1()
        else:
            raise ValueError(f"Can not find the model recipe for {model_type}")


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
    rope_scaling: Optional[Dict[str, Any]] = None
    # For pipeline parallel
    n_stages: int = 1
    stage_idx: int = 0

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
    def from_params(cls, params):
        replace = [("rope_theta", "rope_base"), ("n_kv_heads", "n_local_heads")]
        for _from, _to in replace:
            if _from in params:
                params[_to] = params.pop(_from)
        return cls(**params)


@dataclass
class ModelArgs:
    model_type: ModelType
    transformer_args: Dict[str, Union[Dict, TransformerArgs]]

    def __init__(
        self,
        transformer_args: Union[TransformerArgs, Dict[str, TransformerArgs]],
        model_type: ModelType = ModelType.TextOnly,
    ) -> None:
        self._sanity_check(transformer_args, model_type)

        self.model_type = model_type
        if isinstance(transformer_args, TransformerArgs):
            assert model_type == ModelType.TextOnly
            self.transformer_args = {"text": transformer_args}
        else:
            self.transformer_args = transformer_args

    def _sanity_check(
        self,
        transformer_args: Union[TransformerArgs, Dict[str, TransformerArgs]],
        model_type: ModelType,
    ) -> None:
        assert isinstance(model_type, ModelType)
        assert isinstance(transformer_args, (TransformerArgs, dict))

    @classmethod
    def from_params(cls, params_path):
        with open(params_path, "r") as f:
            loaded_params = json.loads(f.read())

        try:
            # try to interpret as a single transformer config
            transformer_args: Dict[str, TransformerArgs] = {}
            transformer_args["text"] = TransformerArgs.from_params(loaded_params)
            if (model_type := loaded_params.get("model_type", None)) is None:
                model_type = ModelType.TextOnly

        except TypeError:
            # try to interpret as a dict of transformer configs
            model_type = ModelType(loaded_params["model_type"])
            transformer_args = {
                k: v for k, v in loaded_params.items() if k != "model_type"
            }
        return cls(transformer_args, model_type)

    @classmethod
    def from_table(cls, name: str):
        json_path = config_path / f"{name}.json"
        if json_path.is_file():
            return ModelArgs.from_params(json_path)
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
            return ModelArgs.from_params(json_path)

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

        return ModelArgs.from_params(config_path / f"{config[0]}.json")


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


class Model(ABC, nn.Module):
    """
    The entrance for model construction in torchchat.
    """

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model = self.build_model()

    def build_model(self) -> nn.Module:
        """
        Builds a model based on the provided configuration.
        This method retrieves a ModelRecipe instance corresponding to the specified model type,
        constructs the required Transformer modules, and combines them using the fusion class.
        Returns:
            The constructed model instance.
        """
        recipe = ModelRecipe.get_recipe(self.config.model_type)
        modules = {}
        for name, module_class in recipe.modules.items():
            if isinstance(config_args := self.config.transformer_args[name], dict):
                modules[name] = module_class(**config_args)
            else:
                modules[name] = module_class(config_args)

        return recipe.fusion_class(**modules)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method is not implemented")

    @abstractmethod
    def setup_caches(self, *args, **kwargs):
        raise NotImplementedError("setup_caches method is not implemented")

    @classmethod
    def _get_model_instance(cls, config: ModelArgs):
        model_class = MODEL_TYPE_TO_CLASS.get(config.model_type)
        if model_class is None:
            raise ValueError("Unsupported model type:", str(config.model_type))
        return model_class(config)

    @classmethod
    def from_model_args(cls, config: ModelArgs):
        return cls._get_model_instance(config)

    @classmethod
    def from_name(cls, name: str):
        return cls._get_model_instance(ModelArgs.from_name(name))

    @classmethod
    def from_table(cls, name: str):
        return cls._get_model_instance(ModelArgs.from_table(name))

    @classmethod
    def from_params(cls, params_path: str):
        return cls._get_model_instance(ModelArgs.from_params(params_path))

    @classmethod
    def from_gguf(cls, gguf_path: str, **kwargs):
        from torchchat.utils.gguf_loader import load_model_and_state_dict

        model, state_dict = load_model_and_state_dict(gguf_path, **kwargs)
        if state_dict != {}:
            model.load_state_dict(state_dict, assign=True)
        return model


class TextOnlyModel(Model):
    def forward(self, tokens: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        return self.model(tokens, input_pos)

    def setup_caches(self, max_batch_size, max_seq_length):
        self.model.setup_caches(max_batch_size, max_seq_length)


class Llama31Model(Model):
    def forward(self, tokens: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        return self.model(tokens=tokens, input_pos=input_pos)

    def setup_caches(self, max_batch_size, dtype):
        self.model.setup_caches(max_batch_size, dtype=dtype)

    def reset_caches(self):
        self.model.reset_caches()


class FlamingoModel(Model):
    def forward(
        self,
        tokens: Tensor,
        encoder_input: Optional[Dict[str, Tensor]] = None,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if encoder_input is None:
            return self.model(tokens, encoder_mask=encoder_mask)
        return self.model(
            tokens, encoder_input=encoder_input, encoder_mask=encoder_mask
        )

    def setup_caches(self, max_batch_size, dtype):
        self.model.setup_caches(max_batch_size, dtype=dtype)

    def reset_caches(self):
        self.model.reset_caches()


MODEL_TYPE_TO_CLASS = {
    ModelType.TextOnly: TextOnlyModel,
    ModelType.Flamingo: FlamingoModel,
    ModelType.Llama3_1: Llama31Model,
}


class Transformer(nn.Module):
    def __init__(self, config: TransformerArgs) -> None:
        super().__init__()
        self.config = config
        layers_per_stage = config.n_layers // config.n_stages

        self.tok_embeddings = (
            nn.Embedding(config.vocab_size, config.dim)
            if config.stage_idx == 0
            else None
        )

        # Use ModuleDict so that each layer can be assigned its layer ID in the original model
        self.layers = nn.ModuleDict()

        for layer_id in range(
            layers_per_stage * config.stage_idx,
            layers_per_stage * (config.stage_idx + 1),
        ):
            self.layers[str(layer_id)] = TransformerBlock(config)

        if config.stage_idx == config.n_stages - 1:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        else:
            self.norm = None
            self.output = None

        self.max_batch_size = -1
        self.max_seq_length = -1
        # For supporting sequence parallel (default is off, thus value of 1)
        self.seq_parallel_degree = 1

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
            rope_scaling=self.config.rope_scaling,
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
                RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
            )

        for layer in self.layers.values():
            layer.distribute(device_mesh)

        if self.norm:
            parallelize_module(self.norm, device_mesh, SequenceParallel())

        if self.output:
            parallelize_module(
                self.output,
                device_mesh,
                ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                ),
            )

        self.seq_parallel_degree = device_mesh.size()

    # This is a temporary solution to pass input_pos to non-0 pipeline stages
    # TODO: make `step()` function of dist.pipelining accept args for non-0 stages
    def setup_input_pos(self, input_pos: Tensor) -> None:
        self._input_pos = input_pos

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        # TODO: find a better way to pass input_pos to non-0 pipeline stages
        input_pos = input_pos if input_pos is not None else self._input_pos
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
        # print(f"output shape: {x.shape}")
        return x


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
        parallelize_module(self.ffn_norm, device_mesh, SequenceParallel())
        parallelize_module(self.attention_norm, device_mesh, SequenceParallel())

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

    def distribute(self, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh
        parallelize_module(self.wq, device_mesh, ColwiseParallel())
        parallelize_module(self.wk, device_mesh, ColwiseParallel())
        parallelize_module(self.wv, device_mesh, ColwiseParallel())
        parallelize_module(
            self.wo, device_mesh, RowwiseParallel(output_layouts=Shard(1))
        )
        # TODO: enable kv cache in distributed case
        self.kv_cache = None

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        # Gather sequence back in case of sequence parallelism before attention
        if isinstance(x, DTensor):
            x = x.redistribute(self.device_mesh, [Replicate()])

        bsz, seqlen, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # kv_size = self.n_local_heads * self.head_dim
        # q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        # Giving "-1" to view ops so that they infer the correct number of heads
        # from the input tensor.  This is done to support both TP and non-TP
        # cases where the former would divide n_heads by tp_degree.
        # -1 = self.n_heads
        q = q.view(bsz, seqlen, -1, self.head_dim)
        # -1 = self.n_local_heads
        k = k.view(bsz, seqlen, -1, self.head_dim)
        # -1 = self.n_local_heads
        v = v.view(bsz, seqlen, -1, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        # -1 = self.dim
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: TransformerArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def distribute(self, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh
        parallelize_module(self.w1, device_mesh, ColwiseParallel())
        parallelize_module(
            self.w2, device_mesh, RowwiseParallel(output_layouts=Shard(1))
        )
        parallelize_module(self.w3, device_mesh, ColwiseParallel())

    def forward(self, x: Tensor) -> Tensor:
        # Gather sequence back in case of sequence parallelism
        if isinstance(x, DTensor):
            x = x.redistribute(self.device_mesh, [Replicate()])

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


def apply_scaling(freqs: torch.Tensor, rope_scaling: Dict[str, Any]):
    # Check for the presence of the required keys
    required_keys = {
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    }
    if not required_keys.issubset(rope_scaling.keys()):
        raise ValueError(
            f"Missing required keys in apply_scaling. Expected: {required_keys}"
        )

    scale_factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

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
    n_elem: int,
    seq_len: int,
    base: int = 10000,
    dtype=None,
    rope_scaling: Optional[Dict[str, Any]] = None,
) -> Tensor:
    if not dtype:
        dtype = get_precision()
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    if rope_scaling is not None:
        freqs = apply_scaling(freqs, rope_scaling)
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
    from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache  # no-qa

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

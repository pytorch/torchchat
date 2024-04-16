# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import gguf

import torch
import torch.nn as nn

from gguf import GGUFValueType, ReaderTensor
from quantize import (
    group_dequantize_tensor_from_qparams,
    pack_scales_and_zeros,
    WeightOnlyInt4Linear,
)

from build.gguf_util import F16, F32, Q4_0, Q6_K

wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from typing import Set

from model import ModelArgs, Transformer

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class AttentionArgs:
    head_count: int
    head_count_kv: int
    layer_norm_rms_epsilon: float


@dataclass
class RopeArgs:
    dimension_count: int | None = None
    freq_base: float | None = None


@dataclass
class GGUFModelArgs:
    arch: str
    embedding_length: int
    block_count: int
    feed_forward_length: int
    vocab_size: int
    attention: AttentionArgs
    rope: RopeArgs


@dataclass
class GGUFWeights:
    tensors: list[ReaderTensor]


def _create_pt_model(
    gguf_model_args: GGUFModelArgs,
) -> nn.Module:
    llama_model_args = ModelArgs(
        dim=gguf_model_args.embedding_length,
        n_layers=gguf_model_args.block_count,
        n_heads=gguf_model_args.attention.head_count,
        n_local_heads=gguf_model_args.attention.head_count_kv,
        vocab_size=gguf_model_args.vocab_size,
        norm_eps=gguf_model_args.attention.layer_norm_rms_epsilon,
        hidden_dim=gguf_model_args.feed_forward_length,
    )
    pt_model = Transformer(llama_model_args)
    pt_model.eval()
    return pt_model


_name_replacements = [
    ("blk", "layers"),
    ("token_embd", "tok_embeddings"),
    ("attn_q", "attention.wq"),
    ("attn_k", "attention.wk"),
    ("attn_v", "attention.wv"),
    ("attn_output", "attention.wo"),
    ("attn_norm", "attention_norm"),
    ("output_norm.weight", "norm.weight"),
    ("ffn_down", "feed_forward.w2"),
    ("ffn_gate", "feed_forward.w1"),
    ("ffn_up", "feed_forward.w3"),
]


def _convert_gguf_tensor_name_to_llama_nn(gguf_name: str) -> str:
    result = copy.deepcopy(gguf_name)
    for gguf_string, replacement in _name_replacements:
        result = result.replace(gguf_string, replacement)
    return result


def _build_model_args(metadata: dict[str, Any]) -> GGUFModelArgs:
    arch = metadata["general.architecture"]
    assert (
        arch == "llama"
    ), f"Only general.architecture=llama is supported, but got general.architecture={arch}"
    return GGUFModelArgs(
        arch=arch,
        embedding_length=metadata[f"{arch}.embedding_length"],
        block_count=metadata[f"{arch}.block_count"],
        feed_forward_length=metadata[f"{arch}.feed_forward_length"],
        vocab_size=len(metadata["tokenizer.ggml.tokens"]),
        attention=AttentionArgs(
            head_count=metadata[f"{arch}.attention.head_count"],
            head_count_kv=metadata[f"{arch}.attention.head_count_kv"],
            layer_norm_rms_epsilon=metadata[f"{arch}.attention.layer_norm_rms_epsilon"],
        ),
        rope=RopeArgs(
            freq_base=metadata.get(f"{arch}.rope.freq_base", None),
            dimension_count=metadata.get(f"{arch}.rope.dimension_count", None),
        ),
    )


def _fqn_lookup(fqn: str, module: torch.nn.Module) -> Any:
    if fqn == "":
        return module
    atoms = fqn.split(".")
    curr = module
    for a in atoms:
        curr = getattr(curr, a)
    return curr


def _fqn_down(fqn: str, name: str) -> str:
    if fqn == "":
        return name
    return f"{fqn}.{name}"


def _fqn_up(fqn: str) -> str:
    atoms = fqn.split(".")
    if len(atoms) == 1:
        return ""
    return ".".join(atoms[0:-1])


def _fqn_last(fqn: str) -> str:
    atoms = fqn.split(".")
    return atoms[-1]


def load_weights(
    pt_model: torch.nn.Module, weight_map: Dict[str, ReaderTensor], inner_k_tiles=8
) -> None:
    fqns = []
    for fqn in pt_model.state_dict():
        assert _fqn_last(fqn) == "weight"
        fqns.append(_fqn_up(fqn))

    state_dict = {}
    for fqn in fqns:
        mod = _fqn_lookup(fqn, pt_model)

        t = weight_map[f"{fqn}.weight"]

        if (
            isinstance(mod, torch.nn.Linear)
            and t.tensor_type == gguf.GGMLQuantizationType.Q4_0
        ):
            assert not mod.bias
            out_features = mod.out_features
            in_features = mod.in_features
            assert all(t.shape == (in_features, out_features))

            q, s, z = Q4_0.unpack(t)
            scales_and_zeros = pack_scales_and_zeros(s, z)
            weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                q, inner_k_tiles
            )

            state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
            state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")

            parent = _fqn_lookup(_fqn_up(fqn), pt_model)
            setattr(
                parent,
                _fqn_last(fqn),
                WeightOnlyInt4Linear(
                    "cpu",  # TODO: should --device work for gguf load? (yes?!)
                    in_features,
                    out_features,
                    bias=False,
                    groupsize=Q4_0.groupsize,
                    inner_k_tiles=inner_k_tiles,
                ),
            )
        else:
            # All other weights are dequantized to float
            if t.tensor_type == gguf.GGMLQuantizationType.Q4_0:
                as_float = group_dequantize_tensor_from_qparams(
                    *Q4_0.unpack(t), Q4_0.n_bit, Q4_0.groupsize
                )
            elif t.tensor_type == gguf.GGMLQuantizationType.Q6_K:
                as_float = group_dequantize_tensor_from_qparams(
                    *Q6_K.unpack(t), Q6_K.n_bit, Q6_K.groupsize
                )
            elif t.tensor_type == gguf.GGMLQuantizationType.F16:
                as_float = F16.unpack(t)
            elif t.tensor_type == gguf.GGMLQuantizationType.F32:
                as_float = F32.unpack(t)
            else:
                raise ValueError(f"Unsupported tensor type {t.tensor_type}")

            state_dict[f"{fqn}.weight"] = as_float.to("cpu")

    pt_model.load_state_dict(state_dict)
    return pt_model


def _get_metadata(reader: gguf.GGUFReader) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for idx, field in enumerate(reader.fields.values()):
        val = None
        if field.types[:1] == [GGUFValueType.ARRAY]:
            itype = field.types[-1]
            if itype == GGUFValueType.STRING:
                val = [
                    str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data
                ]
            else:
                val = [pv for idx in field.data for pv in field.parts[idx].tolist()]
        elif field.types[0] == GGUFValueType.STRING:
            val = str(bytes(field.parts[-1]), encoding="utf-8")
        else:
            val = field.parts[-1].tolist()[0]

        metadata[field.name] = val

    return metadata


def load_llama_from_gguf_file(gguf_file: str) -> torch.nn.Module:
    """
    Load a LLaMa model from a GGUF file and return a PT nn.Module.
    """
    if not Path(gguf_file).is_file():
        raise ValueError(f"Could not find file {gguf_file}")

    logger.info("Parsing GGUF metadata.")
    reader = gguf.GGUFReader(gguf_file, "r")
    metadata = _get_metadata(reader)
    model_args = _build_model_args(metadata)
    assert (
        model_args.arch == "llama"
    ), "Only LLaMa models are supported by this converter."

    logger.info("Creating initial PT model.")
    pt_model = _create_pt_model(model_args)

    logger.info("Reading GGUF weights.")
    gguf_weights = GGUFWeights(tensors=reader.tensors)

    logger.info("Building GGUF weight map.")
    # map from fqn in pt_model to gguf tensor
    weight_map = {
        _convert_gguf_tensor_name_to_llama_nn(tensor.name): tensor
        for tensor in gguf_weights.tensors
    }

    logger.info("Loading weights into state_dict")
    pt_model = load_weights(pt_model, weight_map, inner_k_tiles=8)
    return pt_model

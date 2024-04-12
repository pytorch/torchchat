# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Dict
import logging

import gguf

import torch
import torch.nn as nn

from gguf import GGUFValueType, ReaderTensor

wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs, Transformer
from typing import Set
from ggml_quantization_type import Q4_0

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
        n_layer=gguf_model_args.block_count,
        n_head=gguf_model_args.attention.head_count,
        n_local_heads=gguf_model_args.attention.head_count_kv,
        vocab_size=gguf_model_args.vocab_size,
        norm_eps=gguf_model_args.attention.layer_norm_rms_epsilon,
        intermediate_size=gguf_model_args.feed_forward_length,
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
    assert arch == "llama", f"Only general.architecture=llama is supported, but got general.architecture={arch}"
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


# def _load_by_state_dict(pt_model: torch.nn.Module, state_dict: Dict[str, Any], fqn: str, gguf_tensor: ReaderTensor) -> bool:
#     if gguf_tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
#         reversed_shape = gguf_tensor.shape[::-1]
#         new_tensor = gguf_tensor.data.reshape(reversed_shape)
#         state_dict[fqn] = torch.from_numpy(new_tensor)
#         return True
#     elif gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0 and gguf_tensor.name == "token_embd.weight":
#         unpacked = Q4_0.to_float(torch.from_numpy(gguf_tensor.data.reshape(-1, 18)))
#         state_dict[fqn] = unpacked.reshape(
#             pt_model.config.vocab_size, pt_model.config.dim
#         )
#         return True
#     return False


# def _load_by_parameter(pt_model: torch.nn.Module, fqn: str, gguf_tensor: ReaderTensor) -> bool:
#     assert isinstance(_fqn_lookup(fqn, pt_model), torch.nn.Parameter)
#     parent: torch.nn.Module = _fqn_lookup(_fqn_up(fqn), pt_model)

#     if gguf_tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0 and isinstance(parent, torch.nn.Linear) and _fqn_last(fqn) == "weight":
#         packed = torch.from_numpy(gguf_tensor.data).reshape(-1, 18)
#         scale = Q4_0._unpack_two_uint8(packed[:, :2]).to(dtype=torch.float16)
#         parent.weight = torch.nn.Parameter(
#             Q4_0.GGMLInt4LinearWeight(packed, scale, parent.weight.shape)
#         )
#         pt_model = pt_model.to(dtype=torch.float32)
#         return True

#     return False


# def _load_weights(pt_model: torch.nn.Module, weight_map: Dict[str, ReaderTensor]) -> None:
#     loaded_by_state_dict: Set[str] = set()
#     loaded_by_parameter: Set[str] = set()

#     # state_dict pass
#     logger.info("Loading weights by state_dict.")
#     state_dict = {}
#     for fqn in pt_model.state_dict():
#         if fqn not in weight_map:
#             continue
#         tensor = weight_map[fqn]
#         loaded = _load_by_state_dict(pt_model, state_dict, fqn, tensor)
#         if loaded:
#             loaded_by_state_dict.add(fqn)

#     # allow partial loading
#     pt_model.load_state_dict(state_dict, strict=False)

#     # parameter pass
#     logger.info("Loading weights by parameter.")
#     for fqn, param in pt_model.named_parameters():
#         if fqn not in weight_map:
#             continue
#         tensor = weight_map[fqn]
#         loaded = _load_by_parameter(pt_model, fqn, tensor)
#         if loaded:
#             loaded_by_parameter.add(fqn)

#     # Sanity checks
#     for fqn in loaded_by_state_dict:
#         if not(fqn not in loaded_by_parameter):
#             msg = f"{fqn} was loaded by both state_dict and parameter"
#             raise Exception(msg)

#     for fqn in weight_map:
#         if not (fqn in (loaded_by_state_dict | loaded_by_parameter)):
#             msg = f"{fqn} in weight_map was not loaded"
#             raise Exception(msg)

#     for fqn in pt_model.state_dict():
#         if not (fqn in (loaded_by_state_dict | loaded_by_parameter)):
#             msg = f"{fqn} in model.state_dict() was not loaded"
#             raise Exception(msg)


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

# TODO: finish weight loading
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
    # logger.info("Loading GGUF weights into PT model.")
    # _load_weights(pt_model, weight_map)

    return pt_model

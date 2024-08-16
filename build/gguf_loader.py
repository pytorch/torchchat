# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import logging
from typing import Any

import gguf

import torch

from gguf import GGUFValueType
from quantization.qops import LinearInt4 as WeightOnlyInt4Linear
from quantization.quantize import pack_scales_and_zeros
from build.gguf_util import Q4_0, to_float
from build.model import TransformerArgs, Transformer

logger: logging.Logger = logging.getLogger(__name__)


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


def load_model(gguf_file: str) -> torch.nn.Module:
    """
    Parses the GGUF file and returns an nn.Module on meta device.
    """

    logger.info("Parsing GGUF metadata.")
    reader = gguf.GGUFReader(gguf_file, "r")
    metadata = _get_metadata(reader)

    arch = metadata["general.architecture"]
    assert arch == "llama", "Only LLaMa models are supported by this converter."

    model_args = TransformerArgs(
        dim=metadata[f"{arch}.embedding_length"],
        n_layers=metadata[f"{arch}.block_count"],
        n_heads=metadata[f"{arch}.attention.head_count"],
        n_local_heads=metadata[f"{arch}.attention.head_count_kv"],
        vocab_size=len(metadata["tokenizer.ggml.tokens"]),
        norm_eps=metadata[f"{arch}.attention.layer_norm_rms_epsilon"],
        hidden_dim=metadata[f"{arch}.feed_forward_length"],
    )

    # TODO: what to do with rope args like
    # metadata.get(f"{arch}.rope.freq_base", None)
    # metadata.get(f"{arch}.rope.dimension_count", None)

    with torch.device("meta"):
        model = Transformer(model_args)
    return model


def load_model_and_state_dict(
    gguf_file: str,
    *,
    load_state_dict: bool = True,
    load_as_quantized: bool = True,
    inner_k_tiles=8,
) -> torch.nn.Module:
    """
    Parses the GGUF file and returns an nn.Module on meta device along with a state_dict
    that can be loaded into it.

    When load_as_quantized, the method tries to preserve the GGUF quantization when it
    is natively supported by PyTorch, otherwise it converts quantized tensors to FP32.
    """

    model = load_model(gguf_file)

    reader = gguf.GGUFReader(gguf_file, "r")
    weight_map = {
        _convert_gguf_tensor_name_to_llama_nn(tensor.name): tensor
        for tensor in reader.tensors
    }

    state_dict = {}
    for fqn in weight_map:
        assert _fqn_last(fqn) == "weight"
        fqn = _fqn_up(fqn)

        mod = _fqn_lookup(fqn, model)
        t = weight_map[f"{fqn}.weight"]

        if (
            isinstance(mod, torch.nn.Linear)
            and t.tensor_type == gguf.GGMLQuantizationType.Q4_0
            and load_as_quantized
        ):
            assert not mod.bias
            out_features = mod.out_features
            in_features = mod.in_features
            assert all(t.shape == (in_features, out_features))

            if load_state_dict:
                q, s, z = Q4_0.unpack(t)
                scales_and_zeros = pack_scales_and_zeros(s, z)
                q_uint8 = (q[::, ::2] << 4 | q[::, 1::2]).to(torch.uint8)
                weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                    q_uint8, inner_k_tiles
                )
                state_dict[f"{fqn}.weight"] = weight_int4pack
                state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros

            parent = _fqn_lookup(_fqn_up(fqn), model)
            setattr(
                parent,
                _fqn_last(fqn),
                WeightOnlyInt4Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    device="meta",
                    groupsize=Q4_0.groupsize,
                    inner_k_tiles=inner_k_tiles,
                ),
            )
        else:
            if load_state_dict:
                state_dict[f"{fqn}.weight"] = to_float(t)

    assert (state_dict == {}) == (not load_state_dict)
    return model, state_dict


def load_llama_from_gguf_file(gguf_file: str) -> torch.nn.Module:
    model, state_dict = load_model_and_state_dict(gguf_file, load_as_quantized=True)
    model.load_state_dict(state_dict, assign=True)
    return model

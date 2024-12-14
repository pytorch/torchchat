# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import logging
from typing import Any, Optional

import gguf

import torch
import torch.nn.functional as F

from gguf import GGUFValueType

from torchchat.model import Model, ModelArgs, TransformerArgs

from torchchat.utils.build_utils import find_multiple, get_precision
from torchchat.utils.quantize import (
    group_dequantize_tensor_from_qparams,
    pack_scales_and_zeros,
)

from torchao.dtypes.utils import is_device


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
    result = "model." + result
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


#########################################################################
# Note: int4 quantization is migrated to torchao for general quantization.
# TODO: GGUF workflow needs migration to torchao
#########################################################################


def linear_int4(input, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_input_size = input.size()
    input = input.reshape(-1, origin_input_size[-1])

    if "cuda" in str(input.device):
        c = torch.ops.aten._weight_int4pack_mm(
            input.to(torch.bfloat16),
            weight_int4pack,
            groupsize,
            scales_and_zeros.to(torch.bfloat16),
        ).to(
            input.dtype
        )  # cast back to input.dtype
    else:
        c = torch.ops.aten._weight_int4pack_mm_for_cpu(
            input,
            weight_int4pack,
            groupsize,
            scales_and_zeros,
        )

    new_shape = origin_input_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    scales_and_zeros: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        *,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
        weight: Optional[torch.Tensor] = None,
        scales_and_zeros: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.padding = not self._check_k(
            k=in_features,
            groupsize=groupsize,
            inner_k_tiles=inner_k_tiles,
        )
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert (
            in_features % (inner_k_tiles * 16) == 0
        ), "require in_features % (innerKTiles * 16) == 0"
        assert (weight is None) == bool(
            scales_and_zeros is None
        ), "must specify both weights and scales_and_zeros, or neither"

        if weight is None:
            if is_device(device, "cpu"):
                weight = torch.empty(
                    (
                        out_features,
                        in_features // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
            else:
                weight = torch.empty(
                    (
                        out_features // 8,
                        in_features // (inner_k_tiles * 16),
                        32,
                        inner_k_tiles // 2,
                    ),
                    dtype=torch.int32,
                    device=device,
                )

            scales_and_zeros = torch.empty(
                (in_features // groupsize, out_features, 2),
                dtype=get_precision(),
                device=device,
            )

        self.register_buffer(
            "weight",
            weight,
        )
        self.register_buffer(
            "scales_and_zeros",
            scales_and_zeros,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_int4(
            input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )

    @classmethod
    def _check_k(cls, *, k, groupsize=1, inner_k_tiles=1):
        return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0

    @classmethod
    def _prepare_weight_and_scales_and_zeros(
        cls, weight_bf16, groupsize, inner_k_tiles
    ):
        from torchchat.utils.quantize import group_quantize_tensor

        weight_int32, scales_and_zeros = group_quantize_tensor(
            weight_bf16, n_bit=4, groupsize=groupsize
        )
        if is_device(weight_int32.device.type, "cpu"):
            weight_int4pack = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                weight_int32, inner_k_tiles
            )
        else:
            weight_uint8 = (weight_int32[::, ::2] << 4 | weight_int32[::, 1::2]).to(
                torch.uint8
            )
            weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                weight_uint8, inner_k_tiles
            )
        return weight_int4pack, scales_and_zeros

    @classmethod
    def _calc_padded_size(cls, *, k, groupsize=1, innner_k_tiles=1):
        return find_multiple(k, 1024)


#########################################################################
# Quant Loading for GGUF


def to_float(t: gguf.gguf_reader.ReaderTensor):
    """
    Unpack and dequantize GGUF tensor to torch tensor of type torch.float32.
    """

    # All other weights are dequantized to float
    if t.tensor_type == gguf.GGMLQuantizationType.Q4_0:
        return group_dequantize_tensor_from_qparams(
            *Q4_0.unpack(t), Q4_0.n_bit, Q4_0.groupsize
        ).to(torch.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.Q6_K:
        return group_dequantize_tensor_from_qparams(
            *Q6_K.unpack(t), Q6_K.n_bit, Q6_K.groupsize
        ).to(torch.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.F16:
        return F16.unpack(t).to(torch.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.F32:
        return F32.unpack(t).to(torch.float32)
    else:
        raise ValueError(f"Unsupported tensor type {t.tensor_type}")


def test_by_to_float(source_file: str, target_file: str) -> None:
    """
    Tests methods in this file by using the to_float method, and comparing with a correct
    reference.  Raises error if there is a mismatch.

    In more detail, a GGUF source_file with various GGUF tensor types is parsed, and these
    tensors are converted with to_float.  These are then compared against a GGUF target_file.
    The target GGUF file must only contain F32 tensors, and should be generated by a method
    that is known to be correct.
    """

    gguf_sources = {t.name: t for t in gguf.GGUFReader(source_file, "r").tensors}
    gguf_targets = {t.name: t for t in gguf.GGUFReader(target_file, "r").tensors}

    for t in gguf_targets.values():
        assert (
            t.tensor_type == gguf.GGMLQuantizationType.F32
        ), f"target_file must only contain F32 tensors, but found tensor {t.name} with type {repr(t.tensor_type)}."
    assert (
        gguf_sources.keys() == gguf_targets.keys()
    ), "source_file and target_file should have the same tensors (by name)"

    for k in gguf_sources:
        source = to_float(gguf_sources[k])
        target = to_float(gguf_targets[k])

        if not torch.allclose(source, target):
            print(
                f"After calling to_float on source tensor {k} of type {repr(gguf_sources[k].tensor_type)} it does not match its target."
            )
            print("First 5 elements of converted source: ", source.reshape(-1)[0:5])
            print("First 5 elements of target: ", target.reshape(-1)[0:5])
            raise AssertionError("found mismatch")

    print("All tensors match.")


class F16:
    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF F16 tensor.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.F16
        reversed_shape = gguf_tensor.shape[::-1]
        new_tensor = gguf_tensor.data.reshape(reversed_shape)
        return torch.from_numpy(new_tensor).to(torch.float16)


class F32:
    @staticmethod
    def unpack(gguf_tensor: gguf.gguf_reader.ReaderTensor):
        """
        Unpacks GGUF F32 tensor.
        """
        assert gguf_tensor.tensor_type == gguf.GGMLQuantizationType.F32
        reversed_shape = gguf_tensor.shape[::-1]
        new_tensor = gguf_tensor.data.reshape(reversed_shape)
        return torch.from_numpy(new_tensor).to(torch.float32)


class Q4_0:
    groupsize = 32
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
        nc, nr = gguf_tensor.shape  # GGUF tensor has reversed shape

        QK4_0 = 32  # groupsize

        # Parse block_q4_0
        block_q4_0_size = int(2 + QK4_0 / 2)
        packed = torch.from_numpy(gguf_tensor.data.reshape(-1, block_q4_0_size))
        assert packed.dtype == torch.uint8
        ng = packed.shape[0]  # number of groups/blocks

        curr = 0
        size = 2  # half size
        d = packed[:, curr : (curr + size)].contiguous()
        d = torch.tensor(d.untyped_storage(), dtype=torch.float16).reshape(ng, 1)
        curr += size

        size = int(QK4_0 / 2)
        qs = packed[:, curr : (curr + size)].contiguous()
        curr += size

        # Check we finished parsing
        assert curr == block_q4_0_size

        # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract 8
        x0 = qs & 0x0F
        x1 = qs >> 4

        int32_data = torch.cat([x0, x1], dim=1).to(torch.int32).reshape(ng, QK4_0)
        assert int32_data.dtype == torch.int32
        assert int32_data.min().item() >= 0
        assert int32_data.max().item() <= 2**4 - 1
        assert int32_data.shape == (ng, QK4_0)

        # Prepare for return
        q = int32_data.to(torch.int32).reshape(nr, nc)
        s = d.to(torch.float32).reshape(nr, -1)
        z = torch.zeros(s.shape).to(torch.float32)
        return q, s, z


class Q6_K:
    groupsize = 16
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
        nc, nr = gguf_tensor.shape  # GGUF tensor has reversed shape
        QK_K = 256

        # Parse block_q6_K
        block_q6_K_size = int(QK_K / 2 + QK_K / 4 + QK_K / 16 + 2)
        packed = torch.from_numpy(gguf_tensor.data.reshape(-1, block_q6_K_size))
        assert packed.dtype == torch.uint8
        ng = packed.shape[0]  # number of groups/blocks

        curr = 0

        size = int(QK_K / 2)
        ql = packed[:, curr : (curr + size)].contiguous()
        assert ql.shape == (ng, 128)
        curr += size

        size = int(QK_K / 4)
        qh = packed[:, curr : (curr + size)].contiguous()
        assert qh.shape == (ng, 64)
        curr += size

        size = int(QK_K / 16)
        scales = packed[:, curr : (curr + size)].contiguous()
        scales = (
            torch.tensor(scales.untyped_storage(), dtype=torch.int8)
            .reshape(ng, int(QK_K / 16))
            .to(torch.float32)
        )
        curr += size

        size = 2  # half size
        d = packed[:, curr : (curr + size)].contiguous()
        d = (
            torch.tensor(d.untyped_storage(), dtype=torch.float16)
            .reshape(ng, 1)
            .to(torch.float32)
        )
        curr += size

        # Check we finished parsing
        assert curr == block_q6_K_size

        # Unpack quantized values.  Unlike the code in ggml-quants.c, we do not subtract 32
        q1 = (ql[:, 0:32] & 0xF) | (((qh[:, 0:32] >> 0) & 3) << 4)
        q2 = (ql[:, 32:64] & 0xF) | (((qh[:, 0:32] >> 2) & 3) << 4)
        q3 = (ql[:, 0:32] >> 4) | (((qh[:, 0:32] >> 4) & 3) << 4)
        q4 = (ql[:, 32:64] >> 4) | (((qh[:, 0:32] >> 6) & 3) << 4)

        q5 = (ql[:, 64:96] & 0xF) | (((qh[:, 32:64] >> 0) & 3) << 4)
        q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:64] >> 2) & 3) << 4)
        q7 = (ql[:, 64:96] >> 4) | (((qh[:, 32:64] >> 4) & 3) << 4)
        q8 = (ql[:, 96:128] >> 4) | (((qh[:, 32:64] >> 6) & 3) << 4)

        q = torch.cat([q1, q2, q3, q4, q5, q6, q7, q8], dim=1).to(torch.int32)
        assert q.shape == (ng, QK_K)
        assert q.min().item() >= 0
        assert q.max().item() <= 2**6 - 1

        # Unpack scales
        s1 = d * torch.cat(
            [scales[:, 0].reshape(-1, 1), scales[:, 1].reshape(-1, 1)], dim=1
        )
        s2 = d * torch.cat(
            [scales[:, 2].reshape(-1, 1), scales[:, 3].reshape(-1, 1)], dim=1
        )
        s3 = d * torch.cat(
            [scales[:, 4].reshape(-1, 1), scales[:, 5].reshape(-1, 1)], dim=1
        )
        s4 = d * torch.cat(
            [scales[:, 6].reshape(-1, 1), scales[:, 7].reshape(-1, 1)], dim=1
        )

        s5 = d * torch.cat(
            [scales[:, 8].reshape(-1, 1), scales[:, 9].reshape(-1, 1)], dim=1
        )
        s6 = d * torch.cat(
            [scales[:, 10].reshape(-1, 1), scales[:, 11].reshape(-1, 1)], dim=1
        )
        s7 = d * torch.cat(
            [scales[:, 12].reshape(-1, 1), scales[:, 13].reshape(-1, 1)], dim=1
        )
        s8 = d * torch.cat(
            [scales[:, 14].reshape(-1, 1), scales[:, 15].reshape(-1, 1)], dim=1
        )
        s = torch.cat([s1, s2, s3, s4, s5, s6, s7, s8], dim=1)
        assert s.shape == (ng, 16)

        # Prepare for return
        q = q.to(torch.int32).reshape(nr, nc)
        s = s.reshape(nr, -1)
        z = torch.zeros(s.shape).to(torch.float32)

        return q, s, z


def load_model(gguf_file: str) -> torch.nn.Module:
    """
    Parses the GGUF file and returns an nn.Module on meta device.
    """

    logger.info("Parsing GGUF metadata.")
    reader = gguf.GGUFReader(gguf_file, "r")
    metadata = _get_metadata(reader)

    arch = metadata["general.architecture"]
    assert arch == "llama", "Only LLaMa models are supported by this converter."

    model_args = ModelArgs(
        {
            "text": {
                "dim": metadata[f"{arch}.embedding_length"],
                "n_layers": metadata[f"{arch}.block_count"],
                "n_heads": metadata[f"{arch}.attention.head_count"],
                "n_local_heads": metadata[f"{arch}.attention.head_count_kv"],
                "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
                "norm_eps": metadata[f"{arch}.attention.layer_norm_rms_epsilon"],
                "hidden_dim": metadata[f"{arch}.feed_forward_length"],
            }
        }
    )

    # TODO: what to do with rope args like
    # metadata.get(f"{arch}.rope.freq_base", None)
    # metadata.get(f"{arch}.rope.dimension_count", None)

    with torch.device("meta"):
        model = Model.from_model_args(model_args)
    return model


def load_model_and_state_dict(
    gguf_file: str,
    *,
    load_state_dict: bool = True,
    load_as_quantized: bool = True,
    inner_k_tiles=8,
    device="cpu",
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
                if is_device(q.device.type, "cpu"):
                    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                        q, inner_k_tiles
                    )
                else:
                    q_tmp = (q[::, ::2] << 4 | q[::, 1::2]).to(torch.uint8)
                    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                        q_tmp, inner_k_tiles
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
                    device="cpu",
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

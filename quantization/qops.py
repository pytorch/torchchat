# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from build.utils import find_multiple, get_precision, use_et_backend


def linear_int8_aoti(input, weight, scales):
    n_groups = scales.numel() // scales.shape[0]

    # we special-case channel-wise, because we know how to make that fast
    if n_groups == 1:
        scales = scales.view(-1)
        if (
            torch.compiler.is_compiling()
            or input.device.type not in ["cpu", "mps"]
            or not hasattr(torch.ops.aten, "_weight_int8pack_mm")
        ):
            lin = F.linear(input, weight.to(dtype=input.dtype))
            # print(f"linear shape {lin.shape}, scales shape {scales.shape}")
            return lin * scales
        # Use int8pack_mm for CPU eager
        return torch.ops.aten._weight_int8pack_mm(
            input.reshape(-1, input.shape[-1]),
            weight,
            scales,
        ).reshape(input.shape[:-1] + (weight.shape[0],))

    return F.linear(
        input,
        (
            weight.to(dtype=input.dtype).view(weight.shape[0], n_groups, -1)
            * scales.view(weight.shape[0], n_groups, -1)
        ).view(weight.shape[0], -1),
    )


def _qdq_dynamic_quantized_linear(
    x_fp32,
    x_quant_min,
    x_quant_max,
    x_eps,
    weight_i8,
    weight_scale,
    weight_zero_point,
    weight_quant_min,
    weight_quant_max,
    bias_fp32,
):
    x_scale, x_zero_point = torch.ops.quantized_decomposed.choose_qparams(
        x_fp32, x_quant_min, x_quant_max, x_eps, torch.int8
    )
    x_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        x_fp32, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8
    )
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8
    )
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8,
        weight_scale,
        weight_zero_point,
        weight_quant_min,
        weight_quant_max,
        torch.int8,
    )
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    return out_fp32


def linear_int8_et(input, weight, scales):
    n_groups = scales.numel() // scales.shape[0]

    # we special-case channel-wise, because we know how to make that fast
    if n_groups == 1:
        scales = scales.view(-1)

        if True:
            lin = F.linear(input, weight.to(dtype=input.dtype))
            # print(f"linear shape {lin.shape}, scales shape {scales.shape}")
            return lin * scales

        return _qdq_dynamic_quantized_linear(
            x_fp32=input.float(),
            x_quant_min=-128,
            x_quant_max=127,
            x_eps=torch.finfo(input.dtype).eps,
            weight_i8=weight,
            weight_scale=scales.float(),
            weight_zero_point=0,
            weight_quant_min=-128,
            weight_quant_max=127,
            bias_fp32=None,
        ).to(dtype=input.dtype)

    return F.linear(
        input,
        (
            weight.to(dtype=input.dtype).view(weight.shape[0], n_groups, -1)
            * scales.view(weight.shape[0], n_groups, -1)
        ).view(weight.shape[0], -1),
    )


class LinearInt8(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    scales: torch.Tensor

    def __init__(
        self,
        in_features,
        out_features,
        bias=None,
        device=None,
        dtype=None,
        *,
        weight: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        groupsize: Optional[int] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()

        if device is None:
            device = "cpu"

        assert not bias, "Bias is not supported by LinearInt8"
        self.in_features = in_features
        self.out_features = out_features

        assert (weight is None) == bool(
            scales is None
        ), "must specify both weights and scales, or neither"
        if weight is None:
            weight = torch.empty(
                (out_features, in_features),
                dtype=torch.int8,
                device=device,
            )
            if groupsize is None or (groupsize == 0):
                scales = torch.empty(out_features, dtype=dtype, device=device)
            else:
                n_groups = (in_features + groupsize - 1) // groupsize
                scales = torch.empty(out_features, n_groups, dtype=dtype, device=device)

        self.register_buffer("weight", weight.to(device))
        self.register_buffer("scales", scales.to(device))

        if use_et_backend():
            self.forward = self.et_forward
        else:
            self.forward = self.aoti_forward

    def aoti_forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear_int8_aoti(input, self.weight, self.scales)

    def et_forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear_int8_et(input, self.weight, self.scales)


class QuantizedEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # vocab_size: int,
        embedding_dim: int,
        device=None,
        dtype=None,
        *,
        bitwidth: int,
        groupsize: Optional[int] = None,
        weight: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = get_precision()
        if groupsize is None or groupsize == 0:
            groupsize = embedding_dim
        self.groupsize = groupsize
        self.dtype = dtype
        self.bitwidth = bitwidth

        assert (weight is None) == bool(
            scales is None
        ), "must specify both weights and scales, or neither"

        if bitwidth not in [4, 8]:
            raise RuntimeError(
                f"QUantized embedding does not support bitwidth={bitwidth}"
            )

        if weight is None:
            groups_per_row = (embedding_dim + groupsize - 1) // groupsize
            weight = torch.empty(
                (
                    num_embeddings,
                    (embedding_dim * bitwidth) // 8,
                ),
                dtype=torch.int8,
                device=device,
            )
            scales = torch.empty(
                (num_embeddings, groups_per_row),
                dtype=dtype,
                device=device,
            ).squeeze(dim=-1)

        self.register_buffer(
            "weight",
            weight,
        )
        self.register_buffer(
            "scales",
            scales,
        )

        if use_et_backend():
            self.forward = self.et_forward
        else:
            self.forward = self.aoti_forward

    @torch.no_grad()
    def et_forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.bitwidth == 8:
            return torch.ops.quantized_decomposed.embedding_byte.dtype(
                self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype
            )
        else:
            return torch.ops.quantized_decomposed.embedding_4bit.dtype(
                self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype
            )

    @torch.no_grad()
    def aoti_forward(self, indices: torch.Tensor) -> torch.Tensor:
        # result_weights = self.weight.index_select(0, indices.view(-1))
        # result_scales = self.scales.index_select(0, indices.view(-1))

        if self.bitwidth == 4:
            weight_even = self.weight.div(16, rounding_mode="trunc")
            weight_odd = self.weight.remainder(16)
            weight_unpacked = torch.stack((weight_even, weight_odd), dim=-1)
            weight = weight_unpacked.view(self.weight.shape[0], -1)
            weight = weight.to(torch.int8).add(-8)
        else:
            weight = self.weight

        scales = self.scales.view(weight.shape[0], -1)

        result_weights = F.embedding(indices, weight)
        result_scales = F.embedding(indices, scales)

        rw_view = result_weights.to(dtype=result_scales.dtype).view(
            tuple(
                result_weights.shape[:-1]
                + (
                    scales.shape[1],
                    -1,
                )
            )
        )
        rs_view = result_scales.view(
            tuple(result_scales.shape[:-1])
            + (
                scales.shape[1],
                1,
            )
        )
        # print(f"rw_view {rw_view.shape}")
        # print(f"rs_view {rs_view.shape}")

        r = rw_view * rs_view
        return r.view(indices.size() + (-1,))

        # r = result_weights.to(dtype=result_scales.dtype).view(list(result_weights.shape[:-1] + (scales.shape[1], -1, )) * result_scales.view(scales.shape[-1] + (scales.shape[1], 1, ))


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
        c = torch.ops.aten._weight_int4pack_mm(
            input,
            weight_int4pack,
            groupsize,
            scales_and_zeros,
        )
    new_shape = origin_input_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


class LinearInt4(torch.nn.Module):
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
        from quantization.quantize import group_quantize_tensor

        weight_int32, scales_and_zeros = group_quantize_tensor(
            weight_bf16, n_bit=4, groupsize=groupsize
        )
        weight_uint8 = (weight_int32[::, ::2] << 4 | weight_int32[::, 1::2]).to(torch.uint8)
        weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
            weight_uint8, inner_k_tiles
        )
        return weight_int4pack, scales_and_zeros

    @classmethod
    def _calc_padded_size(cls, *, k, groupsize=1, innner_k_tiles=1):
        return find_multiple(k, 1024)


def linear_8da4w(
    input,
    weight_int8,
    scales,
    zeros,
    out_features,
    groupsize,
    precision,
):
    from torchao.quantization.quant_primitives import per_token_dynamic_quant

    input = per_token_dynamic_quant(input)
    # TODO: verify and remove following reshape code
    # origin_input_size = input.size()
    # input = input.reshape(-1, origin_input_size[-1])

    # TODO: better API
    # weight_int8 = torch.ops.quantized_decomposed.unpack_int4_to_int8(weight_int4packed)
    n_bit = 4
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    w_dq = torch.ops.quantized_decomposed.dequantize_per_channel_group(
        weight_int8,
        scales,
        zeros,
        quant_min,
        quant_max,
        torch.int8,
        groupsize,
        precision,
    )

    # input = input.to(torch.float16)
    # w_dq = w_dq.to(torch.float16)
    c = torch.nn.functional.linear(input, w_dq)

    # new_shape = origin_input_size[:-1] + (out_features,)
    # c = c.reshape(new_shape)

    return c


class LinearAct8Int4DQ(torch.nn.Module):
    __constants__ = ["in_features", "origin_in_feature", "out_features"]
    in_features: int
    origin_in_features: int
    out_features: int
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with
    int4 weight.  Weights are per channel groupwise
    quantized. Parameters of importance groupsize: the number of
    elements in each quantized group precision: precision of input and
    output. e.g. torch.float32 means input activation is float32 and
    output is float32.  scales_precision: precision of per group
    scale.  """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        *,
        groupsize: int = 256,
        weight: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        self.origin_in_features = in_features
        in_features = find_multiple(in_features, groupsize)
        self.in_features = in_features
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"

        self.groupsize = groupsize
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        assert (weight is None) == bool(
            scales is None
        ), "must specify both weights and scales_and_zeros, or neither"

        if weight is None:
            weight = torch.empty((out_features, in_features), dtype=torch.int8)
            scales = torch.empty(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            )

        # we received an unpadded weight, so pad it
        if weight.shape[1] != in_features:
            weight = F.pad(weight, pad=(0, self.in_features - self.origin_in_features))

        # currently storing unpacked int8 weights
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer(
            "zeros",
            torch.empty(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        # This operator does not support anything but FP32, so we do the deed
        # Eventually push that into linear_8da4w
        return linear_8da4w(
            input.float(),
            self.weight,
            self.scales,
            self.zeros,
            self.out_features,
            self.groupsize,
            self.precision,
        ).to(dtype=input.dtype)

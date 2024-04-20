# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import json
from functools import reduce
from math import gcd
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



##########################################################################
###               dtype name to torch.dtype mapping                    ###

precision = torch.float


def set_precision(dtype):
    global precision
    precision = dtype


def get_precision():
    global precision
    return precision


def name_to_dtype(name):
    if name in name_to_dtype_dict:
        return name_to_dtype_dict[name]
    else:
        raise RuntimeError(f"unsupported dtype name {name} specified")


name_to_dtype_dict = {
    "fp32": torch.float,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float": torch.float,
    "half": torch.float16,
    "float32": torch.float,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}



def quantize_model(model: nn.Module, device, quantize_options, tokenizer = None):
    """
    Quantize the specified model using the quantizers described by
    a quantization dict of the form:
    {
        'embedding':   {'bitwidth': 8, 'groupsize': 8 },
        'linear:int8': {'bitwidth': 8, 'groupsize': 8},
        'precision':   {'dtype': torch.float16},
    }
    """

    linears_quantized = False
    if isinstance(quantize_options, str):
        quantize_options = json.loads(quantize_options)

    for quantizer, q_kwargs in quantize_options.items():
        if not quantizer in quantizer_class_dict:
            raise RuntimeError(f"unknown quantizer {quantizer} specified")
        
        model = quantizer_class_dict[quantizer](
                model, device, tokenizer, **q_kwargs
            ).quantized_model()

            
#########################################################################
###                QuantHandler API definition                        ###


class QuantHandler:
    def __init__(self, mod, device = "cpu", tokenizer = None):
        self.mod = mod
        self.device = device
        self.tokenizer = tokenizer

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod
    

#########################################################################
###          QuantHandler wrapper for a8w4dq from torchao             ###

class Int8DynActInt4WeightQuantizer(QuantHandler):
    from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer as aoInt8DynActInt4WeightQuantizer
    
    def __init__(self, mod, device = "cpu", tokenizer = None, **kwargs):
        self.mod = mod
        self.device = device
        self.tokenizer = tokenizer
        self.quantizer = aoInt8DynActInt4WeightQuantizer(**kwargs)
            
    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        return self.quantizer.quantize(self.model)

#########################################################################
###          QuantHandler wrapper for a8w4dq from torchao             ###

class PrecisionHandler(QuantHandler):
    def __init__(self, mod, device = "cpu", tokenizer = None, **kwargs):
        self.mod = mod
        self.device = device
        self.tokenizer = tokenizer
            
    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        return self.model.to(device=device, **kwargs)


#########################################################################
#####                     Quantization Primitives                  ######

def dynamically_quantize_per_channel(
    x,
    quant_min,
    quant_max,
    target_dtype,
    groupsize: Optional[int] = None,
    *,
    scales_dtype=torch.float16,
    enable_non_multiple_groups=True,
):
    """
    Dynamically quantize per channel.  This function is used for quantizing weights,
    for linear and embedding layers.

    Arguments:
        x: input tensor,
        quant_min: minimum value after quantization,
        quant_max: maximum value after quantization,
        target_dtype: target data type for weights after quantization,
        groupsize: number of elements of the channel to quantize together

    Keyword arguments:
        scales_dtype: data type of scale,
        enable_non_multiple_groups: if True, allow the rowsize to not be a multiple of group size,
                        with a final group of a size less than group size.

    Assumptions:
        This function assumes symmetric quantization, axis ==0 and a dense memory format.
    """

    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    x_shape_1 = x.shape[1]

    if groupsize is None or groupsize == 0:
        items = x_shape_1
    elif ((x_shape_1 % groupsize) == 0) or not enable_non_multiple_groups:
        assert groupsize > 0, "group size must be positive"
        assert (
            x_shape_1 % groupsize
        ) == 0, f"weights dimension 1 = {x_shape_1} must be a multiple of group size {groupsize}"
        items = groupsize
    else:
        assert groupsize > 0, "group size must be positive"
        print(
            f"row-size of weight matrix {x_shape_1} is not divisible by group size {groupsize}, using nearest neighbor rounding"
        )
        assert (
            x_shape_1 % groupsize != 0
        ), f"expected x.shape[1] to not be a multiple of group size {groupsize}, but got {x_shape_1}"
        padding = groupsize - (x_shape_1 % groupsize)
        x = F.pad(x, (0, padding))
        items = groupsize

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    x = x.view(x.shape[0], x.shape[1] // items, items)
    # get min and max
    min_val, max_val = torch.aminmax(x, dim=2)
    # print(f"min_val {min_val}")
    # print(f"max_val {max_val}")

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = (
        torch.clamp(x_zp, quant_min, quant_max).to(target_dtype).view(x.shape[0], -1)
    )

    scales = scales.to(dtype=scales_dtype)
    quant = quant[:, :x_shape_1]

    return quant, scales, zero_points


def get_group_qparams(w, n_bit=4, groupsize=128, *, scales_dtype=torch.float):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(scales_dtype).reshape(w.shape[0], -1), zeros.to(
        scales_dtype
    ).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros, *, scales_dtype=torch.float):
    assert scales.shape == zeros.shape
    assert scales.dtype == scales_dtype
    assert zeros.dtype == scales_dtype
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int32


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def group_dequantize_tensor_from_qparams(
    w_int32, scales, zeros, n_bit=4, groupsize=128
):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int32.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int32.shape[-1]
    assert w_int32.shape[-1] % groupsize == 0
    assert w_int32.dim() == 2

    w_int32_grouped = w_int32.reshape(-1, groupsize)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    w_dq = (
        w_int32_grouped.sub(2 ** (n_bit - 1)).mul(scales).add(zeros).reshape_as(w_int32)
    )
    return w_dq


def group_dequantize_tensor(w_int32, scales_and_zeros, n_bit=4, groupsize=128):
    scales, zeros = unpack_scales_and_zeros(scales_and_zeros)
    return group_dequantize_tensor_from_qparams(
        w_int32, scales, zeros, n_bit, groupsize
    )




#########################################################################
#####          Weight-only int8 per-channel quantized code         ######


def replace_linear_weight_only_int8_per_channel(
    module, device, node_type, groupsize=None
):
    if groupsize is not None and groupsize != 0:
        pass  # groupsize = 2 ** groupsize

    for name, child in module.named_children():
        # print(f"name: {name}")
        if isinstance(child, nn.Linear):
            if (
                (node_type == "*")
                or (node_type == "output" and name == "output")
                or (node_type == "!output" and name != "output")
            ):
                # print(f"{name, child}")
                # print(f"in_features: {child.in_features}")
                # print(f"out_features: {child.out_features}")
                setattr(
                    module,
                    name,
                    WeightOnlyInt8Linear(
                        device, child.in_features, child.out_features, groupsize
                    ),
                )
        else:
            replace_linear_weight_only_int8_per_channel(
                child, device, node_type, groupsize
            )


class WeightOnlyInt8QuantHandler(QuantHandler):
    def __init__(
        self,
        mod,
        device,
        *,
        node_type: str = "*",
        bitwidth: Optional[int] = None,
        groupsize: Optional[int] = None,
    ):
        self.mod = mod
        self.device = device
        self.groupsize = groupsize
        self.node_type = node_type
        if bitwidth is None:
            self.bitwidth = 8
        else:
            self.bitwidth = bitwidth

    @torch.no_grad()
    def create_quantized_state_dict(self) -> Dict:
        cur_state_dict = self.mod.state_dict()

        if self.bitwidth == 4:
            range_min = -8
            range_max = 7
        elif self.bitwidth == 8:
            range_min = -128
            range_max = 127
        else:
            raise ValueError(f"Unsupported bitwidth {self.bitwidth}")

        for fqn, mod in self.mod.named_modules():
            # print(f"maybe? quantize {fqn}...{type(mod)}")
            if isinstance(mod, torch.nn.Linear):
                # print(f"candidate {fqn}, nodetype {self.node_type}")
                if (
                    (self.node_type == "*")
                    or (self.node_type == "output" and fqn in ["output", "final_proj"])
                    or (
                        self.node_type == "!output"
                        and fqn not in ["output", "final_proj"]
                    )
                ):
                    # print(
                    #     f"quantize {self.node_type} {fqn, mod} with groupsize {self.groupsize}, bitwidth {self.bitwidth}"
                    # )

                    # print(f"initial weight shape {mod.weight.shape}")
                    input_weight = mod.weight.float()

                    # print(f"expanded weight shape {input_weight.shape}")
                    weight, scales, _ = dynamically_quantize_per_channel(
                        input_weight,
                        range_min,
                        range_max,
                        torch.int8,
                        self.groupsize,
                        scales_dtype=mod.weight.dtype,
                    )

                    weight = weight.to(device=self.device)
                    scales = scales.to(device=self.device)
                    cur_state_dict[f"{fqn}.weight"] = weight
                    # squeeze makes groupsize=rowsize unidimensional
                    cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_linear_weight_only_int8_per_channel(
            self.mod, self.device, self.node_type, self.groupsize
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        device,
        in_features: int,
        out_features: int,
        groupsize: Optional[int] = None,
        bias: bool = True,
        dtype=None,
    ) -> None:
        super().__init__()
        # print(f"group size: {groupsize}")

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8, device=device),
        )
        dtype = get_precision()
        if groupsize is None or (groupsize == 0):
            self.register_buffer(
                "scales", torch.ones(out_features, dtype=dtype, device=device)
            )
        else:
            groups = (in_features + groupsize - 1) // groupsize
            self.register_buffer(
                "scales", torch.ones(out_features, groups, dtype=dtype, device=device)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scales = self.scales
        weight = self.weight
        scales = scales.view(scales.shape[0], -1)
        no_groups = scales.shape[1]

        # need a formulation / custom op for good performance on both eager, CUDA compiled, CPU compiled and ET exported
        # maybe use IR-based rewriting?

        # for now, we special-case channel-wise, because we know how to make that fast (but does not work for groupwise)
        if scales.shape[1] == 1:
            return F.linear(input, weight.to(dtype=input.dtype)) * self.scales
        else:
            return F.linear(
                input,
                (
                    weight.to(dtype=input.dtype).view(weight.shape[0], no_groups, -1)
                    * scales.view(weight.shape[0], no_groups, -1)
                ).view(weight.shape[0], -1),
            )


#########################################################################
#####                   embedding table quantization               ######


def replace_embedding_weight_only_grouped_int8_per_channel(
    module, device, bitwidth: int = 8, groupsize: Optional[int] = None, packed=False
):
    for name, child in module.named_children():
        # print(f"name: {name}")
        if isinstance(child, nn.Embedding):
            # print(f"{name, child}")
            # print(f"weights size: {child.weight.size()}")
            setattr(
                module,
                name,
                QuantizedGroupEmbedding(
                    device=device,
                    vocab_size=child.weight.shape[0],
                    embedding_dim=child.weight.shape[1],
                    groupsize=groupsize,
                    packed=packed,
                ),
            )
        else:
            replace_embedding_weight_only_grouped_int8_per_channel(
                child, device, bitwidth, groupsize, packed
            )


class EmbeddingOnlyInt8QuantHandler(QuantHandler):
    def __init__(
        self,
        mod,
        device,
        *,
        bitwidth: int = 8,
        groupsize: Optional[int] = None,
        packed=False,
    ):
        if isinstance(packed, str):
            packed = packed == "True"
        self.mod = mod
        self.device = device
        self.groupsize = groupsize
        self.bitwidth = bitwidth
        self.packed = packed
        if (bitwidth != 4) and packed:
            raise RuntimeError("pack only works with bitsize 4")

    @torch.no_grad()
    def create_quantized_state_dict(self, packed=False) -> Dict:
        cur_state_dict = self.mod.state_dict()

        if self.bitwidth == 4:
            range_min = -8
            range_max = 7
        elif self.bitwidth == 8:
            range_min = -128
            range_max = 127
        else:
            raise ValueError(f"Unsupported bitwidth {self.bitwidth}")

        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, nn.Embedding):
                # print("****")
                # print(f"Embedding identified: {fqn, mod}")
                # print(f"weights size: {mod.weight.size()}")
                # print(f"quantize {fqn}...")

                # print(
                #     f"quantize {fqn, mod} with groupsize {self.groupsize}, bitwidth {self.bitwidth}"
                # )
                weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(),
                    range_min,
                    range_max,
                    torch.int8,
                    self.groupsize,
                    scales_dtype=mod.weight.dtype,
                )

                if packed:
                    if weight.shape[-1] % 2 != 0:
                        raise RuntimeError("automatic padding not implemented yet")

                    weight_range_shifted = weight.add(8).view(torch.uint8)
                    weight_view = weight_range_shifted.view(
                        weight.shape[0], weight.shape[1] // 2, 2
                    )
                    weight_even = weight_view[:, :, 0] * 16  # left shift 4
                    weight_odd = weight_view[:, :, 1]
                    weight_packed = weight_even + weight_odd
                    weight = weight_packed

                weight = weight.to(device=self.device)
                scales = scales.to(device=self.device)
                # Update state dict
                cur_state_dict[f"{fqn}.weight"] = weight
                # squeeze makes groupsize=rowsize unidimensional
                cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_embedding_weight_only_grouped_int8_per_channel(
            self.mod, self.device, self.bitwidth, self.groupsize, self.packed
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict(self.packed)
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class QuantizedGroupEmbedding(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size: int,
        embedding_dim: int,
        groupsize: Optional[int] = None,
        dtype=torch.half,
        packed=False,
    ) -> None:
        super().__init__()
        if groupsize is None or groupsize == 0:
            groupsize = embedding_dim
        self.groupsize = groupsize
        self.dtype = dtype
        self.packed = packed
        if not packed:
            self.register_buffer(
                "weight",
                torch.empty(
                    (vocab_size, embedding_dim), dtype=torch.int8, device=device
                ),
            )
        else:  # packed
            self.register_buffer(
                "weight",
                torch.empty(
                    (vocab_size, embedding_dim // 2), dtype=torch.uint8, device=device
                ),
            )
        groups_per_row = (embedding_dim + groupsize - 1) // groupsize
        if groups_per_row > 1:
            self.register_buffer(
                "scales",
                torch.ones(
                    (vocab_size, groups_per_row), dtype=torch.float16, device=device
                ),
            )
        else:
            self.register_buffer(
                "scales", torch.ones((vocab_size,), dtype=torch.float16, device=device)
            )

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if False:  # Used for Executorch
            return torch.ops.llama_quantized.embedding_byte.dtype(
                self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype
            )

        # result_weights = self.weight.index_select(0, indices.view(-1))
        # result_scales = self.scales.index_select(0, indices.view(-1))

        if self.packed:
            weight_even = self.weight.div(16, rounding_mode="trunc")
            weight_odd = self.weight.remainder(16)
            weight_unpacked = torch.stack((weight_even, weight_odd), dim=-1)
            weight = weight_unpacked.view(self.weight.shape[0], -1)
            weight = weight.view(torch.int8).add(-8)
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


#########################################################################
#####     weight only int4 per channel groupwise quantized code    ######


def _int4_prepare_int4_weight_and_scales_and_zeros(
    weight_bf16, groupsize, inner_k_tiles
):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
        weight_int32, inner_k_tiles
    )
    return weight_int4pack, scales_and_zeros


def _int4_calc_padded_size(k, groupsize=1, innner_k_tiles=1):
    from build.model import find_multiple

    return find_multiple(k, 1024)


def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])

    if "cuda" in str(x.device):
        c = torch.ops.aten._weight_int4pack_mm(
            x.to(torch.bfloat16),
            weight_int4pack,
            groupsize,
            scales_and_zeros.to(torch.bfloat16),
        ).to(
            x.dtype
        )  # cast back to x.dtype
    else:
        c = torch.ops.aten._weight_int4pack_mm(
            x,
            weight_int4pack,
            groupsize,
            scales_and_zeros,
        )
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


def replace_linear_int4(
    module,
    device,
    groupsize,
    inner_k_tiles,
    padding_allowed,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if (
                _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles)
                or padding_allowed
            ):
                setattr(
                    module,
                    name,
                    WeightOnlyInt4Linear(
                        device,
                        child.in_features,
                        child.out_features,
                        bias=False,
                        groupsize=groupsize,
                        inner_k_tiles=inner_k_tiles,
                    ),
                )
        else:
            replace_linear_int4(
                child, device, groupsize, inner_k_tiles, padding_allowed
            )


class WeightOnlyInt4QuantHandler(QuantHandler):
    def __init__(
        self, mod, device, *, groupsize=128, inner_k_tiles=8, padding_allowed=True
    ):
        self.mod = mod
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                # print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = mod.weight.data
                if not _check_linear_int4_k(
                    in_features, self.groupsize, self.inner_k_tiles
                ):
                    if self.padding_allowed:
                        import torch.nn.functional as F
                        from build.model import find_multiple

                        print(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = (
                    _int4_prepare_int4_weight_and_scales_and_zeros(
                        weight.to(torch.float), self.groupsize, self.inner_k_tiles
                    )
                )
                weight_int4pack = weight_int4pack.to(device=self.device)
                scales_and_zeros = scales_and_zeros.to(device=self.device)
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_int4(
            self.mod,
            self.device,
            self.groupsize,
            self.inner_k_tiles,
            self.padding_allowed,
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    scales_and_zeros: torch.Tensor

    def __init__(
        self,
        device: str,
        in_features: int,
        out_features: int,
        bias=True,
        dtype=None,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
    ) -> None:
        super().__init__()
        self.padding = not _check_linear_int4_k(
            in_features, groupsize, inner_k_tiles
        )
        if self.padding:
            from build.model import find_multiple

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
        self.register_buffer(
            "weight",
            torch.empty(
                (
                    out_features // 8,
                    in_features // (inner_k_tiles * 16),
                    32,
                    inner_k_tiles // 2,
                ),
                dtype=torch.int32,
                device=device,
            ),
        )
        # MKG: torch.float
        self.register_buffer(
            "scales_and_zeros",
            torch.empty(
                (in_features // groupsize, out_features, 2),
                dtype=get_precision(),
                device=device,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # MKG torch.float
        # input = input.to(torch.float)
        if self.padding:
            import torch.nn.functional as F

            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )


#########################################################################
#####                           GPTQ                                #####

def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = 1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0


class GPTQQuantHandler(QuantHandler):
    """
    This class implements a GPTQ QuantHandler that can be used to apply GPTQ to a model in concert with the GenericGPTQRunner class.
    Unlike the base QuantHandler class, the user does not need to implement the create_quantized_state_dict, instead they have to reimplement
    __init__ such that it defines the functions for the quantization mode. User is expected to reimplement convert_for_runtime.

    The following functions (which must be defined in __init__) are used to define the quantization mode for both GPTQ and
    create_quantized_state_dict. Here is a description of each function.

    get_qparams_func:
        A function that calculates the quantization qparams for an input tensor.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            qparams: it can have any format but will need to be handled by the other defined functions below.

    quantize_func:
        A function that applies quantization to an input tensor. It should be noted
        that this function needs to be able to handle quantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
            qparams: the output from get_qparams_func
        Returns:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)


    dequantize_func:
        A function that dequantizes an input quantized weight tensor. It should be noted
        that this function needs to be able to handle dequantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            weight: A 2d weight tensor with non-integer dtype.

    combine_qparams_list_func:
        A function that combines several qparams into one qparam.
        Args:
            qparams_list: a list of qparams objects, each obtained by calling get_qparams_func
            on a single group from a weight tensor
        Returns:
            qparams: an object of the same format as the qparams above.

    skip_layer_func:
        A function that determines which linear layers should be skipped during GPTQ
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            skip: boolean indicating whether layer should be skipped

    make_names_and_values_dict_func:
        A function that prepares the qparams and quantized_weight and creates a dictionary indicating how they
        should be inserted into the state_dict. Generally any packing of the weight and qparams should be done here.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            names_and_values_dict: a dictionary mapping the name of the parameters of the quantized module to the
            corresponding quantized weights and qparams.
    """

    def __init__(self):
        assert self.mod is not None
        assert self.get_qparams_func is not None
        assert self.quantize_func is not None
        assert self.dequantize_func is not None
        assert self.combine_qparams_list_func is not None
        assert self.make_names_and_values_dict_func is not None

    @staticmethod
    def get_inputs(
        model,
        tokenizer,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
        device,
    ) -> "MultiInput":
        from GPTQ import InputRecorder
        input_recorder = InputRecorder(
            model,
            tokenizer,
            calibration_seq_length,
            pad_calibration_inputs,
            device,
        )

        try:
            import lm_eval
            lm_eval.tasks.initialize_tasks()
        except:
            pass
        from eval import get_task_dict
        task_dict = get_task_dict(calibration_tasks)
        print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

        from eval import evaluate
        evaluate(
            input_recorder,
            task_dict,
            limit=calibration_limit,
        )
        inputs = input_recorder.get_recorded_inputs()
        assert inputs is not None, (
            f"No inputs were collected, use a task other than {calibration_tasks}, "
            + "use option pad_calibration_inputs, or decrease calibration_sequence_length (currently "
            + f"{calibration_seq_length})"
        )
        print(f"Obtained {len(inputs[0].values)} calibration samples")
        return inputs

    @torch.no_grad()
    def create_quantized_state_dict(
        self,
        tokenizer,
        groupsize,
        blocksize,
        percdamp,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
        device,
    ) -> Dict:  # "StateDict":
        inputs = GPTQQuantHandler.get_inputs(
            self.mod,
            tokenizer,
            calibration_tasks,
            calibration_limit,
            calibration_seq_length,
            pad_calibration_inputs,
            device=device,
        )
        print("Tracing model for GPTQ")
        from GPTQ import GenericGPTQRunner
        GPTQ_runner = GenericGPTQRunner(
            self.mod,
            inputs,
            blocksize,
            percdamp,
            groupsize,
        ).configure_quantization_mode(
            self.get_qparams_func,
            self.quantize_func,
            self.dequantize_func,
            self.combine_qparams_list_func,
            self.make_names_and_values_dict_func,
            self.skip_layer_func,
        )

        print("Applying GPTQ to weights")
        GPTQ_runner.run()
        return GPTQ_runner.get_quantized_state_dict()

    def convert_for_runtime(self) -> "nn.Module":
        pass


class WeightOnlyInt4GPTQQuantHandler(GPTQQuantHandler):
    def __init__(self,
        mod,
        tokenizer,
        device,
        *,
        groupsize=128,
        inner_k_tiles=8,
        padding_allowed=True,
        blocksize=128,
        percdamp=0.01,
        calibration_tasks=["wikitext"],
        calibration_limit=10,
        calibration_seq_length=100,
        pad_calibration_inputs=False,
    ):
        from build.model import find_multiple
        self.mod = mod
        self.tokenizer=tokenizer
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.blocksize = blocksize
        self.percdamp=percdamp
        self.calibration_limit = calibration_limit
        self.calibration_tasks = calibration_tasks
        self.calibration_seq_length = calibration_seq_length
        self.pad_calibration_inputs = pad_calibration_inputs
        self.get_qparams_func = lambda w: get_group_qparams(w, 4, groupsize)
        self.quantize_func = lambda w, qparams: group_quantize_tensor_from_qparams(
            w, qparams[0], qparams[1], 4, groupsize
        )
        self.dequantize_func = lambda q, qparams: group_dequantize_tensor_from_qparams(
            q, qparams[0], qparams[1], 4, groupsize
        ).float()
        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]
        # skip unless padding_allowed=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], groupsize, inner_k_tiles)
            or padding_allowed
        )

        # we need to do the padding here, both for q and the qparams if necessary
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1]
            if not _check_linear_int4_k(k, groupsize, inner_k_tiles):
                new_k = find_multiple(k, 1024)
            else:
                new_k = k
            # how much we need to pad the weight
            delta_k = new_k - q.shape[1]
            final_q = torch.ops.aten._convert_weight_to_int4pack(
                F.pad(q, pad=(0, delta_k)), inner_k_tiles
            )
            scales_and_zeros = pack_scales_and_zeros(*qparams)
            # how many new groups we need for padded weight
            delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
            final_s_and_z = F.pad(
                scales_and_zeros, pad=(0, 0, 0, 0, 0, delta_groups), value=1
            )
            return {"weight": final_q, "scales_and_zeros": final_s_and_z}

        self.make_names_and_values_dict_func = make_names_and_values_dict_func
        super().__init__()

    def convert_for_runtime(self):
        replace_linear_int4(
            self.mod, self.device, self.groupsize, self.inner_k_tiles, self.padding_allowed
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict(
            tokenizer=self.tokenizer,
            groupsize=self.groupsize,
            blocksize=self.blocksize,
            percdamp=self.percdamp,
            calibration_tasks=self.calibration_tasks,
            calibration_limit=self.calibration_limit,
            calibration_seq_length=self.calibration_seq_length,
            pad_calibration_inputs=self.pad_calibration_inputs,
            device=self.device,
        )
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict, strict=False)
        return self.mod


##################################################################
###                           WIP: HQQ                         ###


class WeightOnlyInt4HqqQuantHandler:
    def __init__(self, mod, device, *, groupsize):
        self.mod = mod
        self.device = device
        self.groupsize = groupsize

    def create_quantized_state_dict(self):
        from hqq.core.quantize import Quantizer  # TODO maybe torchao

        for m in self.mod.modules():
            for _name, child in m.named_children():
                if isinstance(child, torch.nn.Linear):
                    child.weight = torch.nn.Parameter(
                        Quantizer.dequantize(
                            *Quantizer.quantize(
                                child.weight,
                                nbits=4,
                                groupsize=self.groupsize,
                                axis=1,
                            )
                        )
                    )

        # we use Int4 packaged in an int8 for now, packing to follow
        # return WeightOnlyInt4QuantHandler(self.mod, self.groupsize).create_quantized_state_dict()
        return WeightOnlyInt8QuantHandler(
            self.mod, self.device, bitwidth=4, groupsize=self.groupsize
        ).create_quantized_state_dict()

    def convert_for_runtime(self):
        # we use Int4 packaged in an int8 for now, packing to follow
        # ALSO: all code must work for CPU, CUDA, MPS
        # return WeightOnlyInt4GPTQQuantHandler(self.mod, self.groupsize).convert_for_runtime()
        return WeightOnlyInt4GPTQQuantHandler(
            self.mod, self.device, bitwidth=4, groupsize=self.groupsize
        ).convert_for_runtime()

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


##################################################################

##########################################################################
###                  process quantization dictionary                   ###

# Map each quantizer configuration to a class implementing that quantizer
quantizer_class_dict = {
    "embedding": EmbeddingOnlyInt8QuantHandler,
    "linear:int8": WeightOnlyInt8QuantHandler,
    "linear:int4": WeightOnlyInt4QuantHandler,
    "linear:a8w4dq": Int8DynActInt4WeightQuantizer,
    "linear:int4-gptq": WeightOnlyInt4GPTQQuantHandler,
    "linear:hqq": WeightOnlyInt4HqqQuantHandler,
    "precision": PrecisionHandler,
}

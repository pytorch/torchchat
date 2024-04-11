# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
from math import gcd
from typing import Dict, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from GPTQ import GenericGPTQRunner, InputRecorder
    from eval import get_task_dict, evaluate, lm_eval
except:
    pass

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
        raise RuntimeError("unsupported dtype specified")
    
name_to_dtype_dict = {
    "fp32" : torch.float,
    "fp16" : torch.float16,
    "bf16" : torch.bfloat16,
    "float" : torch.float,
    "half" : torch.float16,
    "float32" : torch.float,
    "float16" : torch.float16,
    "bfloat16" : torch.bfloat16,
}

##########################################################################
###                  process quantization dictionary                   ###

def quantize_model(model: nn.Module, quantize_options):
    """
    Quantize the specified model using the quantizers described by
    a quantization dict of the form:
    {
        'embedding':   {'bitwidth': 8, 'group_size': 8 },
        'linear:int8': {'bitwidth': 8, 'group_size': 8},
        'precision':   {'dtype': torch.float16},
    }
    """

    linears_quantized = False
    if isinstance(quantize_options, str):
        quantize_options = json.loads(quantize_options)
        
    for quantizer, q_kwargs in quantize_options.items():
        if quantizer == "embedding":
            model = EmbeddingOnlyInt8QuantHandler(
                model,
                **q_kwargs
            ).quantized_model()
        elif linears_quantized:
            assert 0==1, "can only specify one linear quantizer"
        elif quantizer == "linear:int8":
            linears_quantized = True
            model = WeightOnlyInt8QuantHandler(
                model,
                **q_kwargs
            ).quantized_model()
        elif quantizer == "linear:int4":
            linears_quantized = True
            model = WeightOnlyInt4QuantHandler(
                model,
                **q_kwargs
            ).quantized_model()
        elif quantizer == "linear:a8w4dq":
            linears_quantized = True
            model = Int8DynActInt4WeightQuantHandler(
                model,
                **q_kwargs
            ).quantized_model()
        elif quantizer == "linear:gptq":
            linears_quantized = True
            model = WeightOnlyInt4GPTQQuantHandler(
                model,
                **q_kwargs
            ).quantized_model()
        elif quantizer == "linear:hqq":
            linears_quantized = True
            model = WeightOnlyInt4HqqQuantHandler(
                model,
                **q_kwargs
            ).quantized_model()
        elif quantizer == "precision":
            model.to(**q_kwargs)
        else:
            assert 0 == 1, f"quantizer {quantizer} not supported"
            
    
#########################################################################
#####                     Quantization Primitives                  ######

def dynamically_quantize_per_channel(
    x,
    quant_min,
    quant_max,
    target_dtype,
    group_size: Optional[int] = None,
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
        group_size: number of elements of the channel to quantize together

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

    if group_size is None or group_size == 0:
        items = x_shape_1
    elif ((x_shape_1 % group_size) == 0) or not enable_non_multiple_groups:
        assert group_size > 0, "group size must be positive"
        assert (
            x_shape_1 % group_size
        ) == 0, f"weights dimension 1 = {x_shape_1} must be a multiple of group size {group_size}"
        items = group_size
    else:
        assert group_size > 0, "group size must be positive"
        print(
            f"row-size of weight matrix {x_shape_1} is not divisible by group size {group_size}, using nearest neighbor rounding"
        )
        assert (
            x_shape_1 % group_size != 0
        ), f"expected x.shape[1] to not be a multiple of group size {group_size}, but got {x_shape_1}"
        padding = group_size - (x_shape_1 % group_size)
        x = F.pad(x, (0, padding))
        items = group_size

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



def get_group_qparams(w, n_bit=4, groupsize=128, *, scales_dtype= torch.float):
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
###                QuantHandler API definition                        ###


class QuantHandler:
    def __init__(self, mod):
        self.mod = mod

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
#####          Weight-only int8 per-channel quantized code         ######


def replace_linear_weight_only_int8_per_channel(module, node_type, group_size=None):
    if group_size is not None and group_size != 0:
        pass # group_size = 2 ** group_size
        
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
                    WeightOnlyInt8Linear(child.in_features, child.out_features, group_size),
                )
        else:
            replace_linear_weight_only_int8_per_channel(child, node_type, group_size)


class WeightOnlyInt8QuantHandler(QuantHandler):
    def __init__(
        self,
        mod,
        *,
        node_type: str = "*",
        bitwidth: Optional[int] = None,
        group_size: Optional[int] = None,
    ):
        self.mod = mod
        self.group_size = group_size
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
                    print(
                        f"quantize {self.node_type} {fqn, mod} with groupsize {self.group_size}, bitwidth {self.bitwidth}"
                    )

                    # print(f"initial weight shape {mod.weight.shape}")
                    input_weight = mod.weight.float()

                    # print(f"expanded weight shape {input_weight.shape}")
                    weight, scales, _ = dynamically_quantize_per_channel(
                        input_weight,
                        range_min,
                        range_max,
                        torch.int8,
                        self.group_size,
                        scales_dtype=mod.weight.dtype,
                    )

                    cur_state_dict[f"{fqn}.weight"] = weight
                    # squeeze makes groupsize=rowsize unidimensional
                    cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_linear_weight_only_int8_per_channel(self.mod, self.node_type, self.group_size)
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
        in_features: int,
        out_features: int,
        group_size: Optional[int] = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        print(f"group size: {group_size}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        if group_size is None or (group_size == 0):
            self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))
        else:
            groups = (in_features + group_size - 1) // group_size
            self.register_buffer("scales", torch.ones(out_features, groups, dtype=torch.bfloat16))

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
            return F.linear(input, (weight.to(dtype=input.dtype).view(weight.shape[0], no_groups, -1) * scales.view(weight.shape[0], no_groups, -1)).view(weight.shape[0], -1))


#########################################################################
#####                   embedding table quantization               ######


def replace_embedding_weight_only_grouped_int8_per_channel(
    module, bitwidth: int = 8, group_size: Optional[int] = None
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
                    vocab_size=child.weight.shape[0],
                    embedding_dim=child.weight.shape[1],
                    group_size=group_size,
                ),
            )
        else:
            replace_embedding_weight_only_grouped_int8_per_channel(
                child, bitwidth, group_size
            )


class EmbeddingOnlyInt8QuantHandler(QuantHandler):
    def __init__(self, mod, *, bitwidth: int = 8, group_size: Optional[int] = None):
        self.mod = mod
        self.group_size = group_size
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
            if (
                isinstance(mod, nn.Embedding)
            ):
                # print("****")
                # print(f"Embedding identified: {fqn, mod}")
                # print(f"weights size: {mod.weight.size()}")
                # print(f"quantize {fqn}...")

                print(
                    f"quantize {fqn, mod} with groupsize {self.group_size}, bitwidth {self.bitwidth}"
                )
                weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(),
                    range_min,
                    range_max,
                    torch.int8,
                    self.group_size,
                    scales_dtype=mod.weight.dtype,
                )

                # Update state dict
                cur_state_dict[f"{fqn}.weight"] = weight
                # squeeze makes groupsize=rowsize unidimensional
                cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_embedding_weight_only_grouped_int8_per_channel(
            self.mod, self.bitwidth, self.group_size
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class QuantizedGroupEmbedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        group_size: Optional[int] = None,
        device=None,
        dtype=torch.half,
    ) -> None:
        super().__init__()
        if group_size is None or group_size == 0:
            group_size = embedding_dim
        self.group_size = group_size
        self.dtype = dtype
        self.register_buffer(
            "weight", torch.empty((vocab_size, embedding_dim), dtype=torch.int8)
        )
        groups_per_row = (embedding_dim + group_size - 1) // group_size
        if groups_per_row > 1:
            self.register_buffer(
                "scales", torch.ones((vocab_size, groups_per_row), dtype=torch.float16)
            )
        else:
            self.register_buffer(
                "scales", torch.ones((vocab_size,), dtype=torch.float16)
            )

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if False:   # Used for Executorch
            return torch.ops.llama_quantized.embedding_byte.dtype(
                self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype
            )

        
        # result_weights = self.weight.index_select(0, indices.view(-1))
        # result_scales = self.scales.index_select(0, indices.view(-1))

        weight = self.weight
        scales = self.scales.view(weight.shape[0], -1)
        
        result_weights = F.embedding(indices, weight)
        result_scales = F.embedding(indices, scales)

        rw_view = result_weights.to(dtype=result_scales.dtype).view(tuple(result_weights.shape[:-1] + (scales.shape[1], -1, )))
        rs_view = result_scales.view(tuple(result_scales.shape[:-1]) + (scales.shape[1], 1, ))
        # print(f"rw_view {rw_view.shape}")
        # print(f"rs_view {rs_view.shape}")

        r = rw_view * rs_view
        return r.view(indices.size() + (-1,))
        
        # r = result_weights.to(dtype=result_scales.dtype).view(list(result_weights.shape[:-1] + (scales.shape[1], -1, )) * result_scales.view(scales.shape[-1] + (scales.shape[1], 1, ))

        
#########################################################################
#####     weight only int4 per channel groupwise quantized code    ######

def _int4_prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros

def _int4_calc_padded_size(k, groupsize=1, innner_k_tiles=1):
    from model import find_multiple
    return find_multiple(k, 1024)

def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(
        x.to(dtype=torch.bfloat16),
        weight_int4pack,
        groupsize,
        scales_and_zeros.to(dtype=torch.bfloat16)
    ).to(dtype=x.dtype)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


def _int4_check_linear_int4_k(k, groupsize = 1, inner_k_tiles = 1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0

def replace_linear_int4(module, groupsize, inner_k_tiles, padding_allowed, use_cuda=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _int4_check_linear_int4_k(child.in_features, groupsize, inner_k_tiles) or padding_allowed:
                setattr(module, name, WeightOnlyInt4Linear(
                    child.in_features, child.out_features, bias=False,
                    groupsize=groupsize, inner_k_tiles=inner_k_tiles, use_cuda=use_cuda
                ))
        else:
            replace_linear_int4(child, groupsize, inner_k_tiles, padding_allowed, use_cuda)


class WeightOnlyInt4QuantHandler(QuantHandler):
    def __init__(self, mod, group_size=128, inner_k_tiles=8, padding_allowed=True):
        self.mod = mod
        self.groupsize = group_size
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        assert group_size in [32, 64, 128, 256]
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
                print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = mod.weight.data
                if not _int4_check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding_allowed:
                        from model import find_multiple
                        import torch.nn.functional as F
                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
                            "and that groupsize and inner_k_tiles*16 evenly divide into it")
                        continue
                weight_int4pack, scales_and_zeros = _int4_prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.float), self.groupsize, self.inner_k_tiles
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to('cpu')
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to('cpu')

        return cur_state_dict


    def convert_for_runtime(self, use_cuda=False):
        replace_linear_int4(self.mod, self.groupsize, self.inner_k_tiles, self.padding_allowed, use_cuda)
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
            self, in_features: int, out_features: int,
            bias=True, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8, use_cuda=True,
    ) -> None:
        super().__init__()
        self.padding = not _int4_check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            from model import find_multiple
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
        )
        # MKG: torch.float
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.float)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # MKG torch.float
        input = input.to(torch.float)
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )

#########################################################################
#####           Int8 Dynamic Activations 4 Bit Weights              #####

def prepare_int4_weight_and_scales_and_zeros(weight, group_size, precision):
    weight_int8, scales, zeros = group_quantize_tensor_symmetric(
        weight,
        n_bit=4,
        group_size=group_size,
        precision=precision,
    )
    # TODO: better API
    # weight_int4packed = torch.ops.quantized_decomposed.pack_int4_from_int8(weight_int8)
    return weight_int8, scales, zeros


def linear_forward_8da4w(
    x, weight_int8, scales, zeros, out_features, group_size, precision
):
    x = per_token_dynamic_quant(x)
    # TODO: verify and remove following reshape code
    # origin_x_size = x.size()
    # x = x.reshape(-1, origin_x_size[-1])

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
        group_size,
        precision,
    )

    # x = x.to(torch.float16)
    # w_dq = w_dq.to(torch.float16)
    c = torch.nn.functional.linear(x, w_dq)

    # new_shape = origin_x_size[:-1] + (out_features,)
    # c = c.reshape(new_shape)

    return c


def find_multiple(n: int, *args: Tuple[int]) -> int:
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)


def _check_linear_int4_k(k, group_size=1):
    return k % group_size == 0

def _calc_padded_size_linear_int4(k, groupsize=1):
    return find_multiple(k, groupsize)


def replace_linear_8da4w(
    module,
    group_size,
    padding_allowed,
    precision,
    scales_precision,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _check_linear_int4_k(child.in_features, group_size) or padding_allowed:
                setattr(
                    module,
                    name,
                    Int8DynActInt4WeightLinear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        group_size=group_size,
                        precision=precision,
                        scales_precision=scales_precision,
                    ),
                )
        else:
            replace_linear_8da4w(
                child,
                group_size,
                padding_allowed,
                precision,
                scales_precision,
            )


class Int8DynActInt4WeightQuantHandler(QuantHandler):
    def __init__(
        self,
        mod,
        group_size=256,
        padding_allowed=False,
        precision=torch.float32,
        scales_precision=torch.float32,
    ):
        self.mod = mod
        self.group_size = group_size
        self.padding_allowed = padding_allowed
        self.precision = precision
        self.scales_precision = scales_precision
        # assert group_size in [32, 64, 128, 256]

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                in_features = mod.in_features
                # print("in features:", in_features, " out features:", out_features)
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                # print(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert (
                    in_features % self.group_size == 0
                ), f"require in_features:{in_features} % self.group_size:{self.group_size} == 0"

                weight = mod.weight.data
                """
                if not _check_linear_int4_k(
                    in_features, self.group_size
                ):
                    if self.padding_allowed:
                        print(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = _calc_padded_size_linear_int4(
                            in_features, self.group_size
                        )
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        raise RuntimeError(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that group_size"
                        )
                """
                (
                    weight_int4pack,
                    scales,
                    zeros,
                ) = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(self.precision),
                    self.group_size,
                    self.scales_precision,
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales"] = scales.to("cpu")
                cur_state_dict[f"{fqn}.zeros"] = zeros.to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_8da4w(
            self.mod,
            self.group_size,
            self.padding_allowed,
            self.precision,
            self.scales_precision,
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Parameters of importance
    group_size: the number of elements in each quantized group
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    scales_precision: precision of per group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        group_size: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        assert (
            in_features % group_size == 0
        ), f"require in_features:{in_features} % group_size:{group_size} == 0"
        # in_features = _calc_padded_size_linear_int4(
        #    in_features, group_size
        # )
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.group_size = group_size
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=scales_precision,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=scales_precision,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        # padding is removed for perf
        # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_8da4w(
            input,
            self.weight,
            self.scales,
            self.zeros,
            self.out_features,
            self.group_size,
            self.precision,
        )


#########################################################################
#####                           GPTQ                                #####

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
    def get_inputs(model, tokenizer, calibration_tasks, calibration_limit, calibration_seq_length, pad_calibration_inputs) -> "MultiInput":
        input_recorder = InputRecorder(
            model,
            tokenizer,
            calibration_seq_length,
            pad_calibration_inputs,
        )

        try:
            lm_eval.tasks.initialize_tasks()
        except:
            pass
        task_dict = get_task_dict(calibration_tasks)
        print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

        evaluate(
            input_recorder,
            task_dict,
            limit=calibration_limit,
        )
        inputs = input_recorder.get_recorded_inputs()
        assert inputs is not None, (
            f"No inputs were collected, use a task other than {calibration_tasks}, "+
            f"use option pad_calibration_inputs, or decrease calibration_sequence_length (currently "+
            f"{calibration_seq_length})"
        )
        print(f"Obtained {len(inputs[0].values)} calibration samples")
        return inputs

    @torch.no_grad()
    def create_quantized_state_dict(
        self,
        tokenizer,
        blocksize,
        percdamp,
        groupsize,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
    ) -> "StateDict":
        inputs = GPTQQuantHandler.get_inputs(self.mod, tokenizer, calibration_tasks, calibration_limit, calibration_seq_length, pad_calibration_inputs)
        print("Tracing model for GPTQ")
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
            self.skip_layer_func
        )

        print("Applying GPTQ to weights")
        GPTQ_runner.run()
        return GPTQ_runner.get_quantized_state_dict()

    def convert_for_runtime(self) -> "nn.Module":
        pass


class WeightOnlyInt4GPTQQuantHandler(GPTQQuantHandler):
    def __init__(self, mod, groupsize=128, inner_k_tiles=8, padding=True):
        from model import find_multiple
        self.mod = mod
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding = padding
        self.get_qparams_func = lambda w: get_group_qparams(w, 4, groupsize)
        self.quantize_func = lambda w, qparams: \
            group_quantize_tensor_from_qparams(w, qparams[0], qparams[1], 4, groupsize)
        self.dequantize_func = lambda q, qparams: \
            group_dequantize_tensor_from_qparams(q, qparams[0], qparams[1], 4, groupsize).float()
        self.combine_qparams_list_func = lambda qparams_list: \
            [torch.cat(x, dim=1) for x in zip(*qparams_list)]
        # skip unless padding=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], groupsize, inner_k_tiles) or padding
        )
        # we need to do the padding here, both for q and the qparams if necessary
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1]
            new_k = find_multiple(k, 1024)
            # how much we need to pad the weight
            delta_k = new_k - q.shape[1]
            final_q = torch.ops.aten._convert_weight_to_int4pack(F.pad(q, pad=(0, delta_k)), inner_k_tiles)
            scales_and_zeros = pack_scales_and_zeros(*qparams)
            # how many new groups we need for padded weight
            delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
            final_s_and_z = F.pad(scales_and_zeros, pad=(0,0,0,0,0, delta_groups), value=1)
            return {"weight": final_q, "scales_and_zeros": final_s_and_z}
        self.make_names_and_values_dict_func = make_names_and_values_dict_func
        super().__init__()


    def convert_for_runtime(self, use_cuda):
        replace_linear_int4(self.mod, self.groupsize, self.inner_k_tiles, self.padding, use_cuda)
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod
    


# class Int8DynActInt4WeightGPTQQuantHandler(GPTQQuantHandler):
#     def __init__(
#         self,
#         groupsize=128,
#         inner_k_tiles=8,
#         padding_allowed=True,
#         precision=torch.float32,
#     ):

#         self.groupsize = groupsize
#         self.inner_k_tiles = inner_k_tiles
#         self.padding_allowed = padding_allowed
#         self.precision = precision
#         self.dyn_quant_func = lambda x: per_token_dynamic_quant(x)
#         n_bit = 4
#         self.get_qparams_func = lambda w: get_group_qparams_symmetric(
#             w, n_bit, groupsize, self.precision
#         )
#         quant_min = -(2 ** (n_bit - 1))
#         quant_max = 2 ** (n_bit - 1) - 1
#         self.quantize_func = lambda w, qparams: torch.ops.quantized_decomposed.quantize_per_channel_group(
#             w, qparams[0], qparams[1], quant_min, quant_max, torch.int8, groupsize
#         )
#         self.dequantize_func = lambda q, qparams: torch.ops.quantized_decomposed.dequantize_per_channel_group(
#             q,
#             qparams[0],
#             qparams[1],
#             quant_min,
#             quant_max,
#             torch.int8,
#             groupsize,
#             self.precision,
#         )
#         self.combine_qparams_list_func = lambda qparams_list: [
#             torch.cat(x, dim=1) for x in zip(*qparams_list)
#         ]
#         # skip unless padding_allowed=True or its correctly sized
#         self.skip_layer_func = lambda linear_weight: not (
#             _check_linear_int4_k(linear_weight.shape[-1], groupsize, inner_k_tiles)
#             or padding_allowed
#         )

#         # we need to do the padding here, both for q and the qparams if necessary
#         def make_names_and_values_dict_func(q, qparams):
#             k = q.shape[1]
#             new_k = _calc_padded_size_linear_int4(k, groupsize, inner_k_tiles)
#             # how much we need to pad the weight
#             delta_k = new_k - q.shape[1]
#             final_q = F.pad(q, pad=(0, delta_k))
#             scales_and_zeros = pack_scales_and_zeros(*qparams, precision=self.precision)
#             # how many new groups we need for padded weight
#             delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
#             # TODO: split scales and zero_points
#             final_s_and_z = F.pad(
#                 scales_and_zeros, pad=(0, 0, 0, 0, 0, delta_groups), value=1
#             )
#             return {"weight": final_q, "scales_and_zeros": final_s_and_z}

#         self.make_names_and_values_dict_func = make_names_and_values_dict_func
#         super().__init__()

#     def convert_for_runtime(self, model):
#         replace_linear_8da4w(
#             model,
#             self.groupsize,
#             self.padding_allowed,
#             torch.int8,
#             self.precision,
#         )
#         return model

##################################################################
###                           WIP: HQQ                         ###

class WeightOnlyInt4HqqQuantHandler:
    def __init__(self, mod, group_size):
        self.mod = mod
        self.groupsize = group_size

    def create_quantized_state_dict(self):
        from hqq.core.quantize import Quantizer  # TODO maybe torchao

        
        for m in self.mod.modules():
            for name, child in m.named_children():
                if isinstance(child, torch.nn.Linear):
                    child.weight = torch.nn.Parameter(
                        Quantizer.dequantize(
                            *Quantizer.quantize(
                                child.weight,
                                nbits=4,
                                group_size=self.groupsize,
                                axis=1,
                            )
                        )
                    )

        # we use Int4 packaged in an int8 for now, packing to follow
        # return WeightOnlyInt4QuantHandler(self.mod, self.groupsize).create_quantized_state_dict()
        return WeightOnlyInt8QuantHandler(
            self.mod, bitwidth=4, group_size=self.groupsize
        ).create_quantized_state_dict()

    def convert_for_runtime(self):
        # we use Int4 packaged in an int8 for now, packing to follow
        # ALSO: all code must work for CPU, CUDA, MPS
        # return WeightOnlyInt4GPTQQuantHandler(self.mod, self.groupsize).convert_for_runtime(use_cuda=True)
        return WeightOnlyInt4GPTQQuantHandler(
            self.mod, bitwidth=4, group_size=self.groupsize
        ).convert_for_runtime()
    
    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


##################################################################


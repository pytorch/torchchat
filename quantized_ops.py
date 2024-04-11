# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from torch.library import impl, impl_abstract

torchat_lib = torch.library.Library(
    "torchat", "DEF"
)

torchat_lib.define(
    "embedding_int8(Tensor input, Tensor weight, "
    "Tensor scales) -> Tensor",
)

@impl(torchat_lib, "embedding_int8", "CompositeExplicitAutograd")
def embedding_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    indices = input
    # embedding_byte_weight_checks(weight, weight_scales, weight_zero_points)
    group_size = weight.size(1) // (
        scales.size(1) if scales.dim() == 2 else 1
    )
    # ET definition
    if False:
        weight_zero_points = None
        weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            weight.dtype,
            group_size,
            weight_scales.dtype,
        )
        return torch.ops.aten.embedding.default(weight, indices)

    scales = scales.view(weight.shape[0], -1)   
    result_weights = F.embedding(indices, weight)
    result_scales = F.embedding(indices, scales)

    rw_view = result_weights.to(dtype=result_scales.dtype).view(tuple(result_weights.shape[:-1]) + (scales.shape[1], -1, ))
    rs_view = result_scales.view(tuple(result_scales.shape[:-1]) + (scales.shape[1], 1, ))
    # print(f"rw_view {rw_view.shape}")
    # print(f"rs_view {rs_view.shape}")

    r = rw_view * rs_view
    return r.view(indices.size() + (-1,))
        
        
torchat_lib.define(
    "linear_int8(Tensor input, Tensor weight, Tensor scales, "
    "Tensor bias = None) -> Tensor",
)

@impl(torchat_lib, "linear_int8", "CompositeExplicitAutograd")
def linear_int8(
        input: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert bias is None, "bias != None not implemented"
    
    scales = scales.view(scales.shape[0], -1)
    no_groups = scales.shape[1]

    # for now, we special-case channel-wise, because we know how to
    # make that fast with Triton 
    if scales.shape[1] == 1:
        return F.linear(input, weight.to(dtype=input.dtype)) * scales
    else:
        return F.linear(
            input,
            (weight.to(dtype=input.dtype).view(weight.shape[0],no_groups, -1)
             * scales.view(weight.shape[0], no_groups, -1)
            ).view(weight.shape[0], -1)
        )



torchat_lib.define(
    "linear_int4(Tensor input, Tensor weight, Tensor scales_and_zeros, "
    "Tensor bias=None, *, int groupsize, int origin_in_features, "
    "int int_features, int out_features, bool padding = True) -> Tensor",
)

@impl(torchat_lib, "linear_int4", "CompositeExplicitAutograd")
def linear_int4(
        input: torch.Tensor,
        weight: torch.Tensor,
        scales_and_zeros: torch.Tensor,
        bias: torch.Tensor,
        *,
        groupsize: int,
        origin_in_features: int,
        in_features: int,
        out_features: int,
        padding: bool = True,
) -> torch.Tensor:
    assert bias is None, "bias != None not implemented"

    if padding:
        input = F.pad(input, pad=(0, in_features - origin_in_features))

    # the weight is in int4pack format
    # rename to remind ourselves of that
    weight_int4pack = weight
    
    origin_input_size = input.size()
    input = input.reshape(-1, origin_input_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(
        input.to(dtype=torch.bfloat16),
        weight_int4pack,
        groupsize,
        scales_and_zeros.to(dtype=torch.bfloat16)
    ).to(dtype=input.dtype)
    new_shape = origin_input_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


torchat_lib.define(
    "linear_a8w4dq(Tensor input, Tensor weight, Tensor scales, "
    "Tensor zeros, int out_features, int group_size, "
    "dtype precision) -> Tensor",
)

@impl(torchat_lib, "linear_a8w4dq", "CompositeExplicitAutograd")
def linear_a8w4dq(
    input, weight, scales, zeros, out_features, group_size, precision
):
    x = per_token_dynamic_quant(input)
    weight_int8 = weight
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

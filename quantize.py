# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

# from functools import reduce
# from math import gcd
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from build.utils import (
    find_multiple,
    get_device_str,
    get_precision,
    name_to_dtype,
    state_dict_device,
)

from qops import (
    LinearAct8Int4DQ,
    LinearInt4 as WeightOnlyInt4Linear,
    LinearInt8 as WeightOnlyInt8Linear,
    QuantizedEmbedding,
)


#########################################################################
###                  torchchat quantization API                       ###


def quantize_model(model: nn.Module, device, quantize_options, tokenizer=None):
    """
    Quantize the specified model using the quantizers described by
    a quantization dict of the form:
    {
        'embedding':   {'bitwidth': 8, 'groupsize': 8},
        'linear:int8': {'bitwidth': 8, 'groupsize': 8},
        'precision':   {'dtype': torch.float16},
    }
    """

    if isinstance(quantize_options, str):
        quantize_options = json.loads(quantize_options)

    for quantizer, q_kwargs in quantize_options.items():
        if quantizer not in quantizer_class_dict:
            raise RuntimeError(f"unknown quantizer {quantizer} specified")

        model = quantizer_class_dict[quantizer](
            model, device=device, tokenizer=tokenizer, **q_kwargs
        ).quantized_model()


#########################################################################
###                QuantHandler API definition                        ###
###               (unify with torchao in future)                      ###


class QuantHandler:
    def __init__(self, model: nn.Module, device="cpu", tokenizer=None):
        self.model_ = model
        self.device = device
        self.tokenizer = tokenizer

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = state_dict_device(self.create_quantized_state_dict())
        self.convert_for_runtime()
        self.model_.load_state_dict(model_updated_state_dict)
        return self.model_


#########################################################################
###           wrapper for setting precision as a QuantHandler         ###


class PrecisionHandler(QuantHandler):
    def __init__(self, model: nn.Module, device="cpu", tokenizer=None, *, dtype):
        self.model_ = model
        self.device = device
        self.tokenizer = tokenizer

        if isinstance(dtype, str):
            dtype = name_to_dtype(dtype, device)
        self.dtype = dtype

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        return self.model_.to(device=self.device, dtype=self.dtype)


#########################################################################
###            wrapper for setting device as a QuantHandler           ###
###    for onw select device for PyTorch eager and AOTI, in future    ###
###    also use this for selecting delegate when exporting with ET    ###


class ExecutorHandler(QuantHandler):
    def __init__(self, model: nn.Module, device="cpu", tokenizer=None, *, accelerator):
        self.model_ = model

        if isinstance(accelerator, str):
            device = get_device_str(accelerator)
        self.device = device

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        return self.model_.to(device=self.device)


class CustomHandler(QuantHandler):
    def __init__(self, model: nn.Module, device="cpu", tokenizer=None):
        self.model_ = model
        self.device = device
        self.tokenizer = tokenizer

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass

    def quantized_model(self) -> nn.Module:
        self.model_ = self.model_.to(device=self.device)
        from custom_linear_8da4w._custom_linear import (
            _replace_linear_with_custom_linear,
        )

        _replace_linear_with_custom_linear(self.model_)
        return self.model_


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


class WeightOnlyInt8QuantHandler(QuantHandler):
    def __init__(
        self,
        model: nn.Module,
        device,
        tokenizer=None,
        *,
        node_type: str = "*",
        bitwidth: Optional[int] = None,
        groupsize: Optional[int] = None,
    ):
        self.model_ = model
        self.device = device
        self.groupsize = groupsize
        self.node_type = node_type
        if bitwidth is None:
            self.bitwidth = 8
        else:
            self.bitwidth = bitwidth

    @torch.no_grad()
    def quantize(self, module):
        # cur_state_dict = state_dict_device(self.model_.state_dict())
        # dict_device = "cpu"  # self.device

        if self.bitwidth == 4:
            range_min = -8
            range_max = 7
        elif self.bitwidth == 8:
            range_min = -128
            range_max = 127
        else:
            raise ValueError(f"Unsupported bitwidth {self.bitwidth}")

        for name, child in module.named_children():
            # print(f"name: {name}")
            if isinstance(child, nn.Linear):
                if (
                    (self.node_type == "*")
                    or (self.node_type == "output" and name == "output")
                    or (self.node_type == "!output" and name != "output")
                ):
                    # print(f"{name, child}")
                    input_weight = child.weight.float()
                    # print(f"{name, child}")
                    # print(f"in_features: {child.in_features}")
                    # print(f"out_features: {child.out_features}")

                    # print(f"expanded weight shape {input_weight.shape}")
                    weight, scales, _ = dynamically_quantize_per_channel(
                        input_weight,
                        range_min,
                        range_max,
                        torch.int8,
                        self.groupsize,
                        scales_dtype=child.weight.dtype,
                    )

                    setattr(
                        module,
                        name,
                        WeightOnlyInt8Linear(
                            in_features=child.in_features,
                            out_features=child.out_features,
                            device=self.device,
                            # update variables from quantization
                            weight=weight,
                            scales=scales,
                            groupsize=self.groupsize,
                        ),
                    )
            else:
                self.quantize(child)

        return module

    def quantized_model(self) -> nn.Module:
        return self.quantize(self.model_)


#########################################################################
#####                   embedding table quantization               ######


class EmbeddingOnlyQuantHandler(QuantHandler):
    def __init__(
        self,
        model: nn.Module,
        device,
        tokenizer=None,
        *,
        bitwidth: int = 8,
        groupsize: Optional[int] = None,
        packed=True,  # we always pack bitwidth 4 now
    ):
        self.model_ = model
        self.device = device
        self.groupsize = groupsize
        self.bitwidth = bitwidth

    @torch.no_grad()
    def quantize(self, module):
        if self.bitwidth == 4:
            range_min = -8
            range_max = 7
        elif self.bitwidth == 8:
            range_min = -128
            range_max = 127
        else:
            raise ValueError(f"Unsupported bitwidth {self.bitwidth}")

        for name, child in module.named_children():
            # print(f"name: {name}")
            if isinstance(child, nn.Embedding):
                # print(f"Embedding identified: {fqn, mod}")
                # print(f"weights size: {child.weight.size()}")
                # print(f"quantize {fqn}...")

                # print(
                #     f"quantize {fqn, mod} with groupsize {self.groupsize}, bitwidth {self.bitwidth}"
                # )
                weight, scales, _ = dynamically_quantize_per_channel(
                    child.weight.float(),
                    range_min,
                    range_max,
                    torch.int8,
                    self.groupsize,
                    scales_dtype=child.weight.dtype,
                )

                if self.bitwidth == 4:
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

                weight = weight
                scales = scales.squeeze(dim=-1)

                # print(f"{name, child}")
                # print(f"weights size: {child.weight.size()}")
                setattr(
                    module,
                    name,
                    QuantizedEmbedding(
                        num_embeddings=child.weight.shape[0],
                        embedding_dim=child.weight.shape[1],
                        device=self.device,
                        bitwidth=self.bitwidth,
                        groupsize=self.groupsize,
                        weight=weight,
                        scales=scales,
                    ),
                )
            else:
                self.quantize(child)

        return module

    def quantized_model(self) -> nn.Module:
        return self.quantize(self.model_)


#########################################################################
#####     weight only int4 per channel groupwise quantized code    ######


class WeightOnlyInt4QuantHandler(QuantHandler):
    def __init__(
        self,
        model: nn.Module,
        device=None,
        *,
        tokenizer=None,
        groupsize=128,
        inner_k_tiles=8,
        padding_allowed=True,
    ):
        self.model_ = model
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    @torch.no_grad()
    def quantize(self, module):
        for name, child in module.named_children():
            # print(f"name: {name}")
            if isinstance(child, torch.nn.Linear):
                assert not child.bias
                out_features = child.out_features
                in_features = child.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                # print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = child.weight.data
                if not WeightOnlyInt4Linear._check_k(
                    k=in_features,
                    groupsize=self.groupsize,
                    inner_k_tiles=self.inner_k_tiles,
                ):
                    if self.padding_allowed:
                        print(
                            f"warning: {name} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        print(
                            f"warning: {name} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = (
                    WeightOnlyInt4Linear._prepare_weight_and_scales_and_zeros(
                        weight.to(torch.float), self.groupsize, self.inner_k_tiles
                    )
                )
                weight_int4pack = weight_int4pack.to(device=self.device)
                scales_and_zeros = scales_and_zeros.to(device=self.device)

                setattr(
                    module,
                    name,
                    WeightOnlyInt4Linear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        device=self.device,
                        groupsize=self.groupsize,
                        inner_k_tiles=self.inner_k_tiles,
                        weight=weight_int4pack,
                        scales_and_zeros=scales_and_zeros,
                    ),
                )
            else:
                self.quantize(child)

        return module

    def quantized_model(self) -> nn.Module:
        return self.quantize(self.model_)


#########################################################################
#####     weight only int4 per channel groupwise quantized code    ######


class Int8DynActInt4WeightQuantizer(QuantHandler):
    def __init__(
        self,
        model: nn.Module,
        device=None,
        dtype=None,
        *,
        tokenizer=None,
        groupsize=128,
        padding_allowed=True,
        precision=torch.float32,
        scales_precision=torch.float32,
    ):
        if dtype is None:
            dtype = torch.float32

        self.model_ = model
        self.device = device
        self.dtype = dtype

        self.groupsize = groupsize
        self.padding_allowed = padding_allowed
        self.precision = precision
        self.scales_precision = scales_precision
        assert groupsize in [32, 64, 128, 256]

    @torch.no_grad()
    def quantize(self, module):
        from torchao.quantization.quant_primitives import (
            group_quantize_tensor_symmetric,
        )

        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                out_features = child.out_features
                in_features = child.in_features
                weight = child.weight.data
                assert not child.bias
                assert out_features % 8 == 0, "require out_features % 8 == 0"

                padding_multiple = self.groupsize
                padded_in_features = find_multiple(in_features, padding_multiple)
                weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                (
                    weight_int8,
                    scales,
                    zeros,
                ) = group_quantize_tensor_symmetric(
                    weight.float(),
                    4,  # n_bit
                    self.groupsize,
                    self.scales_precision,
                )
                setattr(
                    module,
                    name,
                    LinearAct8Int4DQ(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        device=self.device,
                        dtype=self.dtype,
                        groupsize=self.groupsize,
                        weight=weight_int8.to(device=self.device),
                        scales=scales.to(device=self.device),
                    ),
                )
            else:
                self.quantize(child)

        return module

    def quantized_model(self) -> nn.Module:
        return self.quantize(self.model_)


#########################################################################
#####                           GPTQ                                #####


class GPTQQuantHandler(QuantHandler):
    """This class implements a GPTQ QuantHandler that can be used to
    apply GPTQ to a model in concert with the GenericGPTQRunner class.
    Unlike the base QuantHandler class, the user does not need to
    implement the create_quantized_state_dict, instead they have to
    reimplement __init__ such that it defines the functions for the
    quantization mode. User is expected to reimplement
    convert_for_runtime.

    The following functions (which must be defined in __init__) are
    used to define the quantization mode for both GPTQ and
    create_quantized_state_dict. Here is a description of each
    function.

    get_qparams_func:
        A function that calculates the quantization qparams for an input tensor.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            qparams: it can have any format but will need to be handled by the other defined functions below.

    quantize_func:
        A function that applies quantization to an input tensor. It
        should be noted that this function needs to be able to handle
        quantizing the entire weight tensor, a single group, or a
        single column.

        Args:
            weight: A 2d weight tensor with non-integer dtype.
            qparams: the output from get_qparams_func
        Returns:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)


    dequantize_func:
        A function that dequantizes an input quantized weight
        tensor. It should be noted that this function needs to be able
        to handle dequantizing the entire weight tensor, a single
        group, or a single column.

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
        A function that prepares the qparams and quantized_weight and
        creates a dictionary indicating how they should be inserted
        into the state_dict. Generally any packing of the weight and
        qparams should be done here.

        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            names_and_values_dict: a dictionary mapping the name of
            the parameters of the quantized module to the
            corresponding quantized weights and qparams.
    """

    def __init__(self):
        assert self.model_ is not None
        assert self.get_qparams_func is not None
        assert self.quantize_func is not None
        assert self.dequantize_func is not None
        assert self.combine_qparams_list_func is not None
        assert self.make_names_and_values_dict_func is not None

    @staticmethod
    def get_inputs(
        model: nn.Module,
        tokenizer,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
        device,
    ) -> "MultiInput":
        from eval import evaluate, get_task_dict
        from GPTQ import InputRecorder

        try:
            import lm_eval

            lm_eval.tasks.initialize_tasks()
        except:
            pass

        input_recorder = InputRecorder(
            model,
            tokenizer,
            calibration_seq_length,
            pad_calibration_inputs,
            device,
        )

        task_dict = get_task_dict(calibration_tasks)
        print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)
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
            self.model_,
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
            self.model_,
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
    def __init__(
        self,
        model: nn.Module,
        device,
        tokenizer,
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
        self.model_ = model
        self.tokenizer = tokenizer
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.blocksize = blocksize
        self.percdamp = percdamp
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
            padding_allowed
            or WeightOnlyInt4Linear._check_k(
                k=linear_weight.shape[-1],
                groupsize=groupsize,
                inner_k_tiles=inner_k_tiles,
            )
        )

        # we need to do the padding here, both for q and the qparams if necessary
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1]
            if not WeightOnlyInt4Linear._check_k(
                k=k, groupsize=groupsize, inner_k_tiles=inner_k_tiles
            ):
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

    def replace_linear_int4(
        self,
        module,
    ):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if self.padding_allowed or WeightOnlyInt4Linear._check_k(
                    k=child.in_features,
                    groupsize=self.groupsize,
                    inner_k_tiles=self.inner_k_tiles,
                ):
                    setattr(
                        module,
                        name,
                        WeightOnlyInt4Linear(
                            child.in_features,
                            child.out_features,
                            bias=False,
                            device=self.device,
                            groupsize=self.groupsize,
                            inner_k_tiles=self.inner_k_tiles,
                        ),
                    )
            else:
                self.replace_linear_int4(child)

    def convert_for_runtime(self):
        self.replace_linear_int4(
            self.model_,
        )
        return self.model_

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
        self.model_.load_state_dict(model_updated_state_dict, strict=False)
        return self.model_


##################################################################
###                             HQQ                            ###


class WeightOnlyInt4HqqQuantHandler:
    def __init__(self, model: nn.Module, device, tokenizer=None, *, groupsize):
        self.model_ = model
        self.device = device
        self.groupsize = groupsize

    @torch.no_grad()
    def quantize(self, module):
        from hqq.core.quantize import Quantizer

        for m in self.model_.modules():
            for _name, child in m.named_children():
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

        return WeightOnlyInt4QuantHandler(
            model=self.model_, device=self.device, groupsize=self.groupsize
        ).quantize(self.model_)

    def quantized_model(self) -> nn.Module:
        return self.quantize(self.model_)


##########################################################################
###                       quantization dictionary                      ###

# Map each quantizer configuration to a class implementing that quantizer
# Must come last because __future__ annotations don't work for naked
# class references
quantizer_class_dict = {
    "embedding": EmbeddingOnlyQuantHandler,
    "linear:int8": WeightOnlyInt8QuantHandler,
    "linear:int4": WeightOnlyInt4QuantHandler,
    "linear:a8w4dq": Int8DynActInt4WeightQuantizer,
    "linear:int4-gptq": WeightOnlyInt4GPTQQuantHandler,
    "linear:hqq": WeightOnlyInt4HqqQuantHandler,
    "precision": PrecisionHandler,
    "executor": ExecutorHandler,
    "_custom": CustomHandler,
}

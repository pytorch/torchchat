from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from build.utils import (
    find_multiple,
    get_device_str,
    get_precision,
    name_to_dtype,
    state_dict_device,
    use_et_backend,
)
from torch.nn.parameter import Parameter


def linear_int8(input, weight, scales):
    n_groups = scales.numel() // scales.shape[0]

    # we special-case channel-wise, because we know how to make that fast
    if n_groups == 1:
        if (
            torch.compiler.is_compiling()
            or input.device.type != "cpu"
            or torch.__version__ < "2.4"
        ):
            return F.linear(input, weight.to(dtype=input.dtype)) * scales
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

        if device == "einputecutorch":
            device = "cpu"

        assert not bias, "Bias is not supported by LinearInt8"
        self.in_features = in_features
        self.out_features = out_features

        assert bool(weight) == bool(
            scales
        ), "must specify both weights and scales, or neither"
        if not weight:
            weight = torch.empty(
                (out_features, in_features), dtype=torch.int8, device=device
            )
            if groupsize is None or (groupsize == 0):
                scales = torch.empty(out_features, dtype=dtype, device=device)
            else:
                n_groups = (in_features + groupsize - 1) // groupsize
                scales = torch.empty(out_features, n_groups, dtype=dtype, device=device)

        self.register_buffer("weight", weight.to(device))
        self.register_buffer("scales", scales.to(device))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear_int8(input, self.weight, self.scales)


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
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = torch.half

        if groupsize is None or groupsize == 0:
            groupsize = embedding_dim
        self.groupsize = groupsize
        self.dtype = dtype
        self.bitwidth = bitwidth

        if use_et_backend():
            self.forward = self.et_forward
        else:
            self.forward = self.aoti_forward

        if bitwidth == 8:
            self.register_buffer(
                "weight",
                torch.empty(
                    (num_embeddings, embedding_dim), dtype=torch.int8, device=device
                ),
            )
        elif bitwidth == 4:  # packed
            self.register_buffer(
                "weight",
                torch.empty(
                    (num_embeddings, embedding_dim // 2),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        else:
            raise RuntimeError(
                f"QUantized embedding does not support bitwidth={bitwidth}"
            )

        groups_per_row = (embedding_dim + groupsize - 1) // groupsize
        if groups_per_row > 1:
            self.register_buffer(
                "scales",
                torch.ones(
                    (num_embeddings, groups_per_row), dtype=torch.float16, device=device
                ),
            )
        else:
            self.register_buffer(
                "scales",
                torch.ones((num_embeddings,), dtype=torch.float16, device=device),
            )

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

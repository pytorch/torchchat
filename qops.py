from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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

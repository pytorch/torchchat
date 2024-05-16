from typing import Optional

import torch
import torch.nn as nn

torch.ops.load_library(
    "/Users/scroy/repos/torchchat/custom_linear_8da4w/build/libcustom_linear_8da4w.dylib"
)
from .quantize import convert_to_qc4w, group_quantize_tensor_symmetric

class _CustomLinear(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.weight = weight
        assert bias is None
        torch.set_num_threads(8)

        self.group_size = 256
        w_int, s, z = group_quantize_tensor_symmetric(
            self.weight, self.group_size, torch.float32
        )
        w_packed = convert_to_qc4w(w_int)
        self.s = s
        self.w_packed = w_packed
        torch.set_num_threads(8)
        self.prepacked = torch.ops.torchchat.prepack.default(w_packed, s)

    def forward(self, x):
        assert x.shape[0] == 1
        x = x.squeeze(0)
        torch.set_num_threads(8)
        return torch.ops.torchchat.run.default(self.prepacked, x).unsqueeze(0)


def _replace_linear_with_custom_linear(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _CustomLinear(child.weight, child.bias))
        else:
            _replace_linear_with_custom_linear(child)

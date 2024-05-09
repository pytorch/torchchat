import torch
import torch.nn as nn
from typing import Optional
torch.ops.load_library("_custom_linear/build/libcustom_linear.dylib")
from .quantize import group_quantize_tensor_symmetric, convert_to_qc4w

class _CustomLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weight = weight
        assert bias is None

        self.group_size = 8
        w_int, s, z = group_quantize_tensor_symmetric(self.weight, self.group_size, torch.float32)
        w_packed = convert_to_qc4w(w_int)
        self.prepacked = torch.ops.torchchat.prepack.default(w_packed, s)

    def forward(self, x):
        if x.dtype != torch.float32:
            raise RuntimeError(f"x has dtype {x.dtype}, expected float32")
        assert x.shape[0] == 1
        return torch.ops.torchchat.run.default(self.prepacked, x.squeeze(0)).unsqueeze(0)

def _replace_linear_with_custom_linear(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _CustomLinear(child.weight, child.bias))
        else:
            _replace_linear_with_custom_linear(child)

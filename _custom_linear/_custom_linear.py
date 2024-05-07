import torch
import torch.nn as nn
from typing import Optional

class _CustomLinear(nn.Module):
    def _prepare(self) -> None:
        self.weight.requires_grad = False
        if self.bias:
            self.bias.requires_grad = False
        self.packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(self.weight, self.bias) #, -1, 1)

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self._prepare()

    def forward(self, x):
        if x.dtype != torch.float32:
            raise RuntimeError(f"x has dtype {x.dtype}, expected float32")
        return torch.ops.prepacked.linear_clamp_run(x, self.packed_weight_bias)

def _replace_linear_with_custom_linear(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _CustomLinear(child.weight, child.bias))
        else:
            _replace_linear_with_custom_linear(child)

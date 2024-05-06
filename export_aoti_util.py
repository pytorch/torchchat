import torch
import torch.nn as nn
from typing import Optional

class CustomLinear(nn.Module):
    def _prepare(self) -> None:
        print("Preparing weights")

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self._prepare()

    def forward(self, x):
        return torch.ops.aten.linear.default(x, self.weight, self.bias)

def replace_linear_with_custom_linear(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, CustomLinear(child.weight, child.bias))
        else:
            replace_linear_with_custom_linear(child)

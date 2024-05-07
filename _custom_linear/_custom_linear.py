import torch
import torch.nn as nn
from typing import Optional
torch.ops.load_library("_custom_linear/build/libcustom_linear.dylib")

class _CustomLinear(nn.Module):
    def _prepare(self) -> None:
        self.weight.requires_grad = False
        if self.bias:
            self.bias.requires_grad = False

        # self.packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(self.weight, self.bias)
        self.packed_weight_bias = torch.ops.torchchat.prepack.default(self.weight, self.bias, None, None)

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self._prepare()

    def forward(self, x):
        if x.dtype != torch.float32:
            raise RuntimeError(f"x has dtype {x.dtype}, expected float32")
        # return torch.ops.prepacked.linear_clamp_run(x, self.packed_weight_bias)
        return torch.ops.torchchat.run.default(x, self.packed_weight_bias)

def _replace_linear_with_custom_linear(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _CustomLinear(child.weight, child.bias))
        else:
            _replace_linear_with_custom_linear(child)

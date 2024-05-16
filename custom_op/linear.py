from typing import Optional

import torch
import torch.nn as nn
from random import randrange

torch.ops.load_library(
    "/Users/digantdesai/local/custom_op/build/libcustom_linear_op.dylib"
)

class CustomLinear(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        assert self.bias is None # TODO
        self.input_channels = self.weight.shape[1]
        self.output_channels = self.weight.shape[0]
        self.weight_key = torch.ops.customlinear.get_weights_key(self.weight)
        self.packed_weights = torch.ops.customlinear.prepack_weights(self.weight)

    def forward(self, x):
        print("Running custom linear")
        return torch.ops.customlinear.run(x, self.packed_weights, self.input_channels, self.output_channels, self.weight_key)

        # # Run the custom op without packed weights
        # return torch.ops.customlinear.prepack_and_run(x, self.weight)

        # # Run without the custom op
        # return torch.nn.functional.linear(x, self.weight, self.bias)


def replace_linear_with_custom_linear(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, CustomLinear(child.weight, child.bias))
        else:
            replace_linear_with_custom_linear(child)

if __name__ == "__main__":
    ic = randrange(2, 6);
    oc = randrange(2, 6);
    bs = randrange(2, 6);
    input = torch.rand((bs, ic))
    with torch.no_grad():
        model = nn.Sequential(nn.Linear(ic, oc, bias=False)).eval()
        ref = model(input)

        replace_linear_with_custom_linear(model)
        xnn = model(input)

        print(f"shapes: bs: {bs}, oc: {oc}, ic: {oc}")
        print(f"Functional: {ref}")
        print(f"Custom XNN: {xnn}")
        print(f"Success? {torch.allclose(ref, xnn)}")
        print("---")

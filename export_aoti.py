# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.export import Dim

default_device = "cpu"


def export_model(model: nn.Module, device, output_path, args=None):
    max_seq_length = 350

    input = (
        torch.tensor([[1, 9038, 2501, 263, 931]], dtype=torch.int, device=device),
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int, device=device),
    )

    seq = Dim("seq", min=1, max=max_seq_length)
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"idx": {1: seq}, "input_pos": {0: seq}}

    so = torch._export.aot_compile(
        model,
        args=input,
        options={"aot_inductor.output_path": output_path},
        dynamic_shapes=dynamic_shapes,
    )
    print(f"The generated DSO model can be found at: {so}")
    return so

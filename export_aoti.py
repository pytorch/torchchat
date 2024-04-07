# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim, export

from generate import _load_model, decode_one_token
from quantize import quantize_model

from model import Transformer

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def export_model(model: nn.Module, input, device, output_path, args=None):
    max_seq_length = 350
#    with torch.device(device):
#        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)


    print(f"len(input)={len(input)}")

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

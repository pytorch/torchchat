# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from torchchat.cli.builder import (
    _initialize_model,
    BuilderArgs,
)

from torchchat.utils.build_utils import set_precision

from torchao.quantization import quantize_, int8_weight_only

"""
Exporting Flow
"""


def main(args):
    builder_args = BuilderArgs.from_args(args)
    print(f"{builder_args=}")

    quant_format = "int8_wo"
    # Quant option from cli, can be None
    model = _initialize_model(builder_args, args.quantize)
    if not args.quantize:
        # Not using quantization option from cli;
        # Use quantize_() to quantize the model instead.
        print("Quantizing model using torchao quantize_")
        quantize_(model, int8_weight_only())
    else:
        print(f"{args.quantize=}")

    print(f"Model: {model}")

    # Save model
    model_dir = os.path.dirname(builder_args.checkpoint_path)
    model_dir = Path(model_dir + "-" + quant_format)
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass
    dest = model_dir / "model.pth"
    state_dict = model.state_dict()
    print(f"{state_dict.keys()=}")

    print(f"Saving checkpoint to {dest}. This may take a while.")
    torch.save(state_dict, dest)
    print("Done.")

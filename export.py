# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim, export

from quantize import quantize_model, name_to_dtype, set_precision, get_precision
from cli import cli_args

try:
    executorch_export_available = True
    from export_et import export_model as export_model_et
except Exception as e:
    executorch_exception = f"ET EXPORT EXCEPTION: {e}"
    executorch_export_available = False

from export_aoti import export_model as export_model_aoti

from model import Transformer
from builder import _initialize_model, BuilderArgs, TokenizerArgs
from generate import decode_one_token
from quantize import quantize_model, name_to_dtype
from torch._export import capture_pre_autograd_graph

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")



def main(args):
    builder_args = BuilderArgs.from_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    quantize = args.quantize

    print(f"Using device={builder_args.device}")
    set_precision(builder_args.precision)

    builder_args.dso_path = None
    builder_args.pte_path = None
    builder_args.setup_caches = True
    model = _initialize_model(
        builder_args,
        quantize,
    )

    output_pte_path = args.output_pte_path
    output_dso_path = args.output_dso_path

    with torch.no_grad():
        if output_pte_path:
            output_pte_path = str(os.path.abspath(output_pte_path))
            print(f">{output_pte_path}<")
            if executorch_export_available:
                print(f"Exporting model using Executorch to {output_pte_path}")
                export_model_et(model, builder_args.device, args.output_pte_path, args)
            else:
                print(f"Export with executorch requested but Executorch could not be loaded")
                print(executorch_exception)
        if output_dso_path:
            output_dso_path = str(os.path.abspath(output_dso_path))
            print(f"Exporting model using AOT Inductor to {output_dso_path}")
            export_model_aoti(model, builder_args.device, output_dso_path, args)


def cli():
    args = cli_args()
    main(args)

if __name__ == "__main__":
    cli()

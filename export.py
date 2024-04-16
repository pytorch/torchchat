# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch

from build.builder import _initialize_model, BuilderArgs
from cli import add_arguments_for_export, arg_init, check_args
from export_aoti import export_model as export_model_aoti

from quantize import set_precision

try:
    executorch_export_available = True
    from export_et import export_model as export_model_et
except Exception as e:
    executorch_exception = f"ET EXPORT EXCEPTION: {e}"
    executorch_export_available = False


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
                print(
                    "Export with executorch requested but Executorch could not be loaded"
                )
                print(executorch_exception)
        if output_dso_path:
            output_dso_path = str(os.path.abspath(output_dso_path))
            print(f"Exporting model using AOT Inductor to {output_dso_path}")
            export_model_aoti(model, builder_args.device, output_dso_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="export",
                                    description="Export specific CLI",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments_for_export(parser)
    args = parser.parse_args()
    check_args(args, "export")
    args = arg_init(args)
    main(args)

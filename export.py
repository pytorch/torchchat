# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import torch

from build.builder import (
    _initialize_model,
    _set_gguf_kwargs,
    _unset_gguf_kwargs,
    BuilderArgs,
)
from cli import add_arguments_for_export, arg_init, check_args
from export_aoti import export_model as export_model_aoti

from quantize import set_precision

logger = logging.getLogger(__name__)

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
        logging.error(f"device={device} is not yet suppported")


def main(args):
    builder_args = BuilderArgs.from_args(args)
    quantize = args.quantize

    logging.info(f"Using device={builder_args.device}")
    set_precision(builder_args.precision)

    builder_args.dso_path = None
    builder_args.pte_path = None
    builder_args.setup_caches = True

    output_pte_path = args.output_pte_path
    output_dso_path = args.output_dso_path

    # TODO: clean this up
    # This mess is because ET does not support _weight_int4pack_mm right now
    if not builder_args.gguf_path:
        model = _initialize_model(
            builder_args,
            quantize,
        )
        model_to_pte = model
        model_to_dso = model
    else:
        if output_pte_path:
            _set_gguf_kwargs(builder_args, is_et=True, context="export")
            model_to_pte = _initialize_model(
                builder_args,
                quantize,
            )
            _unset_gguf_kwargs(builder_args)

        if output_dso_path:
            _set_gguf_kwargs(builder_args, is_et=False, context="export")
            model_to_dso = _initialize_model(
                builder_args,
                quantize,
            )
            _unset_gguf_kwargs(builder_args)

    with torch.no_grad():
        if output_pte_path:
            output_pte_path = str(os.path.abspath(output_pte_path))
            logging.debug(f">{output_pte_path}<")
            if executorch_export_available:
                logging.info(f"Exporting model using Executorch to {output_pte_path}")
                export_model_et(
                    model_to_pte, builder_args.device, args.output_pte_path, args
                )
            else:
                logging.error(
                    "Export with executorch requested but Executorch could not be loaded"
                )
                logging.error(executorch_exception)
        if output_dso_path:
            output_dso_path = str(os.path.abspath(output_dso_path))
            logging.info(f"Exporting model using AOT Inductor to {output_dso_path}")
            export_model_aoti(model_to_dso, builder_args.device, output_dso_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export specific CLI.")
    add_arguments_for_export(parser)
    args = parser.parse_args()
    check_args(args, "export")
    args = arg_init(args)
    main(args)

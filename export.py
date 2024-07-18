# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch

from build.builder import (
    _initialize_model,
    _initialize_tokenizer,
    _set_gguf_kwargs,
    _unset_gguf_kwargs,
    BuilderArgs,
    TokenizerArgs,
)

from build.utils import set_backend, set_precision
from cli import add_arguments_for_verb, arg_init, check_args
from export_aoti import export_model as export_model_aoti

try:
    executorch_export_available = True
    from export_et import export_model as export_model_et
except Exception as e:
    executorch_exception = f"ET EXPORT EXCEPTION: {e}"
    executorch_export_available = False


default_device = "cpu"


def main(args):
    builder_args = BuilderArgs.from_args(args)
    quantize = args.quantize

    print(f"Using device={builder_args.device}")
    set_precision(builder_args.precision)
    set_backend(dso=args.output_dso_path, pte=args.output_pte_path, aoti_package=args.output_aoti_package_path)

    builder_args.dso_path = None
    builder_args.pte_path = None
    builder_args.aoti_package_path = None
    builder_args.setup_caches = True

    output_pte_path = args.output_pte_path
    output_dso_path = args.output_dso_path
    output_aoti_package_path = args.output_aoti_package_path

    if output_pte_path and builder_args.device != "cpu":
        print(
            f"Warning! ExecuTorch export target is controlled by export recipe, not device setting. Ignoring device={builder_args.device} setting."
        )
        builder_args.device = "cpu"
    elif "mps" in builder_args.device:
        print("Warning! Device MPS not supported for export. Exporting for device CPU.")
        builder_args.device = "cpu"

    # TODO: clean this up
    # This mess is because ET does not support _weight_int4pack_mm right now
    if not builder_args.gguf_path:
        # tokenizer needed for quantization so get that here,
        try:
            tokenizer_args = TokenizerArgs.from_args(args)
            tokenizer = _initialize_tokenizer(tokenizer_args)
        except:
            tokenizer = None

        model = _initialize_model(
            builder_args,
            quantize,
            tokenizer,
        )
        model_to_pte = model
        model_to_dso = model
        model_to_aoti_package = model
    else:
        if output_pte_path:
            _set_gguf_kwargs(builder_args, is_et=True, context="export")
            model_to_pte = _initialize_model(
                builder_args,
                quantize,
            )
            _unset_gguf_kwargs(builder_args)

        if output_dso_path or output_aoti_package_path:
            _set_gguf_kwargs(builder_args, is_et=False, context="export")
            model_to_aoti_package = _initialize_model(
                builder_args,
                quantize,
            )
            model_to_dso = model_to_aoti_package
            _unset_gguf_kwargs(builder_args)

    with torch.no_grad():
        if output_pte_path:
            output_pte_path = str(os.path.abspath(output_pte_path))
            if executorch_export_available:
                print(f"Exporting model using ExecuTorch to {output_pte_path}")
                export_model_et(
                    model_to_pte, builder_args.device, args.output_pte_path, args
                )
            else:
                print(
                    "Export with executorch requested but ExecuTorch could not be loaded"
                )
                print(executorch_exception)
                
        if output_dso_path:
            output_dso_path = str(os.path.abspath(output_dso_path))
            print(f"Exporting model using AOT Inductor to {output_dso_path}")
            export_model_aoti(model_to_dso, builder_args.device, output_dso_path, args, False)

        if output_aoti_package_path:
            output_aoti_package_path = str(os.path.abspath(output_aoti_package_path))
            print(f"Exporting model using AOT Inductor to {output_aoti_package_path}")
            export_model_aoti(model_to_aoti_package, builder_args.device, output_aoti_package_path, args, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torchchat export CLI")
    add_arguments_for_verb(parser, "export")
    args = parser.parse_args()
    check_args(args, "export")
    args = arg_init(args)
    main(args)

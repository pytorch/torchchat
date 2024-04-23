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

from build.utils import set_backend, set_precision, use_aoti_backend, use_et_backend
from cli import add_arguments, add_arguments_for_export, arg_init, check_args
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
    set_backend(dso=args.output_dso_path, pte=args.output_pte_path)

    builder_args.dso_path = None
    builder_args.pte_path = None
    builder_args.setup_caches = True

    output_pte_path = args.output_pte_path
    output_dso_path = args.output_dso_path

    # tokenizer needed for quantization so get that here,
    try:
        tokenizer_args = TokenizerArgs.from_args(args)
    except:
        tokenizer_args = None

    # This is heroic, but still confusing to the user
    # because they think they are in fect exporting the
    # same model when they's not. In set_backend we just
    # throw an exception if the user asks for both, but
    # we need to know which one.
    model_to_pte = None
    model_to_dso = None

    # TODO: clean this up
    # This mess is because ET does not support _weight_int4pack_mm right now
    if not builder_args.gguf_path:
        model, tokenizer = _initialize_model(
            builder_args,
            tokenizer_args,
            quantize,
        )

        model_to_pte = model
        model_to_dso = model
    else:
        # for now, we simply don't suport GPTQ for GGUF
        if output_pte_path:
            print(
                "Warning: may not be able to represent quantized models, dequantizing when necessary"
            )
            _set_gguf_kwargs(builder_args, is_et=True, context="export")
            model_to_pte, tokenizer = _initialize_model(
                builder_args,
                tokenizer_args,
                quantize,
            )
            _unset_gguf_kwargs(builder_args)

        if output_dso_path:
            _set_gguf_kwargs(builder_args, is_et=False, context="export")
            model_to_dso, tokenizer = _initialize_model(
                builder_args,
                tokenizer_args,
                quantize,
            )
            _unset_gguf_kwargs(builder_args)

    with torch.no_grad():
        if output_pte_path:
            output_pte_path = str(os.path.abspath(output_pte_path))
            # print(f">{output_pte_path}<")
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
            export_model_aoti(model_to_dso, builder_args.device, output_dso_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torchchat export CLI")
    add_arguments(parser)
    add_arguments_for_export(parser)
    args = parser.parse_args()
    check_args(args, "export")
    args = arg_init(args)
    main(args)

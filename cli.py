# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch


default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'

strict = False


def check_args(args, command_name: str):
    global strict

    # chat and generate support the same options
    if command_name in ["generate", "chat", "gui"]:
        # examples, can add more. Note that attributes convert dash to _
        disallowed_args = ["output_pte_path", "output_dso_path"]
    elif command_name == "export":
        # examples, can add more. Note that attributes convert dash to _
        disallowed_args = ["pte_path", "dso_path"]
    elif command_name == "eval":
        # TBD
        disallowed_args = []
    else:
        raise RuntimeError(f"{command_name} is not a valid command")

    for disallowed in disallowed_args:
        if hasattr(args, disallowed):
            text = f"command {command_name} does not support option {disallowed.replace('_', '-')}"
            if strict:
                raise RuntimeError(text)
            else:
                print(f"Warning: {text}")


def add_arguments_for_generate(parser):
    # Only generate specific options should be here
    _add_arguments_common(parser)


def add_arguments_for_eval(parser):
    # Only eval specific options should be here
    _add_arguments_common(parser)


def add_arguments_for_export(parser):
    # Only export specific options should be here
    _add_arguments_common(parser)


def _add_arguments_common(parser):
    # TODO: Refactor this so that only common options are here
    # and subcommand-specific options are inside individual
    # add_arguments_for_generate, add_arguments_for_export etc.
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,  # set None for release
        help="Initialize torch seed",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument(
        "--tiktoken",
        action="store_true",
        help="Whether to use tiktoken tokenizer.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use torchchat for an interactive chat session.",
    )
    parser.add_argument(
        "--is-chat-model",
        action="store_true",
        help="Indicate that the model was trained to support chat functionality.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use torchchat to for an interactive gui-chat session.",
    )
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top-k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile-prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--speculate-k", type=int, default=5, help="Speculative execution depth."
    )
    parser.add_argument(
        "--draft-checkpoint-path",
        type=Path,
        default=None,
        help="Draft checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="not_specified",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Model checkpoint directory.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=None,
        help="Parameter file path.",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        default=None,
        help="GGUF file path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Model checkpoint path.",
    )
    parser.add_argument("--output-pte-path", type=str, default=None, help="Filename")
    parser.add_argument("--output-dso-path", type=str, default=None, help="Filename")
    parser.add_argument(
        "--dso-path", type=Path, default=None, help="Use the specified AOTI DSO model."
    )
    parser.add_argument(
        "--pte-path",
        type=Path,
        default=None,
        help="Use the specified Executorch PTE model.",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="float32",
        help="Override the dtype of the model (default is the checkpoint dtype). Options: bf16, fp16, fp32",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        default="info",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level to display",
    )
    parser.add_argument(
        "--quantize", type=str, default="{ }", help="Quantization options."
    )
    parser.add_argument("--params-table", type=str, default=None, help="Device to use")
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["hellaswag"],
        help="list of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="number of samples to evaluate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="maximum length sequence to evaluate",
    )


def arg_init(args):

    if Path(args.quantize).is_file():
        with open(args.quantize, "r") as f:
            args.quantize = json.loads(f.read())

    if args.seed:
        torch.manual_seed(args.seed)
    return args

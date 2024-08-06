# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

from build.utils import allowable_dtype_names, allowable_params_table, get_device_str
from download import download_and_convert, is_model_downloaded

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

default_device = os.getenv("TORCHCHAT_DEVICE", "fast")
default_model_dir = Path(
    os.getenv("TORCHCHAT_MODELDIR", "~/.torchchat/model-cache")
).expanduser()


# Subcommands related to downloading and managing model artifacts
INVENTORY_VERBS = ["download", "list", "remove", "where"]

# Subcommands related to generating inference output based on user prompts
GENERATION_VERBS = ["browser", "chat", "generate", "server"]

# List of all supported subcommands in torchchat
KNOWN_VERBS = GENERATION_VERBS + ["eval", "export"] + INVENTORY_VERBS


# Handle CLI arguments that are common to a majority of subcommands.
def check_args(args, verb: str) -> None:
    # Handle model download. Skip this for download, since it has slightly
    # different semantics.
    if (
        verb not in INVENTORY_VERBS
        and args.model
        and not is_model_downloaded(args.model, args.model_directory)
    ):
        download_and_convert(args.model, args.model_directory, args.hf_token)


# Given a arg parser and a subcommand (verb), add the appropriate arguments
# for that subcommand.
def add_arguments_for_verb(parser, verb: str) -> None:
    # Argument closure for inventory related subcommands
    if verb in INVENTORY_VERBS:
        _configure_artifact_inventory_args(parser, verb)
        _add_cli_metadata_args(parser)
        return

    # Add argument groups for model specification (what base model to use)
    _add_model_specification_args(parser)

    # Add argument groups for model configuration (compilation, quant, etc)
    _add_model_config_args(parser, verb)

    # Add thematic argument groups based on the subcommand
    if verb in GENERATION_VERBS:
        _add_exported_input_path_args(parser)
        _add_generation_args(parser, verb)
    if verb == "export":
        _add_export_output_path_args(parser)
    if verb == "eval":
        _add_exported_input_path_args(parser)
        _add_evaluation_args(parser)

    # Add CLI Args related to downloading of model artifacts (if not already downloaded)
    _add_jit_downloading_args(parser)

    # Add CLI Args that are general to subcommand cli execution
    _add_cli_metadata_args(parser)

    # WIP Features (suppressed from --help)
    _add_distributed_args(parser)
    _add_custom_model_args(parser)
    _add_speculative_execution_args(parser)


# Add CLI Args related to model specification (what base model to use)
def _add_model_specification_args(parser) -> None:
    model_specification_parser = parser.add_argument_group(
        "Model Specification",
        "(REQUIRED) Specify the base model. Args are mutually exclusive.",
    )
    exclusive_parser = model_specification_parser.add_mutually_exclusive_group(
        required=True
    )
    exclusive_parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model name for well-known models",
    )
    exclusive_parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="not_specified",
        help="Use the specified model checkpoint path",
    )
    # See _add_custom_model_args() for more details
    exclusive_parser.add_argument(
        "--gguf-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
        # "Use the specified GGUF model file",
    )

    model_specification_parser.add_argument(
        "--is-chat-model",
        action="store_true",
        # help="Indicate that the model was trained to support chat functionality",
        help=argparse.SUPPRESS,
    )


# Add CLI Args related to model configuration (compilation, quant, etc)
# Excludes compile args if subcommand is export
def _add_model_config_args(parser, verb: str) -> None:
    model_config_parser = parser.add_argument_group(
        "Model Configuration", "Specify model configurations"
    )

    if verb != "export":
        model_config_parser.add_argument(
            "--compile",
            action="store_true",
            help="Whether to compile the model with torch.compile",
        )
        model_config_parser.add_argument(
            "--compile-prefill",
            action="store_true",
            help="Whether to compile the prefill. Improves prefill perf, but has higher compile times.",
        )

    model_config_parser.add_argument(
        "--dtype",
        default="fast",
        choices=allowable_dtype_names(),
        help="Override the dtype of the model (default is the checkpoint dtype). Options: bf16, fp16, fp32, fast16, fast",
    )
    model_config_parser.add_argument(
        "--quantize",
        type=str,
        default="{ }",
        help=(
            'Quantization options. pass in as \'{"<mode>" : {"<argname1>" : <argval1>, "<argname2>" : <argval2>,...},}\' '
            + "modes are: embedding, linear:int8, linear:int4, linear:a8w4dq, precision."
        ),
    )
    model_config_parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["fast", "cpu", "cuda", "mps"],
        help="Hardware device to use. Options: cpu, cuda, mps",
    )


# Add CLI Args representing output paths of exported model files
def _add_export_output_path_args(parser) -> None:
    output_path_parser = parser.add_argument_group(
        "Export Output Path",
        "Specify the output path for the exported model files",
    )
    exclusive_parser = output_path_parser.add_mutually_exclusive_group()
    exclusive_parser.add_argument(
        "--output-pte-path",
        type=str,
        default=None,
        help="Output to the specified ExecuTorch .pte model file",
    )
    exclusive_parser.add_argument(
        "--output-dso-path",
        type=str,
        default=None,
        help="Output to the specified AOT Inductor .dso model file",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Call torch.export with dynamic shapes",
    )


# Add CLI Args representing user provided exported model files
def _add_exported_input_path_args(parser) -> None:
    exported_model_path_parser = parser.add_argument_group(
        "Exported Model Path",
        "Specify the path of the exported model files to ingest",
    )
    exclusive_parser = exported_model_path_parser.add_mutually_exclusive_group()
    exclusive_parser.add_argument(
        "--dso-path",
        type=Path,
        default=None,
        help="Use the specified AOT Inductor .dso model file",
    )
    exclusive_parser.add_argument(
        "--pte-path",
        type=Path,
        default=None,
        help="Use the specified ExecuTorch .pte model file",
    )


# Add CLI Args related to JIT downloading of model artifacts
def _add_jit_downloading_args(parser) -> None:
    jit_downloading_parser = parser.add_argument_group(
        "Model Downloading",
        "Specify args for model downloading (if model is not downloaded)",
    )
    jit_downloading_parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="A HuggingFace API token to use when downloading model artifacts",
    )
    jit_downloading_parser.add_argument(
        "--model-directory",
        type=Path,
        default=default_model_dir,
        help=f"The directory to store downloaded model artifacts. Default: {default_model_dir}",
    )


# Add CLI Args that are general to subcommand cli execution
def _add_cli_metadata_args(parser) -> None:
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        # help="Profile path.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Initialize torch seed",
    )


# Configure CLI Args specific to Model Artifact Management
def _configure_artifact_inventory_args(parser, verb: str) -> None:
    if verb in ["download", "remove", "where"]:
        parser.add_argument(
            "model",
            type=str,
            nargs="?",
            default=None,
            help="Model name for well-known models",
        )

    if verb in INVENTORY_VERBS:
        parser.add_argument(
            "--model-directory",
            type=Path,
            default=default_model_dir,
            help=f"The directory to store downloaded model artifacts. Default: {default_model_dir}",
        )

    if verb == "download":
        parser.add_argument(
            "--hf-token",
            type=str,
            default=None,
            help="A HuggingFace API token to use when downloading model artifacts",
        )


# Add CLI Args specific to user prompted generation
# Include prompt and num_sample args when the subcommand is generate
def _add_generation_args(parser, verb: str) -> None:
    generator_parser = parser.add_argument_group(
        "Generation", "Configs for generating output based on provided prompt"
    )

    if verb == "generate":
        generator_parser.add_argument(
            "--prompt",
            type=str,
            default="Hello, my name is",
            help="Input prompt for manual output generation",
        )
        generator_parser.add_argument(
            "--num-samples",
            type=int,
            default=1,
            help="Number of samples",
        )

    generator_parser.add_argument(
        "--chat",
        action="store_true",
        # help="Whether to start an interactive chat session",
        help=argparse.SUPPRESS,
    )
    generator_parser.add_argument(
        "--gui",
        action="store_true",
        # help="Whether to use a web UI for an interactive chat session",
        help=argparse.SUPPRESS,
    )
    generator_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens",
    )
    generator_parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Top-k for sampling",
    )
    generator_parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling"
    )
    generator_parser.add_argument(
        "--sequential-prefill",
        action="store_true",
        # help="Whether to perform prefill sequentially. Only used for model debug.",
        help=argparse.SUPPRESS,
    )


# Add CLI Args specific to Model Evaluation
def _add_evaluation_args(parser) -> None:
    eval_parser = parser.add_argument_group(
        "Evaluation", "Configs for evaluating model performance"
    )
    eval_parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="List of lm-eluther tasks to evaluate. Usage: --tasks task1 task2",
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to evaluate",
    )
    eval_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum length sequence to evaluate",
    )


# Add CLI Args related to distributed inference
# This feature is currently a [WIP] and hidden from --help
def _add_distributed_args(parser) -> None:
    parser.add_argument(
        "--distributed",
        action="store_true",
        help=argparse.SUPPRESS,
        # "Whether to enable distributed inference",
    )
    parser.add_argument(
        "--dcp-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
        # "Use the specified model checkpoint directory",
    )


# Add CLI Args related to custom model inputs (e.g. GGUF)
# This feature is currently a [WIP] and hidden from --help
def _add_custom_model_args(parser) -> None:
    parser.add_argument(
        "--params-table",
        type=str,
        default=None,
        choices=allowable_params_table(),
        help=argparse.SUPPRESS,
        # "Parameter table to use",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
        # "Use the specified parameter file",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
        # "Use the specified model tokenizer file",
    )


# Add CLI Args related to speculative execution
# This feature is currently a [WIP] and hidden from --help
def _add_speculative_execution_args(parser) -> None:
    parser.add_argument(
        "--speculate-k",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
        # "Speculative execution depth",
    )
    parser.add_argument(
        "--draft-checkpoint-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
        # "Use the specified draft checkpoint path",
    )
    parser.add_argument(
        "--draft-quantize",
        type=str,
        default="{ }",
        help=argparse.SUPPRESS,
        # (
        #     "Quantization options. Same format as quantize, "
        #     + "or 'quantize' to indicate same options specified by "
        #     + "--quantize to main model. Applied to draft model."
        # ),
    )


def arg_init(args):
    if not (torch.__version__ > "2.3"):
        raise RuntimeError(
            f"You are using PyTorch {torch.__version__}. At this time, torchchat uses the latest PyTorch technology with high-performance kernels only available in PyTorch nightly until the PyTorch 2.4 release"
        )

    if sys.version_info.major != 3 or sys.version_info.minor < 10:
        raise RuntimeError("Please use Python 3.10 or later.")

    if hasattr(args, "quantize") and Path(args.quantize).is_file():
        with open(args.quantize, "r") as f:
            args.quantize = json.loads(f.read())

    if isinstance(args.quantize, str):
        args.quantize = json.loads(args.quantize)

    # if we specify dtype in quantization recipe, replicate it as args.dtype
    args.dtype = args.quantize.get("precision", {}).get("dtype", args.dtype)

    if getattr(args, "output_pte_path", None):
        if args.device not in ["cpu", "fast"]:
            raise RuntimeError("Device not supported by ExecuTorch")
        args.device = "cpu"
    else:
        args.device = get_device_str(
            args.quantize.get("executor", {}).get("accelerator", args.device)
        )

    if "mps" in args.device:
        if hasattr(args, "compile") and hasattr(args, "compile_prefill"):
            print(
                "Warning: compilation is not available with device MPS, ignoring option to engage compilation"
            )
            vars(args)["compile"] = False
            vars(args)["compile_prefill"] = False

    if hasattr(args, "seed") and args.seed:
        torch.manual_seed(args.seed)
    return args

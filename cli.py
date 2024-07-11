# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

# List of all supported subcommands in torchchat
KNOWN_VERBS = ["chat", "browser", "generate", "eval", "export"] + INVENTORY_VERBS


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


def add_arguments_for_verb(parser, verb: str) -> None:
    # Model specification. TODO Simplify this.
    # A model can be specified using a positional model name or HuggingFace
    # path. Alternatively, the model can be specified via --gguf-path or via
    # an explicit --checkpoint-dir, --checkpoint-path, or --tokenizer-path.

    if verb in INVENTORY_VERBS:
        _configure_artifact_inventory_args(parser, verb)
        _add_cli_metadata_args(parser)
        return

    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model name for well-known models",
    )

    if verb in ["browser", "chat", "generate"]:
        _add_generation_args(parser) 

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether to enable distributed inference",
    )
    parser.add_argument(
        "--is-chat-model",
        action="store_true",
        help="Indicate that the model was trained to support chat functionality",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Initialize torch seed",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model with torch.compile",
    )
    parser.add_argument(
        "--compile-prefill",
        action="store_true",
        help="Whether to compile the prefill. Improves prefill perf, but has higher compile times.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Profile path.",
    )
    parser.add_argument(
        "--draft-checkpoint-path",
        type=Path,
        default=None,
        help="Use the specified draft checkpoint path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="not_specified",
        help="Use the specified model checkpoint path",
    )
    parser.add_argument(
        "--dcp-dir",
        type=Path,
        default=None,
        help="Use the specified model checkpoint directory",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=None,
        help="Use the specified parameter file",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        default=None,
        help="Use the specified GGUF model file",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Use the specified model tokenizer file",
    )

    _add_exported_model_input_args(parser)
    _add_export_output_path_args(parser)

    parser.add_argument(
        "--dtype",
        default="fast",
        choices=allowable_dtype_names(),
        help="Override the dtype of the model (default is the checkpoint dtype). Options: bf16, fp16, fp32, fast16, fast",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="{ }",
        help=(
            'Quantization options. pass in as \'{"<mode>" : {"<argname1>" : <argval1>, "<argname2>" : <argval2>,...},}\' '
            + "modes are: embedding, linear:int8, linear:int4, linear:a8w4dq, precision."
        ),
    )
    parser.add_argument(
        "--draft-quantize",
        type=str,
        default="{ }",
        help=(
            "Quantization options. Same format as quantize, "
            + "or 'quantize' to indicate same options specified by "
            + "--quantize to main model. Applied to draft model."
        ),
    )
    parser.add_argument(
        "--params-table",
        type=str,
        default=None,
        choices=allowable_params_table(),
        help="Parameter table to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["fast", "cpu", "cuda", "mps"],
        help="Hardware device to use. Options: cpu, cuda, mps",
    )

    if verb == "eval":
        _add_evaluation_args(parser)

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="A HuggingFace API token to use when downloading model artifacts",
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=default_model_dir,
        help=f"The directory to store downloaded model artifacts. Default: {default_model_dir}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web server in browser mode",
    )
    _add_cli_metadata_args(parser)


# Add CLI Args representing user provided exported model files
def _add_export_output_path_args(parser) -> None:
    output_path_parser = parser.add_argument_group("Export Output Path Args", "Specify the output path for the exported model files")
    output_path_parser.add_argument(
        "--output-pte-path",
        type=str,
        default=None,
        help="Output to the specified ExecuTorch .pte model file",
    )
    output_path_parser.add_argument(
        "--output-dso-path",
        type=str,
        default=None,
        help="Output to the specified AOT Inductor .dso model file",
    )


# Add CLI Args representing user provided exported model files
def _add_exported_model_input_args(parser) -> None:
    exported_model_path_parser = parser.add_argument_group("Exported Model Path Args", "Specify the path of the exported model files to ingest")
    exported_model_path_parser.add_argument(
        "--dso-path",
        type=Path,
        default=None,
        help="Use the specified AOT Inductor .dso model file",
    )
    exported_model_path_parser.add_argument(
        "--pte-path",
        type=Path,
        default=None,
        help="Use the specified ExecuTorch .pte model file",
    )

    
# Add CLI Args that are relevant to any subcommand execution
def _add_cli_metadata_args(parser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
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
def _add_generation_args(parser) -> None:
    generator_parser = parser.add_argument_group("Generation Args", "Configs for generating output based on provided prompt")
    generator_parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Input prompt for manual output generation",
    )
    generator_parser.add_argument(
        "--chat",
        action="store_true",
        help="Whether to start an interactive chat session",
    )
    generator_parser.add_argument(
        "--gui",
        action="store_true",
        help="Whether to use a web UI for an interactive chat session",
    )
    generator_parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples",
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
        help="Whether to perform prefill sequentially. Only used for model debug.",
    )
    generator_parser.add_argument(
        "--speculate-k",
        type=int,
        default=5,
        help="Speculative execution depth",
    )


# Add CLI Args specific to Model Evaluation
def _add_evaluation_args(parser) -> None:
    eval_parser = parser.add_argument_group("Evaluation Args", "Configs for evaluating model performance")
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

    if args.output_pte_path:
        if args.device not in ["cpu", "fast"]:
            raise RuntimeError("Device not supported by ExecuTorch")
        args.device = "cpu"
    else:
        args.device = get_device_str(
            args.quantize.get("executor", {}).get("accelerator", args.device)
        )

    if "mps" in args.device:
        if args.compile or args.compile_prefill:
            print(
                "Warning: compilation is not available with device MPS, ignoring option to engage compilation"
            )
            args.compile = False
            args.compile_prefill = False

    if hasattr(args, "seed") and args.seed:
        torch.manual_seed(args.seed)
    return args

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

FORMAT = (
    "%(levelname)s: %(asctime)-15s: %(filename)s: %(funcName)s: %(module)s: %(message)s"
)
logging.basicConfig(filename="/tmp/torchchat.log", level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

default_device = os.getenv("TORCHCHAT_DEVICE", "fast")
default_model_dir = Path(
    os.getenv("TORCHCHAT_MODELDIR", "~/.torchchat/model-cache")
).expanduser()


KNOWN_VERBS = ["chat", "browser", "download", "generate", "eval", "export", "list", "remove", "where"]

# Handle CLI arguments that are common to a majority of subcommands.
def check_args(args, verb: str) -> None:
    # Handle model download. Skip this for download, since it has slightly
    # different semantics.
    if (
        verb not in ["download", "list", "remove"]
        and args.model
        and not is_model_downloaded(args.model, args.model_directory)
    ):
        download_and_convert(args.model, args.model_directory, args.hf_token)


def add_arguments_for_verb(parser, verb: str):
    # Model specification. TODO Simplify this.
    # A model can be specified using a positional model name or HuggingFace
    # path. Alternatively, the model can be specified via --gguf-path or via
    # an explicit --checkpoint-dir, --checkpoint-path, or --tokenizer-path.
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model name for well-known models",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Whether to start an interactive chat session",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Whether to use a web UI for an interactive chat session",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Input prompt",
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
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Top-k for sampling",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling"
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
        "--sequential-prefill",
        action="store_true",
        help="Whether to perform prefill sequentially. Only used for model debug.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Profile path.",
    )
    parser.add_argument(
        "--speculate-k",
        type=int,
        default=5,
        help="Speculative execution depth",
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
    parser.add_argument(
        "--output-pte-path",
        type=str,
        default=None,
        help="Output to the specified ExecuTorch .pte model file",
    )
    parser.add_argument(
        "--output-dso-path",
        type=str,
        default=None,
        help="Output to the specified AOT Inductor .dso model file",
    )
    parser.add_argument(
        "--dso-path",
        type=Path,
        default=None,
        help="Use the specified AOT Inductor .dso model file",
    )
    parser.add_argument(
        "--pte-path",
        type=Path,
        default=None,
        help="Use the specified ExecuTorch .pte model file",
    )
    parser.add_argument(
        "--dtype",
        default="fast",
        choices=allowable_dtype_names(),
        help="Override the dtype of the model (default is the checkpoint dtype). Options: bf16, fp16, fp32, fast16, fast",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="{ }",
        help=(
            'Quantization options. pass in as \'{"<mode>" : {"<argname1>" : <argval1>, "<argname2>" : <argval2>,...},}\' '
            + "modes are: embedding, linear:int8, linear:int4, linear:gptq, linear:hqq, linear:a8w4dq, precision."
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
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="List of lm-eluther tasks to evaluate. Usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum length sequence to evaluate",
    )
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

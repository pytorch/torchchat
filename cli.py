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

logging.basicConfig(level=logging.INFO,format="%(message)s")
logger = logging.getLogger(__name__)

default_device = os.getenv("TORCHCHAT_DEVICE", "fast")
default_model_dir = Path(
    os.getenv("TORCHCHAT_MODELDIR", "~/.torchchat/model-cache")
).expanduser()


KNOWN_VERBS = ["chat", "browser", "download", "generate", "eval", "export", "list", "remove", "where"]

# Handle CLI arguments that are common to a majority of subcommands.
# - Download the specified model, if not already downloaded
#     (Except for download related commands, which have different semantics)
def check_args(args, verb: str) -> None:
    if (
        verb not in ["download", "list", "remove"]
        and args.model
        and not is_model_downloaded(args.model, args.model_directory)
    ):
        download_and_convert(args.model, args.model_directory, args.hf_token)


# BuilderArgs: checkpoint_dir, checkpoint_path, params_table, model, model_directory, is_chat_model, dso_path, pte_path, gguf_path, output_pte, dtype, device, params_path, output_dso_path, distributed
    # Exclusive:                                params_table,                         is_chat_model,                     gguf_path, output_pte, dtype, device, params_path, output_dso_path, distributed
# TokenizerArgs: tokenizer_path, model, model_directory, checkpoint_path, checkpoint_dir
    # Exclusive: tokenizer_path
# SpeculativeArgs: BuilderArgs, draft_checkpoint_path
    # Exclusive: draft_checkpoint_path
# GeneratorArgs: sequential_prefill, dso_path, pte_path, prompt, chat, gui, num_samples, max_new_tokens, top_k, temperature, compile, compile_prefill, speculate_k
    # Exclusive: sequential_prefill,                     prompt, chat, gui, num_samples, max_new_tokens, top_k, temperature, compile, compile_prefill, speculate_k


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
        help="Model alias for pre-defined, well-known models",
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=default_model_dir,
        help=f"The directory to read/store downloaded model artifacts. Default: {default_model_dir}",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="A HuggingFace API token to use when downloading model artifacts. Only used if model is not downloaded",
    )

    parser.add_argument(
        "--quantize",
        type=str,
        default="{ }",
        help=(
            'Quantization options. pass in as \'{"<mode>" : {"<argname1>" : <argval1>, "<argname2>" : <argval2>,...},}\' '
            + "modes are: embedding, linear:int8, linear:int4, linear:a8w4dq, precision, executor."
        ),
    )
    
    # Hack to be cleaned up: Used for arg validation, but never functionally utilized
    parser.add_argument(
        "--is-chat-model",
        action="store_true",
        help="Indicate that the model was trained to support chat functionality",
    )
    

    # Model Building Arguments
    if True: 
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
            "--dtype",
            default="fast",
            choices=allowable_dtype_names(),
            help="Override the dtype of the model (default is the checkpoint dtype). Options: bf16, fp16, fp32, fast16, fast",
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
            "--distributed",
            action="store_true",
            help="Whether to enable distributed inference",
        )

        # Output: Exported Model Path Args
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


    # Generation Related Arguments (Non Perplexity Eval)
    if True:
        parser.add_argument(
            "--prompt",
            type=str,
            default="Hello, my name is",
            help="Input prompt for model generation",
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

        # Used for Speculative Generation
        parser.add_argument(
            "--draft-checkpoint-path",
            type=Path,
            default=None,
            help="Use the specified draft checkpoint path",
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
        
        # Used only for debugging
        parser.add_argument(
            "--sequential-prefill",
            action="store_true",
            help="Whether to perform prefill sequentially. Only used for model debug.",
        )

        # Hack to be cleaned up: Curried arg that is not manually specified
        # Whether to start an interactive chat session
        parser.add_argument(
            "--chat",
            action="store_true",
            help=argparse.SUPPRESS,
        )
        # Hack to be cleaned up: Curried arg that is not manually specified
        # Whether to use a web UI for an interactive chat session
        parser.add_argument(
            "--gui",
            action="store_true",
            help=argparse.SUPPRESS,
        )

    
    # Input: Exported Model Path Args
    if True:
        # Should be exclusive with pte-path
        parser.add_argument(
            "--dso-path",
            type=Path,
            default=None,
            help="Use the specified AOT Inductor .dso model file",
        )
        # Should be exclusive with dso-path
        parser.add_argument(
            "--pte-path",
            type=Path,
            default=None,
            help="Use the specified ExecuTorch .pte model file",
        )

    
    # Model Evaluation Related Args
    if True:
        # Only for Eval
        parser.add_argument(
            "--tasks",
            nargs="+",
            type=str,
            default=["wikitext"],
            help="List of lm-eluther tasks to evaluate. Usage: --tasks task1 task2",
        )
        # Only for Eval
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Number of samples to evaluate",
        )
        # Seems overloaded, used in eval 
        parser.add_argument(
            "--max-seq-length",
            type=int,
            default=None,
            help="Maximum length sequence to evaluate",
        )
    
    # Only for GUI
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web server in browser mode",
    )
    
    # General CLI Args
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Initialize/Freeze the torch seed",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output, when logging",
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

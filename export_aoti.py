# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim, export

from generate import _load_model, decode_one_token
from quantize import quantize_model

from model import Transformer

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def export_model(model: nn.Module, device, output_path):
    max_seq_length = 350
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    input = (
        torch.tensor([[1, 9038, 2501,  263,  931]], dtype=torch.int, device=device),
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int, device=device),
    )

    print(f"len(input)={len(input)}")

    seq = Dim("seq", min=1, max=max_seq_length)
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"idx": {1: seq}, "input_pos": {0: seq}}

    so = torch._export.aot_compile(
        model,
        args=input,
        options={"aot_inductor.output_path": output_path},
        dynamic_shapes=dynamic_shapes,
    )
    print(f"The generated DSO model can be found at: {so}")
    return so



def main(checkpoint_path, device, output_path, quantize = "{ }"):
    assert checkpoint_path.is_file(), checkpoint_path

    torch.manual_seed(1234)

    print(f"Using device={device}")
    precision = torch.float  # bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    quantize_model(model, quantize)
    
    with torch.no_grad():
        export_model(model, device, output_path)


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    ######################################################################
    ### We accept these options so we can ignore them w/o error
    
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--speculate_k", type=int, default=5, help="Speculative execution depth."
    )
    parser.add_argument(
        "--draft_checkpoint_path",
        type=Path,
        default=None,
        help="Draft checkpoint path.",
    )
    #####################################################################
    
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default=None,
        help="Override the dtype of the model (default is the checkpoint dtype). Options: fp16, fp32",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--quantize",
        type=str,
        default="{ }",
        help="Quantization options."
    )

    args = parser.parse_args()
    main(args.checkpoint_path, args.device, args.output_path, args.quantize)

if __name__ == "__main__":
    cli()

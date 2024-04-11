# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim, export

try:
    executorch_export_available = True
    from export_et import export_model as export_model_et
except Exception as e:
    print("ET EXPORT EXCEPTION: ", e) # TODO: remove
    executorch_export_available = False

from export_aoti import export_model as export_model_aoti

from model import Transformer
from generate import _load_model, decode_one_token
from quantize import quantize_model, name_to_dtype
from torch._export import capture_pre_autograd_graph

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


class model_wrapper(nn.Module):
    def __init__(self, model, device):
        super().__init__()

        max_seq_length = 350
        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        self.model = model
        # init model here if necessary

    def forward(self, idx, input_pos):
        # input_pos: [B, 1]
        # assert failed on symbolic shape during aot_compile?!
        # but not for ET?
        # assert input_pos.shape[-1] == 1
        logits = self.model(idx, input_pos)
        return logits  # sample(logits, **sampling_kwargs)


def main(checkpoint_path, device, quantize = "{ }", args = None):
    assert checkpoint_path.is_file(), checkpoint_path

    print(f"Using device={device}")
    precision = torch.float  # bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(
        checkpoint_path, device=device, precision=precision, use_tp=False)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    quantize_model(model, args.quantize)

    # dtype:
    if args.dtype:
        model.to(dtype=name_to_dtype(args.dtype))

    model = model_wrapper(model, device=device)

    output_pte_path = args.output_pte_path
    output_dso_path = args.output_dso_path

    with torch.no_grad():
        if output_pte_path:
            output_pte_path = str(os.path.abspath(output_pte_path))
            print(f">{output_pte_path}<")
            if executorch_export_available:
                print(f"Exporting model using Executorch to {output_pte_path}")
                export_model_et(model, device, args.output_pte_path, args)
            else:
                print(f"Export with executorch requested but Executorch could not be loaded")
        if output_dso_path:
            output_dso_path = str(os.path.abspath(output_dso_path))
            print(f"Exporting model using AOT Inductor to {output_dso_path}")
            export_model_aoti(model, device, output_dso_path, args)


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
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples.")
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
    parser.add_argument(
        "--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--speculate-k", type=int, default=5, help="Speculative execution depth."
    )
    parser.add_argument(
        "--draft-checkpoint-path",
        type=Path,
        default=None,
        help="Draft checkpoint path.",
    )
    #####################################################################

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="not_specified",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--output-pte-path",
        type=str,
        default=None,
        help="Filename"
    )
    parser.add_argument(
        "--output-dso-path",
        type=str,
        default=None,
        help="Filename"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default=None,
        help="Override the dtype of the model (default is the checkpoint dtype). Options: bf16, fp16, fp32",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--quantize",
        type=str,
        default="{ }",
        help="Quantization options."
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )


    args = parser.parse_args()
    main(args.checkpoint_path, args.device, args.quantize, args)

if __name__ == "__main__":
    cli()

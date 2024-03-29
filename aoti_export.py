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


class model_wrapper(nn.Module):
    def __init__(self, model, device):
        super().__init__()

        max_seq_length = 350
        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        self.model = model
        # init model here if necessary

    def forward(self, x, input_pos):
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        logits = self.model(x, input_pos)
        return logits  # sample(logits, **sampling_kwargs)


def export_model(model: nn.Module, device, output_path):

    export_model = model_wrapper(model, device=device)
    print(export_model)

    input = (
        torch.tensor([[1]], dtype=torch.long, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
    )

    print(f"len(input)={len(input)}")

    batch = Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"idx": {0: batch}, "input_pos": {0: batch}}

    so = torch._export.aot_compile(
        export_model,
        args=input,
        options={"aot_inductor.output_path": output_path},
    )
    print(f"The generated DSO model can be found at: {so}")
    return so

    #################################################################
    ### FIXME.... used for separate export & compile
    #################################################################
    
    exported_program: torch.export.ExportedProgram = export(
        export_model, args=input
    )  # , dynamic_shapes=dynamic_shapes)

    torch.export.save(exported_program, "exported_gpt-fast.pt2")

    print(exported_program)

    return

    so = torch._export.aot_compile(exported_program, args=input)
    print(f"{so=}")
    assert so is not None


def main(checkpoint_path, device, output_path, quantize = "{ }"):
    assert checkpoint_path.is_file(), checkpoint_path

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
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
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
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    parser.add_argument(
        "--out-path", type=str, default="stories15M.so", help="Filename"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="{ }",
        help="Quantization options."
    )

    args = parser.parse_args()
    main(args.checkpoint_path, args.device, args.out_path, args.quantize)

if __name__ == "__main__":
    cli()

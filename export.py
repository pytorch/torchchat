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

from quantize import quantize_model, name_to_dtype, set_precision, get_precision
from cli import cli_args

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


def main(args):
    checkpoint_path = args.checkpoint_path
    device = args.device
    quantize = args.quantize

    assert checkpoint_path.is_file(), checkpoint_path

    print(f"Using device={device}")
    precision = name_to_dtype(args.dtype)  # torch.float  # bfloat16
    set_precision(precision)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(
        checkpoint_path,
        args.checkpoint_dir,
        args.params_path,
        args.params_table,
        device=device,
        precision=precision,
        use_tp=False
    )

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    quantize_model(model, args)

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
    args = cli_args()
    main(args)

if __name__ == "__main__":
    cli()

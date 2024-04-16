# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from cli import check_args, cli_args
from eval import main as eval_main

from export import main as export_main
from generate import main as generate_main
from torch.export import Dim, export

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


def cli():
    args = cli_args()

    if args.generate or args.chat:
        check_args(args, "generate")
        generate_main(args)
    elif args.eval:
        eval_main(args)
    elif args.export:
        check_args(args, "export")
        export_main(args)
    else:
        raise RuntimeError("must specify either --generate or --export")


if __name__ == "__main__":
    cli()

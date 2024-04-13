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

from export import main as export_main
from generate import main as generate_main
from cli import cli_args

default_device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'

def cli():
    args = cli_args()
    
    if args.generate or args.chat:
        check_args(args, "generate")
        generate_main(args)
    elif args.export:
        check_args(args, "export")
        export_main(args)
    else:
        raise RuntimeError("must specify either --generate or --export")
    
if __name__ == "__main__":
    cli()

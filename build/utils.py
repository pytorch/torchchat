# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
from functools import reduce
from math import gcd
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchao.quantization.quant_api as quant_api


##########################################################################
###               dtype name to torch.dtype mapping                    ###

precision = torch.float32


def set_precision(dtype):
    global precision
    precision = dtype


def get_precision():
    global precision
    return precision


def name_to_dtype(name):
    if name in name_to_dtype_dict:
        return name_to_dtype_dict[name]
    else:
        raise RuntimeError(f"unsupported dtype name {name} specified")


name_to_dtype_dict = {
    "fp32": torch.float,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float": torch.float,
    "half": torch.float16,
    "float32": torch.float,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

#########################################################################
###             general model build utility functions                 ###


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def device_sync(device="cpu"):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        logging.error(f"device={ device } is not yet suppported")


#########################################################################
###                   general utilkity functions                      ###


# in fbcode, we can intercept certain local paths that
# should be interpreted as part of an XAR package
def canonical_path(path):
    return path

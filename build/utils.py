# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict

import torch


##########################################################################
###     set and get target backend is aoti or et for this model        ###

active_builder_args_dso = None
active_builder_args_pte = None


def set_backend(dso, pte):
    global active_builder_args_dso
    global active_builder_args_pte
    active_builder_args_dso = dso
    active_builder_args_pte = pte


def use_aoti_backend() -> bool:
    global active_builder_args_dso
    global active_builder_args_pte

    # eager == aoti, which is when backend has not been explicitly set
    if (not active_builder_args_dso) and not (active_builder_args_pte):
        return True

    if active_builder_args_pte and active_builder_args_dso:
        raise RuntimeError(
            "code generation needs to choose different implementations for DSO and PTE path.  Please only use one export option, and call export twice if necessary!"
        )

    return bool(active_builder_args_dso)


def use_et_backend() -> bool:
    global active_builder_args_dso
    global active_builder_args_pte

    # eager == aoti, which is when backend has not been explicitly set
    if not (active_builder_args_pte or active_builder_args_dso):
        return False

    if active_builder_args_pte and active_builder_args_dso:
        raise RuntimeError(
            "code generation needs to choose different implementations for DSO and PTE path.  Please only use one export option, and call export twice if necessary!"
        )

    return bool(active_builder_args_pte)


##########################################################################
###          set and get target precision for this model               ###

precision = torch.float32


def set_precision(dtype):
    global precision
    precision = dtype


def get_precision():
    global precision
    return precision


##########################################################################
###               dtype name to torch.dtype mapping                    ###
def name_to_dtype(name):
    if name in name_to_dtype_dict:
        return name_to_dtype_dict[name]
    else:
        raise RuntimeError(f"unsupported dtype name {name} specified")


def allowable_dtype_names() -> List[str]:
    return name_to_dtype_dict.keys()


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
###         general model build utility functions for CLI             ###


def allowable_params_table() -> List[str]:
    config_path = Path(f"{str(Path(__file__).parent)}/known_model_params")
    known_model_params = [
        config.replace(".json", "") for config in os.listdir(config_path)
    ]
    return known_model_params


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
###                    general utility functions                      ###


# in fbcode, we can intercept certain local paths that
# should be interpreted as part of an XAR package
def canonical_path(path):
    return path


#########################################################################
###                    general utility functions                      ###

def state_dict_device(d, device = "cpu") -> Dict:
    for key, weight in d.items():
        d[key] = weight.to(device=device)

    return d

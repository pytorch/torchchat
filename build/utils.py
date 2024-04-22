# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

import torch


##########################################################################
###     set and get target backend is aoti or et for this model        ###

active_builder_args = None


def set_backend(builder_args: "BuilderArgs"):
    global active_builder_args
    active_builder_args = builder_args


def use_aoti_backend() -> bool:
    global active_builder_args

    # eager == aoti, which is when backend has not been explicitly set
    if not active_builder_args:
        return True

    if active_builder_args.output_pte_path and active_builder_args.output_dso_path:
        raise RuntimeError(
            "code generation needs to choose different implementations for DSO and PTE path.  Please only use one export option, and call export twice if necessary!"
        )

    return bool(active_builder_args.output_dso_path)


def use_et_backend() -> bool:
    global active_builder_args

    # eager == aoti, which is when backend has not been explicitly set
    if not active_builder_args:
        return False

    if active_builder_args.output_pte_path and active_builder_args.output_dso_path:
        raise RuntimeError(
            "code generation needs to choose different implementations for DSO and PTE path.  Please only use one export option, and call export twice if necessary!"
        )

    return bool(active_builder_args.output_pte_path)


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

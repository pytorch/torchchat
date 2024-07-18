# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch

##########################################################################
###                       unpack packed weights                        ###


def unpack_packed_weights(
    packed_weights: Dict[str, Any],
    packed_linear: Callable,
    input_dtype: torch.dtype,
    unpacked_dims: Tuple,
) -> torch.Tensor:
    """Given a packed weight matrix `packed_weights`, a Callable
    implementing a packed linear function for the packed format, and the
    unpacked dimensions of the weights, recreate the unpacked weight
    matrix.  In addition to the packed weights, as a dictionary to specify
    whatever arguments the packed routine expects, we also need the input
    data type because packing may depend on input dtype, or only some
    input dtypes may be supported. We also need the dimensions of the
    unpacked matrix.  At present, this does not handle padding, but that will
    be straightforward to add. Similarly, the same approach can be used
    for both linear and mm operators.

        Args:
            packed_weights: Dict[str, Any],
            packed_linear: Callable,
            input_dtype: torch.dtype,
            unpacked_dims: Optional[Tuple]=None

        Example usage:
            packed_weights = {
                 "weight" : weight_int4pack,
                 "qGroupSize": groupsize,
                 "scales_and_zeros": scales_and_zeros
            }
            unpacked_weights = unpack_packed_weights(
                 _weight_int4pack_linear,
                 packed_weights,
                 torch.bfloat6,
                 (256, 1024),
            )


    """
    assert len(unpacked_dims) == 2, "unpacked_dims must be a tuple of length 2"
    cols = unpacked_dims[1]

    unpacked_weights = packed_linear(
        torch.eye(cols, dtype=input_dtype), **packed_weights
    ).transpose(0, 1)
    return unpacked_weights


##########################################################################
###     set and get target backend is aoti or et for this model        ###

active_builder_args_dso = None
active_builder_args_pte = None
active_builder_args_aoti_package = None


def set_backend(dso, pte, aoti_package):
    global active_builder_args_dso
    global active_builder_args_pte
    active_builder_args_dso = dso
    active_builder_args_aoti_package = aoti_package
    active_builder_args_pte = pte


def use_aoti_backend() -> bool:
    global active_builder_args_dso
    global active_builder_args_aoti_package
    global active_builder_args_pte

    # eager == aoti, which is when backend has not been explicitly set
    if (not active_builder_args_pte) and (not active_builder_args_aoti_package):
        return True

    if active_builder_args_pte and active_builder_args_aoti_package:
        raise RuntimeError(
            "code generation needs to choose different implementations for AOTI and PTE path.  Please only use one export option, and call export twice if necessary!"
        )

    return bool(active_builder_args_dso) or bool(active_builder_args_aoti_package)


def use_et_backend() -> bool:
    global active_builder_args_dso
    global active_builder_args_aoti_package
    global active_builder_args_pte

    # eager == aoti, which is when backend has not been explicitly set
    if (not active_builder_args_pte) and (not active_builder_args_aoti_package):
        return True

    if active_builder_args_pte and active_builder_args_aoti_package:
        raise RuntimeError(
            "code generation needs to choose different implementations for AOTI and PTE path.  Please only use one export option, and call export twice if necessary!"
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


def name_to_dtype(name, device):
    if (name == "fast") or (name == "fast16"):
        # MacOS now supports bfloat16
        import platform

        if platform.processor() == "arm":
            device = get_device_str(device)
            # ARM CPU is faster with float16, MPS with bf16 if supported
            if device == "cpu" or int(platform.mac_ver()[0].split(".")[0]) < 14:
                return torch.float16
        return torch.bfloat16

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
    "fast": None,
    "fast16": None,
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
###                move state dict to specified device                ###


def state_dict_device(d, device="cpu") -> Dict:
    for key, weight in d.items():
        d[key] = weight.to(device=device)

    return d


#########################################################################
###                move state dict to specified device                ###


def is_mps_available() -> bool:
    if not torch.backends.mps.is_available():
        return False

    # out system says mps is available, but it's not on VMs
    # so let's set up some memry, and see if that work:
    try:
        mps_tensor = torch.zeros(1024, dtype=torch.float16, device="mps")
    except RuntimeError:
        return False

    # MPS, is that you?
    return True


def get_device_str(device) -> str:
    if isinstance(device, str) and device == "fast":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if is_mps_available() else "cpu"
        )
        return device
    else:
        return str(device)


def get_device(device) -> str:
    if isinstance(device, str) and device == "fast":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if is_mps_available() else "cpu"
        )
    return torch.device(device)


def is_cuda_or_cpu_device(device) -> bool:
    return device == "" or str(device) == "cpu" or ("cuda" in str(device))


def is_cpu_device(device) -> bool:
    return device == "" or str(device) == "cpu"

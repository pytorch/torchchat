# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import platform
from subprocess import check_output

import torch


def get_device_info(device: str) -> str:
    """Returns a human-readable description of the hardware based on a torch.device.type

    Args:
        device: A torch.device.type string: one of {"cpu", "cuda", "xpu"}.
    Returns:
        str: A human-readable description of the hardware or an empty string if the device type is unhandled.

    For example, if torch.device.type == "cpu" on a MacBook Pro, this function will return "Apple M1 Pro".
    """
    if device == "cpu":
        if platform.system() == "Darwin":
            return (
                check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .decode("utf-8")
                .strip()
            )
        if platform.system() == "Linux":
            return (
                check_output(
                    ["sed", "-nr", "s/^model name\\s+: (.*)$/\\1/p", "/proc/cpuinfo"]
                )
                .decode("utf-8")
                .split("\n")[0]
            )
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "xpu":
        return (
            check_output(
                ["xpu-smi discovery |grep 'Device Name:'"], shell=True
            )
            .decode("utf-8")
            .split("\n")[0]
            .split("Device Name:")[1]
            )
    return ""

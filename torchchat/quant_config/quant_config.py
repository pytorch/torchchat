# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Union

"""
Known Quantization Configs:
For devices that are known to work with torch-quant, we provide a config under
config/data/quant.json to support automatically determining the quantization
options for use with torch-quant.
"""


# Specifies the execution mode to run models in. Enum variants are specified as
# strings to simplify JSON (de)serialization.
class ExecutionMode(str, Enum):
    # Run models in eager mode.
    EAGER = "eager"
    # Run models in compiled mode.
    COMPILE = "compile"


@dataclass
class QuantScheme:
    scheme: str = field(default="")
    weight_dtypes: List[int] = field(default_factory=list)
    activation_dtypes: List[int] = field(default_factory=list)


@dataclass
class QuantizationOptions:
    quant_schemes: List[QuantScheme] = field(default_factory=list)
    embedding_quant_schemes: List[QuantScheme] = field(default_factory=list)
    weight_group_sizes: List[int] = field(default_factory=list)
    embedding_group_sizes: List[int] = field(default_factory=list)


@dataclass
class ExecutionModeConfig:
    mode: ExecutionMode = field(default=ExecutionMode.EAGER)
    quantization_options: QuantizationOptions = field(
        default_factory=QuantizationOptions
    )


@dataclass
class ModelTypeConfig:
    type: str = field(default="")
    execution_modes: List[ExecutionModeConfig] = field(default_factory=list)


@dataclass
class QuantConfig:
    name: str = field(default="")
    model_types: List[ModelTypeConfig] = field(default_factory=list)


# Keys are stored in lowercase.
device_configs: Dict[str, QuantConfig] = None


def load_device_configs() -> Dict[str, QuantConfig]:
    global device_configs
    device_configs = {}
    with open(Path(__file__).parent / "quant.json", "r") as f:
        device_config_dict = json.load(f)
    for device in device_config_dict["devices"]:
        config = QuantConfig(**device)
        device_configs[config.name.lower()] = config
    return device_configs


def resolve_device_config(device: str) -> QuantConfig:
    device = device.lower()
    # Lazy load device config from JSON.
    if not device_configs:
        load_device_configs()
    if device not in device_configs:
        raise ValueError(f"Unknown device '{device}'.")
    return device_configs[device]


def resolve_model_type_config(device: str, model_type: str) -> ModelTypeConfig:
    device_config = resolve_device_config(device)
    for model_type_config in device_config.model_types:
        if model_type_config.type == model_type:
            return model_type_config
    raise ValueError(f"Unknown model type '{model_type}' for device '{device}'.")

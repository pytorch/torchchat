# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Sequence, Union

"""
Known Model Configs:

For models that are known to work with torchchat, we provide a config under
config/data/models.json to support automatically downloading the model and
converting to the expected format for use with torchchat.

There are two supported distribution channels:

1) HuggingFaceSnapshot: Download a model from HuggingFace.
2) DirectDownload: Download a list of model artifacts from URLs. No conversion
   is done.
"""


# Specifies the distribution channel to download model artifacts from. Enum
# variants are specified as strings to simplify JSON (de)serialization.
class ModelDistributionChannel(str, Enum):
    # Download a full model snapshot from HuggingFace, such as
    # meta-llama/Llama-2-7b-chat-hf and convert to torchchat format.
    HuggingFaceSnapshot = "HuggingFaceSnapshot"

    # Download one or more files over HTTP(S).
    DirectDownload = "DirectDownload"


@dataclass
class ModelConfig:
    name: str = field(default="")
    aliases: Sequence[str] = field(default_factory=list)
    distribution_path: Union[str, Sequence[str]] = field(default="")
    distribution_channel: ModelDistributionChannel = field(
        default=ModelDistributionChannel.HuggingFaceSnapshot
    )
    checkpoint_file: str = field(default="model.pth")
    tokenizer_file: str = field(default="tokenizer.model")
    transformer_params_key: str = field(default=None)
    prefer_safetensors: bool = field(default=False)


# Keys are stored in lowercase.
model_aliases: Dict[str, str] = None
model_configs: Dict[str, ModelConfig] = None


def load_model_configs() -> Dict[str, ModelConfig]:
    global model_aliases
    global model_configs

    model_aliases = {}
    model_configs = {}

    with open(Path(__file__).parent / "models.json", "r") as f:
        model_config_dict = json.load(f)

    for key, value in model_config_dict.items():
        config = ModelConfig(**value)
        config.name = key

        key = key.lower()
        model_configs[key] = config

        for alias in config.aliases:
            model_aliases[alias.lower()] = key

    return model_configs


def resolve_model_config(model: str) -> ModelConfig:
    model = model.lower()
    # Lazy load model config from JSON.
    if not model_configs:
        load_model_configs()

    if model in model_aliases:
        model = model_aliases[model]

    if model not in model_configs:
        raise ValueError(f"Unknown model '{model}'.")

    return model_configs[model]

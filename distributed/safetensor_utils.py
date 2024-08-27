# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Callable, Optional, Tuple, List, Set
import torch
from transformers import AutoTokenizer  # AutoConfig
from safetensors import safe_open
from transformers.utils import cached_file
import os
import json


_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"


def get_hf_tokenizer(model_id: str) -> AutoTokenizer:
    """Get the HF tokenizer for a given model id"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert tokenizer is not None, f"Tokenizer not found for model id {model_id}"
    return tokenizer


def get_hf_config_file(model_id: str) -> Tuple[str, str]:
    """Get the config file and file location for a given HF model id"""
    config_file = cached_file(model_id, _CONFIG_NAME)
    assert os.path.exists(config_file), f"Config file {config_file} does not exist."
    with open(config_file, "r") as file:
        config_data = json.load(file)
    file_location = os.path.dirname(config_file)
    return config_data, file_location


def get_hf_path_from_model_id(model_id: str) -> str:
    """Get the HF path for a given HF model id"""
    config_data, file_location = get_config_file(model_id)
    assert os.path.exists(
        file_location
    ), f"HF path {file_location} for {model_id} does not exist."
    return file_location

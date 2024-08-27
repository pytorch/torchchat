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
from torch.nn import Module



_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"



def compare_and_reverse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    t1 = tensor1.shape
    t2 = tensor2.shape
    if t1 == t2:
        return tensor2

    if tensor1.shape == tensor2.shape[::-1]:
        # start_shape = tensor2.shape
        # tensor2.copy_(tensor2.flip(dims=tuple(range(tensor2.dim()))))
        tensor2 = tensor2.permute(*reversed(range(tensor2.dim())))
        # logger.info(f"Reversed tensor2 from {start_shape} =====>>>> {tensor2.shape=}")
        return tensor2
    else:
        assert False, f"tensor1.shape {tensor1.shape} != tensor2.shape {tensor2.shape} and no match if reversed."


def read_weights_from_json(file_path: str) -> Optional[Dict[str, str]]:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        if "weight_map" in data and isinstance(data["weight_map"], dict):
            return data["weight_map"]
        else:
            print("No 'weight_map' dictionary found in the JSON file.")
            return None
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"An error occurred while reading the JSON file: {str(e)}")
        return None


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


def get_hf_weight_map_and_path(
    model_id: str,
) -> Tuple[
    Dict[str, str],
    str,
]:
    """Get the weight map for a given HF model id and also the cache path for loading the weights"""
    index_file = cached_file(model_id, _DEFAULT_SAFETENSOR_FILE_NAME)
    print(f"Index file: {index_file}")
    assert os.path.exists(
        index_file
    ), f"Weight index file for {model_id} does not exist in HF cache...."
    weight_map = read_weights_from_json(index_file)

    assert weight_map is not None, f"Weight map not found in config file {index_file}"
    weight_map, new_to_old_keymap = remap_weight_keys(weight_map)

    weight_path = os.path.dirname(index_file)
    assert os.path.exists(weight_path), f"Weight path {weight_path} does not exist"

    return weight_map, weight_path, new_to_old_keymap


def remap_weight_keys(dictionary):
    """Remap the keys of a dictionary to match the expected format of the tune model."""
    replacements = {
        "embed_tokens": "tok_embeddings",
        "input_layernorm.weight": "sa_norm.scale",
        "self_attn": "attn",
        "o_proj": "output_proj",
        "post_attention_layernorm.weight": "mlp_norm.scale",
        "down_proj": "w1",
        "gate_proj": "w3",
        "up_proj": "w2",
        "norm.weight": "norm.scale",
        "lm_head.weight": "output.weight",
    }

    new_dict = {}
    key_mapping = {}

    for old_key, value in dictionary.items():
        new_key = old_key
        for old_word, new_word in replacements.items():
            if old_word in new_key:
                new_key = new_key.replace(old_word, new_word)
                # logger.info(f"Old key: {old_key}, {value=}, New key: {new_key}")

        new_dict[new_key] = value
        key_mapping[new_key] = old_key
    return new_dict, key_mapping


def load_safetensor_weights(
    stage_module: Module,
    weight_map: Dict[str, str],
    file_location: str,
    new_to_old_keymap: Dict[str, str],
    device: torch.device,
    purge_model_prefix: bool = True,
    ignore_cache_layers: bool = True,
) -> Tuple[int, int]:
    """
    Load safetensor weights into a stage module.

    Args:
        stage_module (Module): The PyTorch module to load weights into.
        weight_map (Dict[str, str]): Mapping of model parameters to file names.
        file_location (str): Directory containing the weight files.
        new_to_old_keymap (Dict[str, str]): Mapping of new parameter names to old ones.
        device (torch.device): The device to load tensors onto.
        purge_model_prefix (bool): Whether to remove 'model.' prefix from keys.
        ignore_cache_layers (bool): Whether to ignore cache layers when reporting missing keys.

    Returns:
        Tuple[int, int]: Number of updated weights and number of missing weights.
    """
    stage_state_dict, weight_map = prepare_state_dict(
        stage_module, weight_map, purge_model_prefix
    )
    needed_files = get_needed_files(stage_state_dict, weight_map)
    updated_states: Set[str] = set()

    for file in needed_files:
        full_path = os.path.join(file_location, file)
        logger.info(f"Loading checkpoint file: {full_path}")
        try:
            checkpoint = load_checkpoint(full_path, "cpu")  # device)
            update_state_dict(
                stage_state_dict,
                checkpoint,
                weight_map,
                new_to_old_keymap,
                file,
                updated_states,
            )
        except FileNotFoundError:
            logger.error(f"File not found: {full_path}")
        except Exception as e:
            logger.error(f"Error loading {full_path}: {str(e)}")

    missing_keys = handle_missing_keys(
        stage_state_dict, updated_states, ignore_cache_layers
    )
    log_loading_status(missing_keys, updated_states)

    stage_module.load_state_dict(stage_state_dict, strict=False, assign=True)
    #logger.info(f"{stage_module=}")

    
    return len(updated_states), len(missing_keys)


def prepare_state_dict(
    module: Module, weight_map: Dict[str, str], purge_model_prefix: bool
) -> Dict[str, torch.Tensor]:
    state_dict = module.state_dict()
    if purge_model_prefix:
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        weight_map = {k.removeprefix("model."): v for k, v in weight_map.items()}
    return state_dict, weight_map


def get_needed_files(
    state_dict: Dict[str, torch.Tensor], weight_map: Dict[str, str]
) -> Set[str]:
    needed_files = set()
    for param in state_dict.keys():
        file = weight_map.get(param)
        if file:
            needed_files.add(file)
        elif param.endswith("weight"):
            logger.warning(
                f"Parameter {param} not found in weight map, please check..."
            )
            raise ValueError(f"Missing file for {param} in {weight_map.keys()}")
    logger.info(f"Needed files: {needed_files}")
    return needed_files


def load_checkpoint(full_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(full_path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    logger.info(f"Loaded {len(tensors)} tensors from {full_path}")
    return tensors


def update_state_dict(
    state_dict: Dict[str, torch.Tensor],
    checkpoint: Dict[str, torch.Tensor],
    weight_map: Dict[str, str],
    new_to_old_keymap: Dict[str, str],
    file: str,
    updated_states: Set[str],
):
    for param, file_with_param in weight_map.items():
        if file_with_param == file and param in state_dict:
            model_param = (
                "output.weight" if param == "output.weight" else f"model.{param}"
            )
            old_param = new_to_old_keymap.get(model_param)

            if old_param not in checkpoint:
                logger.warning(f"Missing {old_param} in checkpoint")
                continue

            checkpoint_tensor = checkpoint[old_param]
            stage_tensor = state_dict[param]

            checkpoint_tensor = compare_and_reverse(stage_tensor, checkpoint_tensor)
            state_dict[param] = checkpoint_tensor

            log_tensor_info(param, state_dict[param])
            updated_states.add(param)


def log_tensor_info(param: str, tensor: torch.Tensor):
    tensor_type = get_tensor_type(tensor)
    if tensor_type == "Fake":
        assert False, f"name: {param=}, {tensor=} is fake tensor"
    #logger.info(f"**** post-load {param}[0] = Tensor Type: {tensor_type} {tensor[0]}")
    #state_param_details = format_tensor_info(tensor)
    #logger.info(f"**** post-load {param} {state_param_details}")


def format_tensor_info(tensor: torch.Tensor) -> str:
    return f"Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}"


def handle_missing_keys(
    state_dict: Dict[str, torch.Tensor],
    updated_states: Set[str],
    ignore_cache_layers: bool,
) -> Set[str]:
    missing_keys = set(state_dict.keys()) - updated_states
    if ignore_cache_layers:
        start_len = len(missing_keys)
        missing_keys = {k for k in missing_keys if not k.endswith(".cache")}
        after_len = len(missing_keys)
        if after_len < start_len:
            logger.info(f"Ignoring {start_len - after_len} missing cache layers")
    return missing_keys


def log_loading_status(missing_keys: Set[str], updated_states: Set[str]):
    if missing_keys:
        logger.warning(
            f"Partially updated state dict. Missing {len(missing_keys)} keys: {missing_keys}"
        )
    else:
        logger.info("Fully updated state dict.")
    logger.info(f"Loaded {len(updated_states)} weights into stage module")

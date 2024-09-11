# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from transformers.utils import cached_file
import os
import json
from torch.nn import Module
from typing import Dict, Tuple, Set, Optional


from distributed.dtensor_utils import is_dtensor, load_into_dtensor


_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"


from distributed.logging_utils import SingletonLogger
logger = SingletonLogger.get_logger()



def compare_and_reverse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Used to compare and reverse the tensors for loading from safetensor shapes, if needed"""
    if tensor1.shape == tensor2.shape:
        return tensor2
    if tensor1.shape == tensor2.shape[::-1]:
        return tensor2.permute(*reversed(range(tensor2.dim())))
    raise ValueError(
        f"Tensor shapes {tensor1.shape} and {tensor2.shape} are incompatible."
    )


def read_weights_from_json(file_path: str) -> Optional[Dict[str, str]]:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        if "weight_map" in data and isinstance(data["weight_map"], dict):
            return data["weight_map"]
        else:
            logger.info("No 'weight_map' dictionary found in the JSON file.")
            return None
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        logger.info(f"An error occurred while reading the JSON file: {str(e)}")
        return None


def get_hf_tokenizer(model_id: str) -> AutoTokenizer:
    """Get the HF tokenizer for a given model id"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer is None:
        raise ValueError(f"Tokenizer not found for model id {model_id}")
    return tokenizer


def get_hf_config_file(model_id: str) -> Tuple[Dict, str]:
    """Get the config file and file location for a given HF model id"""
    config_file = cached_file(model_id, _CONFIG_NAME)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist.")
    with open(config_file, "r") as file:
        config_data = json.load(file)
    return config_data, os.path.dirname(config_file)


def get_hf_path_from_model_id(model_id: str) -> str:
    """Get the HF path for a given HF model id"""
    config_data, file_location = get_hf_config_file(model_id)
    assert os.path.exists(
        file_location
    ), f"HF path {file_location} for {model_id} does not exist."
    return file_location


def get_hf_weight_map_and_path(
    model_id: str,
) -> Tuple[Dict[str, str], str, Dict[str, str]]:
    """Get the weight map for a given HF model id and also the cache path for loading the weights"""
    index_file = cached_file(model_id, _DEFAULT_SAFETENSOR_FILE_NAME)
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"Weight index file for {model_id} does not exist in HF cache."
        )
    weight_map = read_weights_from_json(index_file)
    if weight_map is None:
        raise ValueError(f"Weight map not found in config file {index_file}")
    weight_map, new_to_old_keymap = remap_weight_keys(weight_map)
    weight_path = os.path.dirname(index_file)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight path {weight_path} does not exist")
    return weight_map, weight_path, new_to_old_keymap


def remap_weight_keys(dictionary):
    """Remap the keys of a dictionary to match the expected format of the tune model."""
    # hf_key : dist_model_key
    replacements = {
        "embed_tokens": "tok_embeddings",
        "input_layernorm.weight": "attention_norm.weight",
        "self_attn": "attention",
        "o_proj": "wo",
        "k_proj": "wk",
        "v_proj": "wv",
        "q_proj": "wq",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "down_proj": "w2",
        "gate_proj": "w1",
        "up_proj": "w3",
        "lm_head.weight": "output.weight",
        "mlp": "feed_forward",
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
    device: torch.device = "cuda",
    purge_model_prefix: bool = True,
    ignore_cache_layers: bool = True,
    model_config: Optional[Dict] = None,
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
        model_config (Optional[Dict]): Model configuration.

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
        # logger.info(f"Loading checkpoint file: {full_path}")
        try:
            checkpoint = load_checkpoint(full_path, "cpu")  # device)

            update_state_dict(
                stage_state_dict,
                checkpoint,
                weight_map,
                new_to_old_keymap,
                file,
                updated_states,
                device,
                model_config,
            )
        except FileNotFoundError:
            logger.error(f"File not found: {full_path}")
        except Exception as e:
            logger.error(f"Error during checkpoint processing of {full_path}: {str(e)}")

    missing_keys = handle_missing_keys(
        stage_state_dict, updated_states, ignore_cache_layers
    )
    # log_loading_status(missing_keys, updated_states)
    if missing_keys:
        logger.warning(
            f"Partially updated state dict. Missing {len(missing_keys)} keys: {missing_keys}"
        )
    else:
        logger.info("Fully updated state dict.")

    logger.info(f"Loading {len(updated_states)} weights into stage dict")
    # precount, premap = record_module_dtypes(stage_module)
    stage_module.load_state_dict(stage_state_dict, strict=False, assign=True)
    # postcount, postmap = record_module_dtypes(stage_module)
    # logger.info(f"{precount=}, {postcount=}")
    # logger.info(f"{premap=}, {postmap=}")

    logger.info(f"Successfully loaded {len(updated_states)} weights into stage module")

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


def permute_weight_to_attn_heads(w, n_heads, head_dim, model_dim):
    """Permute the weight tensor to match the attention heads."""
    # TODO - this is borrowed from chat/convert_hf...we should expose this as a direct function
    return (
        w.view(n_heads, 2, head_dim // 2, model_dim)
        .transpose(1, 2)
        .reshape(head_dim * n_heads, model_dim)
    )


def update_state_dict(
    state_dict: Dict[str, torch.Tensor],
    checkpoint: Dict[str, torch.Tensor],
    weight_map: Dict[str, str],
    new_to_old_keymap: Dict[str, str],
    file: str,
    updated_states: Set[str],
    device: torch.device,
    model_config: Optional[Dict] = None,
):
    count_dtensors_loaded = 0
    # for handling attn head permuting
    num_heads = model_config.n_heads
    dim = model_config.dim
    num_local_heads = model_config.n_local_heads
    head_dim = model_config.head_dim

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

            stage_is_dtensor = is_dtensor(stage_tensor)

            if "wq" in param:
                checkpoint_tensor = permute_weight_to_attn_heads(
                    checkpoint_tensor, num_heads, head_dim, dim
                )
            elif "wk" in param:
                checkpoint_tensor = permute_weight_to_attn_heads(
                    checkpoint_tensor, num_local_heads, head_dim, dim
                )

            # here we need to check if the tensor is a DTensor and if so, adjust the
            # shape and placement to match the model DTensor.
            if stage_is_dtensor:
                model_tensor = load_into_dtensor(checkpoint_tensor, stage_tensor)
                # logger.info(f"DTensor: Loaded {param} into {model_tensor=}")
                state_dict[param] = model_tensor
                count_dtensors_loaded += 1

            else:
                # regular tensor, just update directly
                checkpoint_tensor = checkpoint_tensor.to(device)
                state_dict[param] = checkpoint_tensor

            # ensure matching dtypes
            state_dict[param] = state_dict[param].to(checkpoint_tensor.dtype)

            assert state_dict[param].dtype == checkpoint_tensor.dtype

            # log_tensor_info(param, state_dict[param])
            # logger.info(f"Loaded {param} from {file}")
            updated_states.add(param)
    # logger.info(f"Count of loaded DTensors: {count_dtensors_loaded}")


def format_tensor_info(tensor: torch.Tensor) -> str:
    return f"Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}"


def clean_cache_keys(input_set: Set[str]) -> Set[str]:
    """Remove cache, freqs, mask params from checkpoint update set...we expect these to be generated"""
    return {
        item
        for item in input_set
        if not (item.endswith("cache") or item in ["freqs_cis", "causal_mask"])
    }


def handle_missing_keys(
    state_dict: Dict[str, torch.Tensor],
    updated_states: Set[str],
    ignore_cache_layers: bool,
) -> Set[str]:
    """This function handles 'expected' missing keys from the checkpoint update set.
    This is used for ignoring cache, rope freqs, and mask layers that are generated, rather than persisted
    in the checkpoint."""
    missing_keys = set(state_dict.keys()) - updated_states

    if ignore_cache_layers:
        start_len = len(missing_keys)
        missing_keys = clean_cache_keys(missing_keys)
        after_len = len(missing_keys)
        if after_len < start_len:
            logger.info(
                f"Ignoring {start_len - after_len} missing cache, freqs, mask layers"
            )
    return missing_keys


def log_loading_status(missing_keys: Set[str], updated_states: Set[str]):
    if missing_keys:
        logger.warning(
            f"Partially updated state dict. Missing {len(missing_keys)} keys: {missing_keys}"
        )
    else:
        logger.info("Fully updated state dict.")
    logger.info(f"Successfully loaded {len(updated_states)} weights into stage module")


from typing import Dict, Callable, Optional, Tuple
import torch
from transformers import AutoTokenizer # AutoConfig
from safetensors import safe_open
from transformers.utils import cached_file
import logging
import os
import json
import torch.distributed as dist

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# default hf cache lives in ~/.cache/huggingface/hub

_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"

def get_hf_tokenizer(model_id: str) -> AutoTokenizer:
    """ Get the HF tokenizer for a given model id """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert tokenizer is not None, f"Tokenizer not found for model id {model_id}"
    return tokenizer

def init_buffers(
    stage_module: torch.nn.Module,
    device: torch.device,
    init_callbacks: Dict[str, Callable],
    model_config: Optional[Dict[str, str]] = None,
    buffer_dtype: Optional[torch.dtype] = None, # torch.bfloat16,
) -> None:
        """
        Initialize buffers of `stage_module` using the callbacks in `init_callbacks`.

        Args:
            stage_module (torch.nn.Module): The module whose buffers are to be initialized.
            device (torch.device): The device on which to initialize the buffers.
            init_callbacks (Dict[str, Callable]): A dictionary mapping buffer FQNs to their init functions.
            model_config (Optional[Dict[str, str]]): Additional model configuration (unused in this function).

        Returns:
            None
        """
        for name, buf in stage_module.named_buffers():
            #logger.info(f"INSIDE - Checking buffer {name}, {buf=}")
            for buffer_name_to_init, init_func in init_callbacks.items():
                #logger.info(f"INSIDE - Checking buffer {name} against {buffer_name_to_init}")
                if _should_initialize_buffer(name, buffer_name_to_init):
                    #logger.info(f"INSIDE - About to Initializing buffer {name} with {init_func}")
                    _initialize_buffer(stage_module, name, init_func, device, buffer_dtype )
                    break

def _should_initialize_buffer(buffer_name: str, pattern: str) -> bool:
        """
        Determine if a buffer should be initialized based on its name and a pattern.

        Args:
            buffer_name (str): The name of the buffer.
            pattern (str): The pattern to match against.

        Returns:
            bool: True if the buffer should be initialized, False otherwise.
        """
        has_pattern, exact_buf_name = remove_pattern_prefix(pattern)
        if has_pattern:
            return exact_buf_name in buffer_name
        else:
             return exact_buf_name == buffer_name
        return False

def _initialize_buffer(module: torch.nn.Module, buffer_name: str, init_func: Callable, device: torch.device, buffer_dtype=None) -> None:
    """
    Initialize a specific buffer in the module.

    Args:
        module (torch.nn.Module): The module containing the buffer.
        buffer_name (str): The name of the buffer to initialize.
        init_func (Callable): The initialization function.
        device (torch.device): The device on which to initialize the buffer.

    Returns:
        None
    """
    logger.info(f"INSIDE - Initializing buffer {buffer_name} with {init_func}")
    buf_val = init_func(device)
    if buffer_dtype is not None:
        buf_val = buf_val.to(buffer_dtype)
        logger.info(f"Buffer dtype set to: {buf_val.dtype}")
    module_path = buffer_name.split('.')
    target_module = module
    for submodule in module_path[:-1]:
        target_module = getattr(target_module, submodule)
    target_module.register_buffer(module_path[-1], buf_val, persistent=False)
    logger.info(f"Initialized buffer {buffer_name}")
    logger.info(f"Buffer result: {target_module=}")
    

def open_hf_safetensor(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.endswith(".safetensors"):
            raise ValueError("File does not have .safetensors extension")

        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        return tensors
    
    except Exception as e:
        print(f"An error occurred while opening the safetensor file: {str(e)}")
        return None


def remove_pattern_prefix(s):
    """ Remove the prefix 'pattern.' from a string and return bool re: if it was present 
    example: has_pattern, result = remove_pattern_prefix(input_string)
    """
    prefix = "pattern."
    if s.startswith(prefix):
        return True, s[len(prefix):]
    else:
        return False, s


def load_safetensor_weights(
    stage_module: torch.nn.Module,
    weight_map: Dict[str, str],
    file_location: str,
    new_to_old_keymap: Dict[str, str],
    purge_model_prefix: bool = True,
):
    stage_state_dict = stage_module.state_dict()
    if purge_model_prefix:
        stage_state_dict = {key.removeprefix('model.'): value for key, value in stage_state_dict.items()}
        weight_map = {key.removeprefix('model.'): value for key, value in weight_map.items()}

    print(f"Stage state dict: {stage_state_dict.keys()}")
    updated_states = {}

    needed_files = set() 
    #    weight_map.get(param, None)
    #    for param in stage_state_dict.keys()
    #    if weight_map.get(param, None)
    #)

    # The file a param is saved in
    for param in stage_state_dict.keys():
        file = weight_map.setdefault(param, None)
        print(f"mapping {file=}, {param=}")
        if file:
            needed_files.add(file)
    print(f"Needed files: {needed_files=}")

    for file in needed_files:
        print(f"Loading file: {file}")
        full_path = os.path.join(file_location, file)
        checkpoint = open_hf_safetensor(full_path)
        if checkpoint is None:
            continue

        for param in stage_state_dict.keys():
            file_with_param = weight_map.get(param, None)
            if not file_with_param:
                print(f"Warning: {param} not found in weight map, skipping")
            elif weight_map.get(param) == file:
                print(f"Loading param: {param}")
                old_param = new_to_old_keymap.get(param)
                print(f"REMAPPED - {param=} -> {old_param=}")
            
                if old_param in checkpoint:
                    print(f"Loading {old_param} param within: {param}")
                    stage_state_dict[param] = checkpoint[old_param]
                    updated_states[param] = None
                

    print(
        "Fully updated state dict"
        if stage_state_dict.keys() == updated_states.keys()
        else "Partially updated state dict"
    )
    stage_module.load_state_dict(stage_state_dict, assign=True)
    print(f"Loaded {len(updated_states)} weights into stage module")
    dist.barrier()
    assert False, "check weights"

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

def get_config_file(model_id: str) -> Tuple[str,str]:
    """ Get the config file and file location for a given HF model id """
    config_file = cached_file(model_id, _CONFIG_NAME)
    assert os.path.exists(config_file), f"Config file {config_file} does not exist."
    with open(config_file, "r") as file:
        config_data = json.load(file)
    file_location = os.path.dirname(config_file)
    return config_data, file_location

def get_hf_weight_map_and_path(model_id: str) -> Tuple[Dict[str, str], str,]:
    """ Get the weight map for a given HF model id and also the cache path for loading the weights """
    index_file = cached_file(model_id, _DEFAULT_SAFETENSOR_FILE_NAME)
    print(f"Index file: {index_file}")
    assert os.path.exists(index_file), f"Weight index file for {model_id} does not exist in HF cache...."
    weight_map = read_weights_from_json(index_file)

    assert weight_map is not None, f"Weight map not found in config file {index_file}"
    weight_map, new_to_old_keymap = remap_weight_keys(weight_map)
    
    weight_path = os.path.dirname(index_file)
    assert os.path.exists(weight_path), f"Weight path {weight_path} does not exist"

    return weight_map, weight_path, new_to_old_keymap

def remap_weight_keys(dictionary):
    """ Remap the keys of a dictionary to match the expected format of the tune model. """
    replacements = {
        'embed_tokens': 'tok_embeddings',
        'input_layernorm.weight': 'sa_norm.scale',
        'self_attn':'attn',
        'o_proj':'output_proj',
        'post_attention_layernorm.weight':'mlp_norm.scale',
        'down_proj':'w1',
        'gate_proj':'w2',
        'up_proj':'w3',
        'norm.weight':'norm',
        'lm_head':'output',
        
    }
    
    new_dict = {}
    key_mapping = {}
    
    for old_key, value in dictionary.items():
        new_key = old_key
        for old_word, new_word in replacements.items():
            if old_word in new_key:
                new_key = new_key.replace(old_word, new_word)
        
        new_dict[new_key] = value
        # if new_key != old_key:
        key_mapping[new_key] = old_key
    
    return new_dict, key_mapping

    
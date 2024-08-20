from typing import Dict, Callable, Optional, Tuple, List, Set
import torch
from transformers import AutoTokenizer  # AutoConfig
from safetensors import safe_open
from transformers.utils import cached_file
import logging
import os
import json
import torch.distributed as dist
import time
from torch._subclasses import FakeTensor

from safetensors.torch import load_file
from modeling_utils import torch_in_fake_mode, get_tensor_type
from torch.nn import Module

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# default hf cache lives in ~/.cache/huggingface/hub

_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"


def get_hf_tokenizer(model_id: str) -> AutoTokenizer:
    """Get the HF tokenizer for a given model id"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert tokenizer is not None, f"Tokenizer not found for model id {model_id}"
    return tokenizer


def init_buffers(
    stage_module: torch.nn.Module,
    device: torch.device,
    init_callbacks: Dict[str, Callable],
    model_config: Optional[Dict[str, str]] = None,
    buffer_dtype: Optional[torch.dtype] = None,  # torch.bfloat16,
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
        # logger.info(f"INSIDE - Checking buffer {name}, {buf=}")
        for buffer_name_to_init, init_func in init_callbacks.items():
            # logger.info(f"INSIDE - Checking buffer {name} against {buffer_name_to_init}")
            if _should_initialize_buffer(name, buffer_name_to_init):
                # logger.info(f"INSIDE - About to Initializing buffer {name} with {init_func}")
                _initialize_buffer(stage_module, name, init_func, device, buffer_dtype)
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


def _initialize_buffer(
    module: torch.nn.Module,
    buffer_name: str,
    init_func: Callable,
    device: torch.device,
    buffer_dtype=None,
) -> None:
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
    module_path = buffer_name.split(".")
    target_module = module
    for submodule in module_path[:-1]:
        target_module = getattr(target_module, submodule)
    target_module.register_buffer(module_path[-1], buf_val, persistent=False)
    logger.info(f"Initialized buffer {buffer_name}")
    logger.info(f"Buffer result: {target_module=}")


def open_hf_safetensor(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Open a SafeTensors file, return all its contents in a dictionary, and check for fake tensors.

    Args:
        file_path (str): The path to the SafeTensors file.

    Returns:
        Tuple[Dict[str, torch.Tensor], List[str]]: A tuple containing:
            - A dictionary where keys are tensor names and values are the corresponding tensors.
            - A list of names of any fake tensors found.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If there's an issue reading the SafeTensors file.
    """
    try:
        tensors = load_file(file_path)
        #for name, tensor in tensors.items():
        #    logger.info(f"Loaded tensor '{name}' with shape {tensor.shape}")
        
        return tensors

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except Exception as e:
        #logger.info(f"Error in open_hf_safetensor: reading SafeTensors file: {str(e)}")
        raise ValueError(f"Error in open_hf_safetensor for {file_path}: reading SafeTensors file: {str(e)}")

        

'''def open_hf_safetensor_old(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.endswith(".safetensors"):
            raise ValueError("File does not have .safetensors extension")
        
        tensors = {}

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        #logger.info(f"Loaded {len(tensors)} tensors from {file_path}")
        #logger.info(f"Tensor keys: {list(tensors.keys())}")
        #logger.info(f"Tensor shapes: {[t.shape for t in tensors.values()]}")
        
        return tensors

    except Exception as e:
        print(f"An error occurred while opening the safetensor file: {str(e)}")
        return None
'''

def remove_pattern_prefix(s):
    """Remove the prefix 'pattern.' from a string and return bool re: if it was present
    example: has_pattern, result = remove_pattern_prefix(input_string)
    """
    prefix = "pattern."
    if s.startswith(prefix):
        return True, s[len(prefix) :]
    else:
        return False, s



def format_tensor_info(tensor: torch.Tensor) -> str:
    """
    Format tensor information including shape, dtype, device, and type for debugging.
    
    Args:
    tensor (torch.Tensor): The input tensor to analyze.
    
    Returns:
    str: A formatted string containing tensor information.
    """
    # Get shape
    shape = str(tensor.shape)
    
    # Get dtype
    dtype = str(tensor.dtype)
    
    # Get device
    device = str(tensor.device)
    
    # Determine tensor type
    if tensor.is_meta:
        tensor_type = "Meta"
    elif isinstance(tensor, FakeTensor):
        tensor_type = "Fake"
    else:
        tensor_type = "Regular"
    
    # Format the information
    info = f"Shape: {shape}, Dtype: {dtype}, Device: {device}, Type: {tensor_type}"
    
    return info

def new_compare_and_reverse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    t1 = tensor1.shape
    t2 = tensor2.shape
    if t1 == t2:
        return tensor2

    if tensor1.shape == tensor2.shape[::-1]:
        start_shape = tensor2.shape
        # tensor2.copy_(tensor2.flip(dims=tuple(range(tensor2.dim()))))
        tensor2 = tensor2.permute(*reversed(range(tensor2.dim())))
        logger.info(f"Reversed tensor2 from {start_shape} =====>>>> {tensor2.shape=}")
        return tensor2
    else:
        assert False, f"tensor1.shape {tensor1.shape} != tensor2.shape {tensor2.shape} and no match if reversed."

def load_safetensor_weights(
    stage_module: torch.nn.Module,
    weight_map: Dict[str, str],
    file_location: str,
    new_to_old_keymap: Dict[str, str],
    _device: torch.device,
    purge_model_prefix: bool = True,
    ignore_cache_layers: bool = True,
) -> Tuple[int, int]:
    """Load safetensor weights into a stage module.
    Returns the number of weights loaded and the number of missing weights.
    weight_map = {model_param: file_with_param}  # output.weight: model.safetensors.index.json
    new_to_old_keymap = {model_param: old_param}  # output.weight: lm_head.weight
    sample from safetensor:
    "model.layers.31.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    vs tune:
    "layers.31.attn.v_proj.weight"
    """

    def remove_model_prefix(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k.removeprefix("model."): v for k, v in d.items()}

    stage_state_dict = stage_module.state_dict()
    if purge_model_prefix:
        stage_state_dict = remove_model_prefix(stage_state_dict)
        weight_map = remove_model_prefix(weight_map)

    logger.info(f"{_device=}")
    # logger.info(f"Stage state dict: len = {len(stage_state_dict)}, keys = {list(stage_state_dict.keys())}")

    updated_states = set()
    needed_files = set()
    for param in stage_state_dict.keys():
        file = weight_map.get(param)
        if file:
            needed_files.add(file)
        else:
            if param.endswith("weight"):
                    logger.warning(
                        f"**** Parameter {param} not found in weight map, please check..."
                    )
                    assert False, f"Missing file for {param} in {weight_map.keys()}"

    logger.info(f"Needed files: {needed_files}")
    
    # generic check that we have no ambient fake mode
    torch_mode_fake = torch_in_fake_mode()
    assert torch_mode_fake is False, f"torch_in_fake_mode is {torch_mode_fake}"

    for file in needed_files:
        logger.info(f"Loading checkpoint file: {file}")
        full_path = os.path.join(file_location, file)
        try:
            #checkpoint = open_hf_safetensor(full_path)

            tensors = {}
            with safe_open(full_path, framework="pt", device=0) as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            logger.info(f"Loaded {len(tensors)} tensors from {file}")
            checkpoint = tensors
            #with safe_open(full_path, framework="pt", device="cpu") as checkpoint:
            for param, file_with_param in weight_map.items():
                if file_with_param == file and param in stage_state_dict:
                    # have to special case output.weight as only one not preceeded with model.
                    if param == "output.weight":
                        logger.info(
                            f"skipping model prefix for {param} from {file_with_param}"
                        )
                        model_param = param
                    else:
                        model_param = "model." + param

                    old_param = new_to_old_keymap.get(model_param)
                    if not old_param in checkpoint.keys():
                        logger.info(f"missing {old_param} in {checkpoint.keys()}")
                        assert False, f"missing {old_param}"

                    if old_param in checkpoint.keys():
                        checkpoint_tensor = checkpoint[old_param]
                        # checktensor_start_info = format_tensor_info(checkpoint_tensor)
                        # logger.info(f"checkpoint tensor before to: {checktensor_start_info}")
                        # checkpoint_tensor = checkpoint_tensor.to(_device)
                        # checktensor_after_to_info = format_tensor_info(checkpoint_tensor)
                        # logger.info(f"checkpoint tensor after to: {checktensor_after_to_info}")

                        assert checkpoint_tensor is not None, f"Tensor not found for {old_param}"
                        #checktensor_type = get_tensor_type(checkpoint_tensor)
                        #if checktensor_type == "Fake":
                        #    logger.info(f"Fake checkpoint tensor found for {old_param}, {checkpoint_tensor=}")
                        #    assert False, f"Fake checkpoint tensor found for {old_param}, {checkpoint_tensor=}"
                        stage_tensor = stage_state_dict[param]
                        #stagetensor_type = get_tensor_type(stage_tensor)
                        #if stagetensor_type == "Fake":
                        #    logger.info(f"Fake stage tensor found for {old_param}, {stage_tensor=}")
                            
                        # temp in place reverse
                        checkpoint_tensor = new_compare_and_reverse(stage_tensor, checkpoint_tensor)
                        #reversed_checkpoint_tensor_info = format_tensor_info(checkpoint_tensor)
                        #logger.info(f"checkpoint tensor after reverse: {reversed_checkpoint_tensor_info}")




                        #checkpoint_tensor = compare_and_reverse(
                        #    checkpoint_tensor, stage_tensor
                        #)
                        #logger.info(f"checkpoint tensor after reverse: {checkpoint_tensor=}")
                        #checkpoint_tensor = checkpoint_tensor.to(_device)
                        #logger.info(f"checkpoint tensor after to: {checkpoint_tensor=}")
                        #logger.info(f"\n**** pre-load {old_param=}\n {stage_state_dict[param]=}\n{checkpoint_tensor=}\n")
                        #if isinstance(checkpoint_tensor, FakeTensor):
                        #    logger.info(f"Fake checkpoint tensor found for {old_param}, {checkpoint_tensor=}")
                        #    assert False, f"Fake checkpoint tensor found for {old_param}, {checkpoint_tensor=}"
                        stage_state_dict[param] = checkpoint_tensor
                        logger.info(f"**** post-load {stage_state_dict[param][0]=}\n")
                        state_param_details = format_tensor_info(stage_state_dict[param])
                        logger.info(f"**** post-load {param} {state_param_details}\n")
                        updated_states.add(param)
                        
                    else:
                        # potentially catastrophic...
                        if param.endswith("weight"):
                            logger.warning(
                                f"**** Parameter {param} / {old_param} not found in checkpoint from {file_with_param}, please check..."
                            )
                        else:
                            # ignore cache and similar generated layers
                            logger.info(
                                f"**** Parameter {old_param} not found in checkpoint from {model_param}, skipping"
                            )

        except FileNotFoundError:
            logger.error(f"File not found: {full_path}")
        except Exception as e:
            logger.error(f"Error loading {full_path}: {str(e)}")

    missing_keys = set(stage_state_dict.keys()) - updated_states

    # ignore saying partial loading, if only missing items are cache layers (we don't load cache layers)
    if ignore_cache_layers:
        start_len = len(missing_keys)
        missing_keys = {k for k in missing_keys if not k.endswith(".cache")}
        after_len = len(missing_keys)
        if after_len < start_len:
            logger.info(f"Ignoring {start_len - after_len} missing cache layers")

    if missing_keys:
        logger.warning(
            f"Partially updated state dict. Missing {len(missing_keys)} keys: {missing_keys}"
        )
        #logger.warning(f"debug info: \n\n{stage_state_dict=}\n\n{weight_map=}\n\n")
    else:
        logger.info("Fully updated state dict.")

    stage_module.load_state_dict(stage_state_dict, strict=False, assign=True)
    logger.info(f"Loaded {len(updated_states)} weights into stage module")

    return len(updated_states), len(missing_keys)


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


def get_config_file(model_id: str) -> Tuple[str, str]:
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
    assert os.path.exists(file_location), f"HF path {file_location} for {model_id} does not exist."
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
        "gate_proj": "w2",
        "up_proj": "w3",
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
                #logger.info(f"Old key: {old_key}, {value=}, New key: {new_key}")

        new_dict[new_key] = value
        key_mapping[new_key] = old_key
    return new_dict, key_mapping


def compare_and_reverse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Compare the shapes of two tensors and permute second one to match the first one.
    This is expressly used for mapping safetensor weights to the tune models.
    """
    # Compare the shapes of the two tensors
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    if len(shape1) == len(shape2):
        return tensor2

    if shape1 == shape2[::-1]:
        return tensor2.permute(*range(tensor2.dim() - 1, -1, -1))

    return tensor2


def new_load_safetensor_weights(
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
    stage_state_dict, weight_map = prepare_state_dict(stage_module, weight_map, purge_model_prefix)
    needed_files = get_needed_files(stage_state_dict, weight_map)
    updated_states: Set[str] = set()

    for file in needed_files:
        full_path = os.path.join(file_location, file)
        logger.info(f"Loading checkpoint file: {full_path}")
        try:
            checkpoint = load_checkpoint(full_path, "cpu") # device)
            update_state_dict(stage_state_dict, checkpoint, weight_map, new_to_old_keymap, file, updated_states)
        except FileNotFoundError:
            logger.error(f"File not found: {full_path}")
        except Exception as e:
            logger.error(f"Error loading {full_path}: {str(e)}")

    missing_keys = handle_missing_keys(stage_state_dict, updated_states, ignore_cache_layers)
    log_loading_status(missing_keys, updated_states)
    
    stage_module.load_state_dict(stage_state_dict, strict=False, assign=True)
    return len(updated_states), len(missing_keys)

def prepare_state_dict(module: Module, weight_map: Dict[str, str], purge_model_prefix: bool) -> Dict[str, torch.Tensor]:
    state_dict = module.state_dict()
    if purge_model_prefix:
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        weight_map = {k.removeprefix("model."): v for k, v in weight_map.items()}
    return state_dict, weight_map

def get_needed_files(state_dict: Dict[str, torch.Tensor], weight_map: Dict[str, str]) -> Set[str]:
    needed_files = set()
    for param in state_dict.keys():
        file = weight_map.get(param)
        if file:
            needed_files.add(file)
        elif param.endswith("weight"):
            logger.warning(f"Parameter {param} not found in weight map, please check...")
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
    updated_states: Set[str]
):
    for param, file_with_param in weight_map.items():
        if file_with_param == file and param in state_dict:
            model_param = "output.weight" if param == "output.weight" else f"model.{param}"
            old_param = new_to_old_keymap.get(model_param)
            
            if old_param not in checkpoint:
                logger.warning(f"Missing {old_param} in checkpoint")
                continue

            checkpoint_tensor = checkpoint[old_param]
            stage_tensor = state_dict[param]
            
            checkpoint_tensor = new_compare_and_reverse(stage_tensor, checkpoint_tensor)
            state_dict[param] = checkpoint_tensor
            
            log_tensor_info(param, state_dict[param])
            updated_states.add(param)

def log_tensor_info(param: str, tensor: torch.Tensor):
    logger.info(f"**** post-load {param}[0] = {tensor[0]}")
    state_param_details = format_tensor_info(tensor)
    logger.info(f"**** post-load {param} {state_param_details}")

def format_tensor_info(tensor: torch.Tensor) -> str:
    return f"Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}"

def handle_missing_keys(
    state_dict: Dict[str, torch.Tensor],
    updated_states: Set[str],
    ignore_cache_layers: bool
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
        logger.warning(f"Partially updated state dict. Missing {len(missing_keys)} keys: {missing_keys}")
    else:
        logger.info("Fully updated state dict.")
    logger.info(f"Loaded {len(updated_states)} weights into stage module")

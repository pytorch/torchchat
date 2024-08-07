# $ torchrun --nproc-per-node 4 pp_meta.py

import os
import json
from typing import Optional, Dict, Tuple, Any, List
import logging
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors import safe_open
from transformers.utils import cached_file

from utils import Color
from modeling_utils import init_on_meta_device, check_rope_embedding, print_model_structure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# derived from Ke's PR:
# https://github.com/pytorch/PiPPy/pull/1135

# Model configuration

MODEL_CONFIGS = {
    "7b": "meta-llama/Llama-2-7b-chat-hf",
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "405base": "meta-llama/Meta-Llama-3.1-405B",
    "123b": "mistralai/Mistral-Large-Instruct-2407",
    "22b": "mistralai/Codestral-22B-v0.1",
}

_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"

def create_model(model_id: str, device: str = "cuda", rank: int = 0) -> Tuple[AutoModelForCausalLM, FakeTensorMode, Optional[Dict[str, Any]]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    
    with init_on_meta_device(device="meta"):
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch.bfloat16)
    model.eval()
    if rank==0:
        #print(model.config)
        print(f"{model=}")
        #print(f"{model.model.embed_tokens.dtype=}")
        print(f"{model.model.embed_tokens.weight.dtype=}")
        print(f"{model.model.layers[0].self_attn.q_proj=}") 
        print(f"{model.model.layers[0].self_attn.q_proj.weight.dtype=}")
        #embed_tokens.weight.device=}")
        
        #print_model_structure(model)
        #print(f"{model.model.rotary_emb.inv_freq.dtype=}")
        #assert False, "check dtype"
    #if rank == 0:
    #    logger.info(f"Model: {model.model.rope_emb.inv_freq=}")
    #    logger.info(f"Model: {model.model.rope_emb.inv_freq.dtype=}")
    #    assert False, "check dtype"
        #check_rope_embedding(model)

    config = model.config
    # print(f"{config=}")
   
    assert config is not None, "Config is None"
    
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Buffer callback: {model.buf_init_callbacks}")
    if not model.buf_init_callbacks:
        logger.warning("ROPE generation may not succeed - buf_init_callbacks is None")
    
    with fake_mode:
        model.to_empty(device='cuda')
    
    return model, fake_mode, config


def create_pipeline(model, inputs, world_size: int):
    layers_per_rank = model.config.num_hidden_layers // world_size
    split_spec = {
        f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }

    return pipeline(
        model,
        mb_args=(inputs,),
        mb_kwargs={
            "output_attentions": False,
            "output_hidden_states": False,
            "use_cache": False,
        },
        split_spec=split_spec,
    )

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

# --- update init ----
from typing import Dict, Callable, Optional
import torch

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
    #logger.info(f"{target_module=}")
    #assert False, "stop here"


def load_safetensor_weights(
    stage_module: torch.nn.Module,
    weight_map: Dict[str, str],
    file_location: str,
):
    stage_state_dict = stage_module.state_dict()
    updated_states = {}

    needed_files = set() 
    #    weight_map.get(param, None)
    #    for param in stage_state_dict.keys()
    #    if weight_map.get(param, None)
    #)

    # The file a param is saved in
    for param in stage_state_dict.keys():
        file = weight_map.setdefault(param, None)
        if file:
            needed_files.add(file)

    for file in needed_files:
        #checkpoint = open_hf_safetensor(os.path.join(file_location, file))
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
                if param in checkpoint:
                    stage_state_dict[param] = checkpoint[param]
                    updated_states[param] = None
                

    print(
        "Fully updated state dict"
        if stage_state_dict.keys() == updated_states.keys()
        else "Partially updated state dict"
    )
    stage_module.load_state_dict(stage_state_dict, assign=True)
    print(f"Loaded {len(updated_states)} weights into stage module")


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


def main(model_size: str, world_size: int, device: str):
    rank = int(os.environ["RANK"])
    world_size_dist = int(os.environ["WORLD_SIZE"])
    if world_size_dist != world_size:
        logger.warning(f"World size mismatch: {world_size_dist} != {world_size}. Overriding with dist world size")
        world_size = world_size_dist
    logger.info(f"Rank: {rank} / {world_size}")

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    dist.init_process_group(rank=rank, world_size=world_size)

    model_id = MODEL_CONFIGS[model_size]
    logger.info(f"Model ID: {model_id}")

    cfile = cached_file(model_id, _DEFAULT_SAFETENSOR_FILE_NAME)
    config_file = cached_file(model_id, _CONFIG_NAME)
    
    assert os.path.exists(cfile), f"Safetensor index file {cfile} does not exist."
    assert os.path.exists(config_file), f"Config file {config_file} does not exist."

    with open(config_file, "r") as file:
        config_data = json.load(file)

    file_location = os.path.dirname(cfile)

    # Create model on meta device
    model, fake_mode, model_config = create_model(model_id, device, rank)
    assert model.buf_init_callbacks is not None, "buffer_init_callbacks is None"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = ("How do you", "I like to")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    fake_ids = fake_mode.from_tensor(inputs["input_ids"])

    weight_map = read_weights_from_json(cfile)
    if weight_map is None:
        logger.error(f"No weight map found in the JSON file {cfile}.")
        return

    # Create pipeline
    logger.info("Creating pipeline...")
    pipe = create_pipeline(model, fake_ids, world_size)

    # Stage materialization
    logger.info("Materializing each stage...")
    stage_module = pipe.get_stage_module(rank)
    
    logger.info(f"Loading weights into stage {rank}")
    load_safetensor_weights(stage_module, weight_map, file_location)
    if rank == 0:
        logger.info(f"After load safe tensor Stage module type: {type(stage_module)}")
    
    logger.info("About to try to init buffers")
    if hasattr(model, "buf_init_callbacks"):
        logger.info(f"Initializing buffers with device={device}")
        init_buffers(stage_module, device, model.buf_init_callbacks, model_config)
    logger.info(f"Completed load of stage {rank}")
    if rank == 0:
        #logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")
        logger.info(f"{Color.blue}{stage_module.model.rotary_emb=}{Color.reset}")
        logger.info(f"{Color.blue}{stage_module.model.rotary_emb.inv_freq.dtype=}{Color.reset}")
        #assert False, "stop here"
    #else:
    #    logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")
    
    logger.info(f"{Color.blue}\n--->  {rank=} Successfully traced, segmented and loaded weights for model {Color.green}{MODEL_CONFIGS[model_size]}{Color.reset}")

    # Create schedule runtime
    stage = pipe.build_stage(rank, device=device)
    #if rank == 0:
    #    logger.info(f"{rank=} Completed stage building:  {stage=}...")
        #logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")

    #logger.info("TODO = continue here....returning now via early stop for debugging")
    #assert False, "stop here"

    logger.info("Pipeline Complete ---- Running schedule...")

    logger.info(f"{rank=} Completed stage building:  {stage=}...")
    #logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")
    logger.info(f"{Color.blue}{stage_module.print_readable()=}{Color.reset}")

    # Run
    # Run time inputs
    full_batch_prompts = (
        "How do you", "I like to", "Can I help", "You need to",
        "The weather is", "I found a", "What is your", "You are so",
    )  # full batch size = 8
    
    inputs = tokenizer(full_batch_prompts, return_tensors="pt", padding=True).to(device)


    # Attach to a schedule
    # number of microbatches = 8 // 2 = 4
    num_mbs = 4
    schedule = ScheduleGPipe(stage, num_mbs)

    '''if rank == 0:
        args = inputs["input_ids"]
    else:
        args = None
    '''
    
    

    if rank == 0:
        output = schedule.step(inputs['input_ids'])
    else:
        output = schedule.step()
    
    #output = schedule.step(args)

    # Decode
    if output is not None:
        logger.info(f"Output from schedule step {output.shape=}")
        logger.info(f"Output from schedule step {output=}")
        next_token_logits = output[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        logger.info(f"First Pass Generation------")
        logger.info(f"{next_token=}")
        logger.info(f"Results = {tokenizer.batch_decode(next_token)}")
    #else:
    #    logger.info(f"Output from schedule step is None {output=}")
    dist.barrier()
    dist.destroy_process_group()
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Model tracing and segmentation")
    parser.add_argument(
        "--model",
        type=str,
        default="8b",
        choices=MODEL_CONFIGS.keys(),
        help="Model size",
    )
    parser.add_argument(
        "--world_size", type=int, default=4, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    main(args.model, args.world_size, args.device)

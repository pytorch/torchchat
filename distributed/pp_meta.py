# $ torchrun --nproc-per-node 4 pp_meta.py

# derived from Ke's PR:
# https://github.com/pytorch/PiPPy/pull/1135


import os
import json
from typing import Optional, Dict, Tuple, Callable, Any, List
import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# TODO- this is only temp import for now
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils import cached_file
from safetensors import safe_open
from argparse import ArgumentParser

from utils import Color
from modeling_utils import reinit_layers, enumerate_transformer_llm, find_main_llama_rope_embeddings, init_on_meta_device

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

# from HF
#from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as hf_rope_init_functions
#from transformers.modeling_rope_utils import _compute_default_rope_parameters, _compute_linear_scaling_rope_parameters


#def rope_emb_init(_device, config):
#    rope_init_fn = _compute_default_rope_parameters # hf_rope_init_functions[self, 'default']
#    inv_freq, _ = rope_init_fn(config, device=_device,) #  **self.rope_kwargs)
#    return inv_freq


#buf_init_callbacks = { "model.rotary_emb.inv_freq": rope_emb_init}

_default_safetensor_file_name = "model.safetensors.index.json"
_config_name = "config.json"
_model_config = None

def create_model(model_id: str, device: str = "cuda", rank: int=0,)-> Tuple[AutoModelForCausalLM, FakeTensorMode, Optional[Dict[str, str]]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    config = None
    #nonlocal _model_config

    #with torch.device("meta"):
    with init_on_meta_device(device="meta"):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if rank==0:
            print(f"---- precision meta init ----")
            print(f"{model.model=}")
            enumerate_transformer_llm(model.model)
            #print(f"rope type {type(model.model.rotary_emb)}")
            #print(f"{model.model.layers.0.rotary_emb.inv_freq[0:5]=}")
            #print(f"{model.model.layers.0.rotary_emb.inv_freq.device=}")

    config = model.config
    #_model_config = config
    assert config is not None, "config is None"
        # what we expect:
        #model.model.rotary_emb=LlamaRotaryEmbedding()
        #model.model.rotary_emb.inv_freq=tensor([1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
        #    2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
        
    print(f"Model type: {type(model)}")

    print(f"buf callback = {model.buf_init_callbacks}")
    #assert False, "check"
    
    #config.device = device
    model.eval()
    
    with fake_mode:
        model.to_empty(device='cuda')
    if rank==0:
            print(f"---- after fake mode to cuda move ----")
            print(f"{model.model.layers[0].self_attn.rotary_emb=}") #model.model.rotary_emb=}")
            print(f"rope type {type(model.model.layers[0].self_attn.rotary_emb)}")
            #print(f"rope type {type(model.model.rotary_emb)}")
            #print(f"{model.model.rotary_emb.inv_freq[0:2]=}")
            #print(f"{model.model.rotary_emb.inv_freq.device=}")
    
    #model.model.rotary_emb.__init__(config=config)
    #model.model.rotary_emb.to('cuda')

    #if rank==0:
    #        print(f"---- after rope re-init and move to device manually ----")
    #        print(f"{model.model.rotary_emb=}")
    #        print(f"rope type {type(model.model.rotary_emb)}")
    #        print(f"{model.model.rotary_emb.inv_freq[0:2]=}")
    #        print(f"-- final result: {model.model.rotary_emb.inv_freq.device=}")



    #if rank==0:
    #        print(f"---- afterto cuda move, then rotary re-init ----")
    #        print(f"{model.model.rotary_emb=}")
    #        print(f"rope type {type(model.model.rotary_emb)}")
    #        print(f"{model.model.rotary_emb.inv_freq[0:2]=}")

            #print(f"check both devices: {model.model.rotary_emb.inv_freq.device=}")

    #dist.barrier()
    #assert False, "check"

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

def init_buffers(
    stage_module: torch.nn.Module,
    device: torch.device,
    init_callbacks: Dict[str, Callable],
    model_config: Optional[Dict[str, str]] = None,
):
    """
    Initialize buffers of `stage_module` per the callback in `init_callbacks`.
    `init_callbacks` is a dictionary from a buffer's FQN to its init function.
    """
    print(f"checking {init_callbacks=}")
    for name, buf in stage_module.named_buffers():
        print(f"****** checking {name=} and {buf=}")
        fire_init: bool = False
        for buffer_name_to_init, func_to_init in init_callbacks.items():
            fire_init = False
            print(f"checking {buffer_name_to_init=} and {func_to_init=}")
            has_pattern, exact_buf_name = remove_pattern_prefix(buffer_name_to_init)
            print(f"checking {has_pattern=} and {exact_buf_name=}")
            if has_pattern:
                if exact_buf_name in name:
                    print(f"checkmate - Found *pattern* match buffer {name} via {exact_buf_name}")
                    print(f"checkmate - looking for {init_callbacks=}")
                    
                    if "inv_freq" in name:
                        fire_init = True
                elif exact_buf_name == name:
                    print(f"checkmate - Found *exact* match buffer {name} via {exact_buf_name}")
                    print(f"checkmate - looking for {init_callbacks=}")
                    fire_init = True

            if not fire_init:
                continue

            print(f"checkmate - Found buffer {name}")
            print(f"checkmate - looking for {init_callbacks=}")
            
            
            print(f"about to call {name} on {device}")
                
            cb = func_to_init
            print(f"checking cb: {cb=}")
            
            buf_val = cb(device,)
            # Find the parent module
            splits = name.split(".")
            mod = stage_module
            for atom in splits[: -1]:
                mod = getattr(mod, atom)
            print(f"checking mod: {mod=}")
            print(f"checking buf_val: {buf_val=}")
            print(f"{splits=}")
            mod.register_buffer(
                splits[-1], buf_val, persistent=False,
            )
            print(f"checking ====>>>> Initialized buffer {name}")

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
        print(
            f"Warning: world size mismatch: {world_size_dist} != {world_size}. "
            "Overriding with dist world size"
        )
        world_size = world_size_dist
    print(f"Rank: {rank} / {world_size}")

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.distributed.init_process_group(rank=rank, world_size=world_size)

    model_id = MODEL_CONFIGS[model_size]
    print(f"Model ID: {model_id}")

    cfile = cached_file(model_id, _default_safetensor_file_name) # model.safetensors.index.json
    config_file = cached_file(model_id, _config_name)  # config.json
    
    
    assert os.path.exists(cfile), f"safetensor index file {cfile} does not exist."
    assert os.path.exists(config_file), f"config file {config_file} does not exist."

    with open(config_file, "r") as file:
        config_file = json.load(file)

    #print(f"Cache file: {cfile} and config file: {config_file}")

    file_location = os.path.dirname(cfile)

    # ========== Create model on meta device =================
    model, fake_mode, model_config = create_model(model_id, device, rank,)
    assert model.buf_init_callbacks is not None, "buffer_init_callbacks is None"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = ("How do you", "I like to")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    fake_ids = fake_mode.from_tensor(inputs["input_ids"])


    weight_map = read_weights_from_json(cfile)
    if weight_map is None:
        print(f"No weight map found in the JSON file {cfile}.")
        return

    # =========== Create pipeline =================
    print("Creating pipeline...")
    pipe = create_pipeline(model, fake_ids, world_size)

    # ---- stage materialization -------
    print("Materializing each stage...")
    stage_module = pipe.get_stage_module(rank)
    
    print(f"Loading weights into stage {rank}")
    load_safetensor_weights(stage_module, weight_map, file_location)
    if rank==0:
        print(f"after load safe tensor Stage module type: {type(stage_module)}")
    
    # TODO
    #if hasattr(model, "buf_init_callbacks"):
    print(f"about to try to init buffers")
    if hasattr(model, "buf_init_callbacks"):
        print(f"..... checkmate  - init buffers with {device=}, ")
        init_buffers(stage_module, device, model.buf_init_callbacks, model_config)
    #stage_module.print_readable()
    #init_buffers(stage_module, "cuda", buf_init_callbacks, model_config)
    print(f"Completed load of stage {rank}")
    
    print(
        f"{Color.blue}\n--->  {rank=} Successfully traced, segmented and loaded weights for model {Color.green}{MODEL_CONFIGS[model_size]}{Color.reset}"
    )
    dist.barrier()
    if rank==0:
        # model.model.rotary_emb.__init__(config=config)
        # stage_module.model.graph.print_tabular()
        print(f"**********     ran init")
        rotary = stage_module.get_submodule('model.layers.0.self_attn.rotary_emb')
        print(f"checkmate {rotary=}")
        #$submodule = stage_module.get_submodule('model.layers.0.self_attn')
        buffer = rotary._buffers['inv_freq']
        print(f"inv freq checkmate {buffer=}")
        rotary1 = stage_module.get_submodule('model.layers.1.self_attn.rotary_emb')
        print(f"checkmate {rotary1=}")
        #$submodule = stage_module.get_submodule('model.layers.0.self_attn')
        buffer1 = rotary1._buffers['inv_freq']
        print(f"inv freq checkmate {buffer1=}")

        
        '''for node in stage_module.graph.nodes:
            if node.name == "model":
                for subnode in node.children():
                    print(f"{subnode=}")
                    print(f"{subnode.target=}")
                    print(f"{subnode.args=}")
                    print(f"{subnode.kwargs=}")
                    print(f"{subnode.meta=}")
                    print(f"{subnode.name=}")
                    print(f"{subnode.op=}")
                    print(f"{subnode.type=}")
                    print(f"{subnode.users=}")
        '''      
           
                
        
    if rank==1:
        print(f"Stage 2: ")
        # model.model.rotary_emb.__init__(config=config)
        #stage_module.model.graph.print_tabular()
    dist.barrier()
    
    # find the rotary embedding
    if rank==0:
        print(f"Finding the main llama rope embeddings...")
        # model.rotary_emb
        print(f"**********     ran init")
        self_attn = stage_module.get_submodule('model.layers.0.self_attn')
        print(f"checkmate {self_attn=}")
        #print(f"checkmate - {stage_module.model.layers.self_attn.rotary_emb.inv_freq[0:4]=}")

        #print(f"{stage_module.model= }")
        
    # Create schedule runtime
    stage = pipe.build_stage(
        rank,
        device=device,
    )

    if rank == 0:
        #print(f"{stage_module.print_readable()=}")
        #print(f"{Color.green}{rank=} {stage_module.model.rotary_emb=} {Color.reset}")
        #print(f"{stage.device=}")
        print(f"{rank=} Completed stage building:  {stage=}...")
        print(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")

        #print(f"{dir(stage_module)=} {dir(stage_module.model)=} {dir(stage_module)}")
    

    
    print(f"TODO = continue here....returning now via early stop for debugging")
    assert False, "stop here"
    return

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

    if rank == 0:
        args = inputs["input_ids"]
    else:
        args = None

    output = schedule.step(args)

    # Decode
    if output is not None:
        next_token_logits = output[0][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        print(tokenizer.batch_decode(next_token))
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Model tracing and segmentation")
    parser.add_argument(
        "--model_size",
        type=str,
        default="123b",
        choices=MODEL_CONFIGS.keys(),
        help="Model size",
    )
    parser.add_argument(
        "--world_size", type=int, default=4, help="Number of gpus to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    main(args.model_size, args.world_size, args.device)

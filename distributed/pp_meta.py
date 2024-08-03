# $ torchrun --nproc-per-node 4 pp_meta.py

# derived from Ke's PR:
# https://github.com/pytorch/PiPPy/pull/1135


import os
import json
from typing import Optional, Dict, Tuple
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

_default_safetensor_file_name = "model.safetensors.index.json"
_config_name = "config.json"


def create_model(model_id: str, device: str = "cuda", rank: int=0,)-> Tuple[AutoModelForCausalLM, FakeTensorMode, Optional[Dict[str, str]]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    config = None
    #with torch.device("meta"):
    with init_on_meta_device(device="meta"):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if rank==0:
            print(f"---- precision meta init ----")
            print(f"{model.model.rotary_emb=}")
            print(f"rope type {type(model.model.rotary_emb)}")
            print(f"{model.model.rotary_emb.inv_freq[0:5]=}")
            print(f"{model.model.rotary_emb.inv_freq.device=}")

        # what we expect:
        #model.model.rotary_emb=LlamaRotaryEmbedding()
        #model.model.rotary_emb.inv_freq=tensor([1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
        #    2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
        
    print(f"Model type: {type(model)}")

    config = model.config
    config.device = device
    model.eval()
    
    with fake_mode:
        model.to_empty(device='cuda')
    if rank==0:
            print(f"---- after fake mode to cuda move ----")
            print(f"{model.model.rotary_emb=}")
            print(f"rope type {type(model.model.rotary_emb)}")
            print(f"{model.model.rotary_emb.inv_freq[0:2]=}")
            print(f"{model.model.rotary_emb.inv_freq.device=}")
    
    model.model.rotary_emb.__init__(config=config)
    model.model.rotary_emb.to('cuda')

    if rank==0:
            print(f"---- after rope re-init and move to device manually ----")
            print(f"{model.model.rotary_emb=}")
            print(f"rope type {type(model.model.rotary_emb)}")
            print(f"{model.model.rotary_emb.inv_freq[0:2]=}")
            print(f"-- final result: {model.model.rotary_emb.inv_freq.device=}")



    if rank==0:
            print(f"---- afterto cuda move, then rotary re-init ----")
            print(f"{model.model.rotary_emb=}")
            print(f"rope type {type(model.model.rotary_emb)}")
            print(f"{model.model.rotary_emb.inv_freq[0:2]=}")

    print(f"check both devices: {model.model.rotary_emb.inv_freq.device=}")

    dist.barrier()

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


def load_safetensor_weights(
    stage_module: torch.nn.Module,
    weight_map: Dict[str, str],
    file_location: str,
):
    stage_state_dict = stage_module.state_dict()
    updated_states = {}

    needed_files = set(
        weight_map.get(param, None)
        for param in stage_state_dict.keys()
        if weight_map.get(param, None)
    )

    for file in needed_files:
        print(f"Loading file: {file}")
        full_path = os.path.join(file_location, file)
        checkpoint = open_hf_safetensor(full_path)
        if checkpoint is None:
            continue

        for param in stage_state_dict.keys():
            if weight_map.get(param) == file:
                if param in checkpoint:
                    stage_state_dict[param] = checkpoint[param]
                    updated_states[param] = None
                else:
                    # print(f"Warning: {param} not found in {file}")
                    # TODO - need to handle this better
                    pass

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
    
    print(f"Completed load of stage {rank}")
    
    print(
        f"{Color.blue}\n--->  {rank=} Successfully traced, segmented and loaded weights for model {Color.green}{MODEL_CONFIGS[model_size]}{Color.reset}"
    )

    if rank==0:
        # model.model.rotary_emb.__init__(config=config)
        stage_module.model.graph.print_tabular()
        
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
           
                
        # print(f"rank 0 {stage_module=}")
        print(f"===== check rope freq ===>>>> {stage_module.model.rotary_emb.inv_freq[0:2]=}")
    dist.barrier()
    assert False, "stop here"
    # find the rotary embedding
    if rank==0:
        print(f"Finding the main llama rope embeddings...")
        # model.rotary_emb
        print(f"**********     ran init")
        print(f"{stage_module.model.rotary_emb=}")

        print(f"{stage_module.model= }")
        
    # Create schedule runtime
    stage = pipe.build_stage(
        rank,
        device=device,
    )

    if rank == 0:
        print(f"{stage_module.print_readable()=}")
        print(f"{Color.green}{rank=} {stage_module.model.rotary_emb=} {Color.reset}")
        print(f"{stage.device=}")
        print(f"{rank=} Completed stage building:  {stage=}...")
        print(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")

        #print(f"{dir(stage_module)=} {dir(stage_module.model)=} {dir(stage_module)}")
    

    
    print(f"TODO = continue here....returning now via early stop for debugging")
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
        default="8b",
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

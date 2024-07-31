"""
Derived from Ke's PR: 
https://github.com/pytorch/PiPPy/pull/1135

This script demonstrates how to create a LLaMA model in "meta" device mode, partition it
into pipeline stages, and materialize each stage module from Hugging Face checkpoints.

Before running the script, please download the required model files from Hugging Face:
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main

How to run this script:
$ python meta_init.py

To download model weights:
$ pip install huggingface_hub[hf_transfer]
$ HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Llama-2-7b-chat-hf -t ./llama7b_weights

Note: This script doesn't use a distributed runtime. It showcases how to load each stage module.
Modify the script to run in a distributed way by distributing the loop at [Note 3].


"""

import os
import sys
sys.path.append("../")
import torch
from torch.distributed.pipelining import pipeline, SplitPoint
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import Color
#from load_weights import load_weight
import safetensors.torch as st
import json

from typing import Optional, Dict

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

def create_model(model_id, device="cuda"):
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    
    with fake_mode:
        model.to_empty(device=device)
    return model, fake_mode

def create_pipeline(model, inputs, world_size):
    layers_per_rank = model.config.num_hidden_layers // world_size
    split_spec = {
        f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }
    
    return pipeline(
        model,
        mb_args=(inputs,),
        mb_kwargs={"output_attentions": False, "output_hidden_states": False, "use_cache": False},
        split_spec=split_spec,
    )

from safetensors import safe_open

def open_hf_safetensor(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.endswith('.safetensors'):
            raise ValueError("File does not have .safetensors extension")
        
        # Open the safetensor file
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # Load all tensors into a dictionary
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        
        return tensors
    
    except Exception as e:
        print(f"An error occurred while opening the safetensor file: {str(e)}")
        return None

# Example usage
# file_path = 'path/to/your/model.safetensors'
# tensors = open_hf_safetensor(file_path)
# if tensors:
#     for key, tensor in tensors.items():
#         print(f"{key}: {tensor.shape}")
def load_safetensor_weights(
    stage_module: torch.nn.Module,
    weight_map: Dict[str, str],
    file_location: str,
):
    """
    Load weights stored as safetensors, from Hugging Face checkpoints into a stage module.

    """
    stage_state_dict = stage_module.state_dict()
    updated_states = dict()

    needed_files = set()
    for param in stage_state_dict.keys():
        file = weight_map.get(param, None)
        if not file:
            print(f"Warning: {param} not found in weight map")
            continue
        needed_files.add(file)
    
    for file in needed_files:
        print(f"stage file {needed_files=}")
        full_path = os.path.join(file_location, file)
        checkpoint = open_hf_safetensor(full_path)
        print(f"Loaded {full_path}")
        #from safetensors.torch import load_model d
        for param in stage_state_dict.keys():
            valid_weight = weight_map.get(param, None)
            if valid_weight == file:
                stage_state_dict[param] = checkpoint[param]
                updated_states.setdefault(param, None)
            else:
                print(f"Warning: {param} not found in {file}")
        
    # Check if the module's state dict will be fully updated from checkpoint
    if stage_state_dict.keys() == updated_states.keys():
        print("Fully updated state dict")
    else:
        print("Partially updated state dict")


    # Now load the weights into the stage module
    # We use `assign=True` because otherwise the properties of the tensors in
    # the current module are preserved.
    stage_module.load_state_dict(stage_state_dict, assign=True)
    print(f"Loaded {len(updated_states)} weights into stage module {stage_module}")


def load_weights(
    stage_module: torch.nn.Module,
    weight_index_file: Optional[str] = "pytorch_model.bin.index.json",
):
    """
    Load weights from Hugging Face checkpoints into a stage module.

    This is a utility for Hugging Face ModelHub checkpoints that comes with an
    index file and multiple binary files.  The index file indicates which
    parameter is saved in which binary. An example can be found at:
    https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main

    Please download the following files in the same directory as this script:
    - pytorch_model.bin.index.json
    - pytorch_model-00001-of-00002.bin
    - pytorch_model-00002-of-00002.bin
    """

    state_dict = stage_module.state_dict()
    updated_states = dict()

    # Get the weight map -- a map from parameter name to file it is saved in
    f = open(weight_index_file)
    js = json.load(f)
    weight_map = js["weight_map"]

    # Figure the set of binary files we'd need to open in order to fill the
    # state dict of the stage module. It will be a subset of all the binary
    # files because the stage module is a partition of the full model.
    needed_files = set()
    for param in state_dict.keys():
        file = weight_map[param]
        needed_files.add(file)

    # Now we load the needed binary files
    
    for file in needed_files:
        checkpoint = torch.load(file, weights_only=True)
        for param in state_dict.keys():
            if weight_map[param] == file:
                state_dict[param] = checkpoint[param]
                updated_states.setdefault(param, None)

    # Check if the module's state dict will be fully updated from checkpoint
    if state_dict.keys() == updated_states.keys():
        print("Fully updated state dict")
    else:
        print("Partially updated state dict")

    # Now load the weights into the stage module
    # We use `assign=True` because otherwise the properties of the tensors in
    # the current module are preserved.
    stage_module.load_state_dict(state_dict, assign=True)

def read_weights_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        if 'weight_map' in data and isinstance(data['weight_map'], dict):
            return data['weight_map']
        else:
            print("No 'weight_map' dictionary found in the JSON file.")
            return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def main():
    # Configuration
    model_size = "8b"
    world_size = 8
    device = "cuda"  # Change to "cuda" if using GPUs

    # Initialize model and tokenizer
    model_id = MODEL_CONFIGS[model_size]
    print(f"{model_id=}")
    model, fake_mode = create_model(model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare input
    prompts = ("How do you", "I like to")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    fake_ids = fake_mode.from_tensor(inputs["input_ids"])

    
    from transformers.utils import cached_file
    from safetensors import safe_open
    cfile = cached_file(model_id, "model.safetensors.index.json")
    print(f"{cfile=}")
    file_location = os.path.dirname(cfile)

    weight_map = read_weights_from_json(cfile)  
    
    
    # Create pipeline
    print(f"creating pipeline...")
    pipe = create_pipeline(model, fake_ids, world_size)


    print(f"Materialize each stage...")

    # Materialize each stage
    for rank in range(world_size):
        stage_module = pipe.get_stage_module(rank)
        print(f"Loading weights into stage {rank}")
        #gpu_stage, missing = st.load_model(model=stage_module, filename = file_location, device=device, strict=False)
        #print(f"{gpu_stage=}")
        # Uncomment the following line when ready to load weights
        load_safetensor_weights(stage_module, weight_map, file_location)
        print(f"completed dummy load of stage {rank}")
        #stage_module.print_readable()
    
    print(f"{Color.blue}\n--->  Successfully traced and segmented model {Color.green}{MODEL_CONFIGS[model_size]}{Color.reset}")

if __name__ == "__main__":
    main()

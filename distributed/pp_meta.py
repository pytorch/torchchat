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
#from load_weights import load_weights

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

def main():
    # Configuration
    model_size = "70b"
    world_size = 8
    device = "cuda"  # Change to "cuda" if using GPUs

    # Initialize model and tokenizer
    model_id = MODEL_CONFIGS[model_size]
    model, fake_mode = create_model(model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare input
    prompts = ("How do you", "I like to")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    fake_ids = fake_mode.from_tensor(inputs["input_ids"])

    # Create pipeline
    pipe = create_pipeline(model, fake_ids, world_size)

    # Materialize each stage
    for rank in range(world_size):
        stage_module = pipe.get_stage_module(rank)
        print(f"Loading weights into stage {rank}")
        # Uncomment the following line when ready to load weights
        # load_weights(stage_module)
        stage_module.print_readable()
    
    print(f"{Color.blue}\n--->  Successfully traced and segmented model {Color.green}{MODEL_CONFIGS[model_size]}{Color.reset}")

if __name__ == "__main__":
    main()

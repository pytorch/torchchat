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

from utils import Color
from modeling_utils import init_on_meta_device, check_rope_embedding, print_model_structure

from torchtune.models.llama3 import llama3_8b, llama3_70b, Llama3Tokenizer
from torchtune.models.llama3_1 import llama3_1_405b
from hf_utils import get_hf_tokenizer, load_safetensor_weights, read_weights_from_json, get_hf_weight_map_and_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model configuration

MODEL_CONFIGS = {
    "hf_7b": "meta-llama/Llama-2-7b-chat-hf",
    "hf_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "hf_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "hf_405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "hf_405base": "meta-llama/Meta-Llama-3.1-405B",
    "hf_123b": "mistralai/Mistral-Large-Instruct-2407",
    "hf_22b": "mistralai/Codestral-22B-v0.1",
}

TUNE_MODEL_CONFIGS = {
    "8b": (llama3_8b, 'meta-llama/Meta-Llama-3-8B-Instruct'),  # '/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model'),
    "70b": (llama3_70b, '/tmp/Meta-Llama-3-70B-Instruct/original/tokenizer.model'),
    "405b": (llama3_1_405b, '/tmp/Meta-Llama-3.1-405B-Instruct/original/mp16/tokenizer.model'),
}

def create_model(model_id: str, device: str = "cuda", rank: int = 0) -> Tuple[Any, FakeTensorMode, Any]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    
    model_func, hf_path = TUNE_MODEL_CONFIGS[model_id]
    print(f"{model_func=}, {hf_path=}")

    assert model_func is not None, f"Model {model_id} not found in TUNE_MODEL_CONFIGS"
    assert hf_path is not None, f"hf path for {model_id} not found in TUNE_MODEL_CONFIGS"

    with init_on_meta_device(device="meta"):
        print(f"about to init model {model_id}")
        model = model_func().to(torch.bfloat16)
    model.eval()
    if rank==0:
        #print(model.config)
        print(f"{model=}")
        #print(f"{model.model.embed_tokens.dtype=}")
        # print(f"{model.layers[0].attn.pos_embeddings.weight.dtype=}")
        print(f"{model.layers[0].attn.pos_embeddings=}")
        print(f"{model.layers[0].attn.q_proj=}") 
        print(f"{model.layers[0].attn.q_proj.weight.dtype=}")
        
    
    logger.info(f"Model type: {type(model)}")
    #logger.info(f"Buffer callback: {model.buf_init_callbacks}")
    #if not model.buf_init_callbacks:
    #    logger.warning("ROPE generation may not succeed - buf_init_callbacks is None")
    
    with fake_mode:
        model.to_empty(device='cuda')
    
    # create tokenizer
    #tokenizer = Llama3Tokenizer(tokenizer_path)
    tokenizer = get_hf_tokenizer(hf_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, fake_mode, tokenizer, hf_path

def create_pipeline(model, inputs, world_size: int):
    layers_per_rank = model.config.num_hidden_layers // world_size
    split_spec = {
        f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }

    return pipeline(
        model,
        mb_args=(inputs,),
        mb_kwargs=None, # {
        #    "output_attentions": False,
        #    "output_hidden_states": False,
        #    "use_cache": False,
        #},
        split_spec=split_spec,
    )

# --- update init ----

def main(model_id: str, world_size: int, device: str):
    rank = int(os.environ["RANK"])
    world_size_dist = int(os.environ["WORLD_SIZE"])
    if world_size_dist != world_size:
        logger.warning(f"World size mismatch: {world_size_dist} != {world_size}. Overriding with dist world size")
        world_size = world_size_dist
    logger.info(f"Rank: {rank} / {world_size}")

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    dist.init_process_group(rank=rank, world_size=world_size)

    # Create model on meta device
    model, fake_mode, tokenizer, hf_path = create_model(model_id, device, rank)
    
    print(f"{tokenizer.pad_token=}")

    prompts = ("How do you", "I like to")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    fake_ids = fake_mode.from_tensor(inputs["input_ids"])

    weight_map, weight_path = get_hf_weight_map_and_path(hf_path)
    logger.info(f"Weight map: {weight_map=}")
    logger.info(f"Weight path: {weight_path=}")


    dist.barrier()
    assert False, "check paths"

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

    #else:
    #    logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")
    
    logger.info(f"{Color.blue}\n--->  {rank=} Successfully traced, segmented and loaded weights for model {Color.green}{MODEL_CONFIGS[model_size]}{Color.reset}")

    # Create schedule runtime
    stage = pipe.build_stage(rank, device=device)
    #if rank == 0:
    #    logger.info(f"{rank=} Completed stage building:  {stage=}...")
        #logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")

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

    if rank == 0:
        output = schedule.step(inputs['input_ids'])
    else:
        output = schedule.step()
    

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
        default="405b",
        choices=TUNE_MODEL_CONFIGS.keys(),
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

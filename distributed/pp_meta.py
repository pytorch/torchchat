# $ torchrun --nproc-per-node 4 pp_meta.py

import os
from typing import Tuple, Any
import logging
from argparse import ArgumentParser
import time
import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from torch._subclasses.fake_tensor import FakeTensorMode

from utils import Color
from modeling_utils import (
    init_on_meta_device,
    verify_graph_tensor_properties,
    enumerate_transformer_llm,
    inspect_module_tensors,
    torch_in_fake_mode,
)

from torchtune.models.llama3 import llama3_8b, llama3_70b
from torchtune.models.llama3_1 import llama3_1_405b
from hf_utils import (
    get_hf_tokenizer,
    load_safetensor_weights,
    get_hf_weight_map_and_path,
    get_hf_path_from_model_id,
)
from safetensor_utils import (analyze_safetensor_file, analyze_safetensor_directory, summarize_results)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    "8b": (
        llama3_8b,
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),  # '/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model'),
    "70b": (llama3_70b, "meta-llama/Meta-Llama-3-70B-Instruct"),
    "405b": (
        llama3_1_405b,
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
    ),
}


def create_model(
    model_id: str, device: str = "cuda", rank: int = 0
) -> Tuple[Any, FakeTensorMode, Any]:

    logger.info(f"Torch is currently in fake mode: {torch_in_fake_mode()=}")
    
    
    model_func, hf_model_id = TUNE_MODEL_CONFIGS[model_id]
    logger.info(f"{model_func=}, {hf_model_id=}")
    hf_path = get_hf_path_from_model_id(hf_model_id)
    logger.info(f"hf path: {hf_path}")
    
    assert model_func is not None, f"Model {model_id} not found in TUNE_MODEL_CONFIGS"
    assert (
        hf_path is not None
    ), f"hf path for {model_id} not found in TUNE_MODEL_CONFIGS"

    with init_on_meta_device(device="meta"):
        logger.info(f"about to init model on meta device, {model_id=}")
        model = model_func()

    model.eval()
    
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        model.to_empty(device="cuda")
        logger.info(f"Torch in fake mode: {torch_in_fake_mode()=}")
    
    logger.info(f"exited context - Torch in fake mode: {torch_in_fake_mode()=}")

    # create tokenizer
    # tokenizer = Llama3Tokenizer(tokenizer_path)
    tokenizer = get_hf_tokenizer(hf_path)
    tokenizer.pad_token = tokenizer.eos_token

    return model, fake_mode, tokenizer, hf_path


def create_pipeline(model, inputs, world_size: int):
    # Split model into world_size stages

    layers_per_rank = len(model.layers) // world_size

    logger.info(
        f"Splitting model into {world_size} stages with {layers_per_rank} layers each"
    )

    split_spec = {
        f"layers.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }

    return pipeline(
        model,
        mb_args=(inputs,),
        mb_kwargs=None,  # {
        #    "output_attentions": False,
        #    "output_hidden_states": False,
        #    "use_cache": False,
        # },
        split_spec=split_spec,
    )


# --- update init ----


def main(model_id: str, world_size: int, device: str):
    rank = int(os.environ["RANK"])
    world_size_dist = int(os.environ["WORLD_SIZE"])
    if world_size_dist != world_size:
        logger.warning(
            f"World size mismatch: {world_size_dist} != {world_size}. Overriding with dist world size"
        )
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

    
    # logger.info(f"Weight map: {weight_map=}")
    # logger.info(f"Weight path: {weight_path=}")

    # Create pipeline
    logger.info("Creating pipeline...")
    pipe = create_pipeline(model, fake_ids, world_size)
    logger.info(f"Pipeline created: {pipe=}")

    # Stage materialization
    logger.info("Materializing each stage...")
    stage_module = pipe.get_stage_module(rank)
    logger.info(f"Stage module type: {type(stage_module)}")

    logger.info(f"Loading weights into stage {rank}")
    weight_map, weight_path, new_to_old_keymap = get_hf_weight_map_and_path(hf_path)
    total_weight_count, missing_weight_count = load_safetensor_weights(
        stage_module, weight_map, weight_path, new_to_old_keymap, device
    )

    logger.info(
        f"Loaded {total_weight_count} weights into stage {rank} with {missing_weight_count} missing"
    )
    assert (
        missing_weight_count == 0
    ), f"Missing {missing_weight_count} weights in stage {rank}"

    if rank == 0:
        logger.info(f"After load safe tensor Stage module type: {type(stage_module)}")

    dist.barrier()  # wait for all ranks to finish loading weights
    """logger.info("About to try to init buffers")
    if hasattr(model, "buf_init_callbacks"):
        logger.info(f"Initializing buffers with device={device}")
        init_buffers(stage_module, device, model.buf_init_callbacks, model_config)
    logger.info(f"Completed load of stage {rank}")
    if rank == 0:
        #logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")
        logger.info(f"{Color.blue}{stage_module.model.rotary_emb=}{Color.reset}")
        logger.info(f"{Color.blue}{stage_module.model.rotary_emb.inv_freq.dtype=}{Color.reset}")
    """
    # else:
    #    logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")

    logger.info(
        f"{Color.green}\n--->  {Color.yellow}{rank=} {Color.blue}Successfully traced, segmented and loaded weights for model {Color.green}{model_id}{Color.reset}"
    )

    # Verify graph dtypes
    proper_graph, error_list = verify_graph_tensor_properties(stage_module)
    if not proper_graph:
        logger.error(
            f"Graph dtypes are not correct for stage {rank}. Errors: {error_list}"
        )
        # assert False, f"Graph dtypes are not correct for stage {rank}. Errors: {error_list}"
    logger.info(f"{proper_graph=}, {error_list=}")
    dist.barrier()
    time.sleep(5)
    # Create schedule runtime
    stage = pipe.build_stage(rank, device=device)
    # if rank == 0:
    #    logger.info(f"{rank=} Completed stage building:  {stage=}...")
    # logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")

    logger.info("Pipeline Complete ---- Running schedule...")

    logger.info(f"{rank=} Completed stage building:  {stage=}...")
    # logger.info(f"{Color.blue}{type(stage_module)=} {dir(stage_module)=}{Color.reset}")
    # logger.info(f"{Color.blue}{stage_module.print_readable()=}{Color.reset}")
    enumerate_transformer_llm(stage_module)

    tensor_info = inspect_module_tensors(stage_module)
    logger.info(f"{rank=} {tensor_info=}")
    time.sleep(5)
    # Run
    # Run time inputs
    full_batch_prompts = (
        "How do you",
        "I like to",
        "Can I help",
        "You need to",
        "The weather is",
        "I found a",
        "What is your",
        "You are so",
    )  # full batch size = 8

    inputs = tokenizer(full_batch_prompts, return_tensors="pt", padding=True).to(device)
    #logger.info(f"check {inputs=}")

    # Attach to a schedule
    # number of microbatches = 8 // 2 = 4
    num_mbs = 4
    schedule = ScheduleGPipe(stage, num_mbs)

    if rank == 0:
        output = schedule.step(inputs["input_ids"])
    else:
        output = schedule.step()

    # Decode
    if output is not None:
        logger.info(f"Output from schedule step {output.shape=}")
        logger.info(f"Output from schedule step {output=}")
        next_token_logits = output[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        logger.info("First Pass Generation------")
        logger.info(f"{next_token=}")
        logger.info(f"Results = {tokenizer.batch_decode(next_token)}")
    # else:
    #    logger.info(f"Output from schedule step is None {output=}")
    dist.barrier()
    dist.destroy_process_group()



def verify_safetensor_weights(directory_path: str):
    logger.info(f"Verifying safetensor weights for {directory_path}")
    all_results = analyze_safetensor_directory(directory_path)
    summary = summarize_results(all_results)

    print("Summary of all safetensor files in the directory:")
    print("\nDtype distribution:")
    for dtype, count in summary['dtypes'].items():
        print(f"  {dtype}: {count}")

    print("\nTensor type distribution:")
    for tensor_type, count in summary['tensor_types'].items():
        print(f"  {tensor_type}: {count}")

    print("\nDetailed results for each file:")
    for filename, file_results in all_results.items():
        print(f"\nFile: {filename}")
        for tensor_name, (dtype, tensor_type) in file_results.items():
            print(f"  Tensor: {tensor_name}")
            print(f"    dtype: {dtype}")
            print(f"    type: {tensor_type}")

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

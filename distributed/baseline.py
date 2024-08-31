# $ torchrun --nproc-per-node 4 pippy_llama.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe

# Grab the model
llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", #low_cpu_mem_usage=True
)
print(llama)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
mb_prompts = (
    "How do you", "I like to",
)  # microbatch size = 2

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
torch.distributed.init_process_group(rank=rank, world_size=world_size)

llama.to(device).eval()

# Cut model by equal number of layers per rank
layers_per_rank = llama.config.num_hidden_layers // world_size
print(f"layers_per_rank = {layers_per_rank}")
split_spec = {
    f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
    for i in range(1, world_size)
}

# Create a pipeline representation from the model
mb_inputs = tokenizer(mb_prompts, return_tensors="pt", padding=True).to(device)
pipe = pipeline(llama, mb_args=(mb_inputs["input_ids"]), split_spec=split_spec)

# Create pipeline stage for each rank
stage = pipe.build_stage(rank, device=device)

# Run time inputs
full_batch_prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    #"The weather is", "I found a", "What is your", "You are so",
)  # full batch size = 8
inputs = tokenizer(full_batch_prompts, return_tensors="pt", padding=True).to(device)

# Attach to a schedule
# number of microbatches = 8 // 2 = 4
num_mbs = 4
schedule = ScheduleGPipe(stage, num_mbs)

# Run
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

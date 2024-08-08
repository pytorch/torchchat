
import torch 
from torchtune.models.llama3 import llama3_8b

model = llama3_8b()
model.eval()
print(f"{model.layers[0].attn.q_proj.weight.dtype=}")

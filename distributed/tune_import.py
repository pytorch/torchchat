
import torch 
from torchtune.models.llama3 import llama3_8b, llama3_70b

model = llama3_70b()
model.eval()
print(f"{model.layers[0].attn.pos_embeddings.weight.dtype=}")
print(f"{model=}")


import torch 
from torchtune.models.llama3 import llama3_8b, llama3_70b

model = llama3_8b()
model.eval()
print(f"{model.layers[0].attn.pos_embeddings.weight.dtype=}")
print(f"{model=}")



weight_map, weight_path, new_to_old_keymap = get_hf_weight_map_and_path(hf_path)

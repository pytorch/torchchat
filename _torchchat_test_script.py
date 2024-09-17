
import torch
import sys
import os

from torchtune import training
from torchtune.models.flamingo import flamingo_decoder, flamingo_vision_encoder, FlamingoTransform
from torchtune.modules.model_fusion import DeepFusionModel

from torchchat.model import Model

import re

from typing import Dict
from torchtune.generation._generation import sample
from torchtune.training import set_default_dtype
import numpy as np
import PIL

from torchtune.data import Message

def flamingo_transform(tokenizer_path):
    return FlamingoTransform(
        tokenizer_path,
        tile_size=448,
        patch_size=14,
        max_num_tiles=4,
        max_seq_len=8192,
        encoder_max_seq_len=4100,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        prompt_template=None,
    )

def padded_collate(batch, device='cuda', dtype=torch.bfloat16, padding_idx=0):
    # Placeholder Collator until https://github.com/pytorch/torchtune/pull/1156 lands
    assert len(batch) == 1, "Test collate function only supports bs = 1"
    sample = batch[0]
    sample["tokens"] = torch.Tensor(sample["tokens"])[None, ...].to(device).long()
    sample["mask"] = torch.Tensor(sample["mask"])[None, ...].to(device).bool()
    sample["encoder_input"]["images"] = torch.stack(sample["encoder_input"]["images"])[None, ...].to(device)
    sample["encoder_input"]["aspect_ratio"] = torch.stack(sample["encoder_input"]["aspect_ratio"])[None, ...].to(device)
    assert len(sample["encoder_mask"]), "Test collate function only supports 1 image per sequence"
    # Pad encoder mask to max_num_tiles sequence length (4100)
    s_x, s_y = sample["encoder_mask"][0].shape
    mask_padding = torch.zeros((s_x, 4100 - s_y), dtype=torch.bool)
    encoder_mask = torch.cat([sample["encoder_mask"][0], mask_padding], dim=1)
    sample["encoder_mask"] = encoder_mask[None, ...].to(device)
    return sample



_FROM_META = {
    "text_model.tok_embeddings.weight": "decoder.tok_embeddings.weight",
    "text_model.learnable_embedding.weight": "decoder.tok_embeddings.fusion_embedding.weight",
    "text_model.norm.weight": "decoder.norm.scale",
    "text_model.output.weight": "decoder.output.weight",

    "text_model.layers.{}.attention_norm.weight": "decoder.layers.{}.sa_norm.scale",
    "text_model.layers.{}.attention.wq.weight": "decoder.layers.{}.attn.q_proj.weight",
    "text_model.layers.{}.attention.wk.weight": "decoder.layers.{}.attn.k_proj.weight",
    "text_model.layers.{}.attention.wv.weight": "decoder.layers.{}.attn.v_proj.weight",
    "text_model.layers.{}.attention.wo.weight": "decoder.layers.{}.attn.output_proj.weight",
    "text_model.layers.{}.ffn_norm.weight": "decoder.layers.{}.mlp_norm.scale",
    "text_model.layers.{}.feed_forward.w1.weight": "decoder.layers.{}.mlp.w1.weight",
    "text_model.layers.{}.feed_forward.w3.weight": "decoder.layers.{}.mlp.w3.weight",
    "text_model.layers.{}.feed_forward.w2.weight": "decoder.layers.{}.mlp.w2.weight",

    "text_model.cross_attention_layers.{}.gate_attn": "decoder.layers.{}.fusion_layer.ca_scale.scale",
    "text_model.cross_attention_layers.{}.gate_ffwd": "decoder.layers.{}.fusion_layer.mlp_scale.scale",
    "text_model.cross_attention_layers.{}.attention_norm.weight": "decoder.layers.{}.fusion_layer.ca_norm.scale",
    "text_model.cross_attention_layers.{}.ffn_norm.weight": "decoder.layers.{}.fusion_layer.mlp_norm.scale",
    "text_model.cross_attention_layers.{}.attention.wq.weight": "decoder.layers.{}.fusion_layer.attn.q_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wk.weight": "decoder.layers.{}.fusion_layer.attn.k_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wv.weight": "decoder.layers.{}.fusion_layer.attn.v_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wo.weight": "decoder.layers.{}.fusion_layer.attn.output_proj.weight",
    "text_model.cross_attention_layers.{}.attention.inner_attention.q_norm.weight": "decoder.layers.{}.fusion_layer.attn.q_norm.scale",
    "text_model.cross_attention_layers.{}.attention.inner_attention.k_norm.weight": "decoder.layers.{}.fusion_layer.attn.k_norm.scale",
    "text_model.cross_attention_layers.{}.feed_forward.w1.weight": "decoder.layers.{}.fusion_layer.mlp.w1.weight",
    "text_model.cross_attention_layers.{}.feed_forward.w3.weight": "decoder.layers.{}.fusion_layer.mlp.w3.weight",
    "text_model.cross_attention_layers.{}.feed_forward.w2.weight": "decoder.layers.{}.fusion_layer.mlp.w2.weight",

    "vision_model.vision_encoder.positional_embedding": "encoder.clip.token_pos_embedding.local_token_positional_embedding",
    "vision_model.vision_encoder.gated_positional_embedding": "encoder.clip.token_pos_embedding.global_token_positional_embedding",
    "vision_model.vision_encoder.gated_positional_embedding_gate": "encoder.clip.token_pos_embedding.gate",
    "vision_model.vision_encoder.ln_pre.weight": "encoder.clip.ln_pre.weight",
    "vision_model.vision_encoder.ln_pre.bias": "encoder.clip.ln_pre.bias",
    "vision_model.vision_encoder.ln_post.weight": "encoder.clip.ln_post.weight",
    "vision_model.vision_encoder.ln_post.bias": "encoder.clip.ln_post.bias",
    "vision_model.vision_encoder.pre_tile_pos_embed.embedding": "encoder.clip.pre_tile_pos_embed.embedding",
    "vision_model.vision_encoder.pre_tile_pos_embed.gate": "encoder.clip.pre_tile_pos_embed.gate",
    "vision_model.vision_encoder.post_tile_pos_embed.embedding": "encoder.clip.post_tile_pos_embed.embedding",
    "vision_model.vision_encoder.post_tile_pos_embed.gate": "encoder.clip.post_tile_pos_embed.gate",
    "vision_model.vision_encoder.class_embedding" : "encoder.clip.cls_token_embedding.weight",
    "vision_model.vision_encoder.conv1._linear.weight" : "encoder.clip.conv.weight",
    
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wq.weight": "encoder.clip.layers.{}.attn.q_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wk.weight": "encoder.clip.layers.{}.attn.k_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wv.weight": "encoder.clip.layers.{}.attn.v_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wo.weight": "encoder.clip.layers.{}.attn.output_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_fc.weight": "encoder.clip.layers.{}.mlp.w1.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_fc.bias": "encoder.clip.layers.{}.mlp.w1.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_proj.weight": "encoder.clip.layers.{}.mlp.w2.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_proj.bias": "encoder.clip.layers.{}.mlp.w2.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_1.weight": "encoder.clip.layers.{}.sa_norm.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_1.bias": "encoder.clip.layers.{}.sa_norm.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_2.weight": "encoder.clip.layers.{}.mlp_norm.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_2.bias": "encoder.clip.layers.{}.mlp_norm.bias",

    "vision_model.vision_projection.weight" : "encoder.projection.output.weight",
    "vision_model.vision_projection.bias" : "encoder.projection.output.bias",

    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wq.weight": "encoder.projection.layers.{}.attn.q_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wk.weight": "encoder.projection.layers.{}.attn.k_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wv.weight": "encoder.projection.layers.{}.attn.v_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wo.weight": "encoder.projection.layers.{}.attn.output_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.weight": "encoder.projection.layers.{}.mlp.w1.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.bias": "encoder.projection.layers.{}.mlp.w1.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.weight": "encoder.projection.layers.{}.mlp.w2.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.bias": "encoder.projection.layers.{}.mlp.w2.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_1.weight": "encoder.projection.layers.{}.sa_norm.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_1.bias": "encoder.projection.layers.{}.sa_norm.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_2.weight": "encoder.projection.layers.{}.mlp_norm.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_2.bias": "encoder.projection.layers.{}.mlp_norm.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.gate_attn": "encoder.projection.layers.{}.sa_scale.scale",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.gate_ffn": "encoder.projection.layers.{}.mlp_scale.scale",
}


def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def flamingo_meta_to_tune(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convertor from Meta state dict to torchtune state dict. This handles:
    - Updateing the cross attention layer numbers
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        if key == "text_model.rope.freqs":
            continue
        new_key = get_mapped_key(key, _FROM_META)
        if "cross_attention_layers" in key:
            layer = int(key.split(".")[2])
            # TODO: grab num_layers and generalize this
            new_layer = (layer + 1) * 4 - 1
            key_lst = new_key.split(".")
            key_lst[2] = str(new_layer)
            new_key = ".".join(key_lst)
            if "gate_ffwd" in key or "gate_attn" in key:
                value = value[:1]
        elif "conv1" in key:
            # TODO: get patch size and generalize
            value = value.reshape(-1, 3, 14, 14)
        converted_state_dict[new_key] = value
    return converted_state_dict



if __name__ == "__main__":
    llava3_2_dir = str(sys.argv[1])
    param_path = os.path.join(llava3_2_dir, "flamingo.json")
    tokenizer_path = os.path.join(llava3_2_dir, "tokenizer.model")
    checkpoint_path = os.path.join(llava3_2_dir, "consolidated.pth")
    image_path = os.path.join(llava3_2_dir, "dog.jpg")

    if len(sys.argv) > 2:
        device = torch.device(str(sys.argv[2]))
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Loading model from {param_path}")

    dtype = torch.bfloat16
    with set_default_dtype(dtype), device:
        model = Model.from_params(param_path)

    transform = flamingo_transform(tokenizer_path)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)
    print("Converting state dict into flamingo format")
    state_dict = flamingo_meta_to_tune(state_dict)
    print("Loading state dict into model")
    model.model.load_state_dict(state_dict)

    model = torch.compile(model)
    images = [PIL.Image.open(image_path)]

    dialog = [
        Message(
            role="user",
            content=[
                {"type": "image"},
                {"type": "text", "content": "What's in this image?"},
            ],
            eot=True,
        ),
        Message(role="assistant", content="")
    ]

    data = transform({"images": images, "messages": dialog}, inference=True)

    model.eval()
    with device:
        model.setup_caches(1, dtype=torch.bfloat16)


    max_generated_tokens = 100
    temperature = .6
    top_k = 500

    print("Generating...")

    generated_tokens = []
    model.reset_caches()
    with torch.no_grad():
        batch = padded_collate([data], device, dtype)
        batch.pop("mask")

        logits = model(**batch)[:, -1]
        tok = sample(logits, temperature, top_k)
        generated_tokens.append(tok.item())

        cache_mask = batch["encoder_mask"][:, -1:]
        for _ in range(max_generated_tokens):
            if tok.item() in transform.stop_tokens:
                break
            logits = model(tok, encoder_mask=cache_mask)[:, -1]
            tok = sample(logits, temperature, top_k)
            generated_tokens.append(tok.item())

    print(transform.decode(generated_tokens))



""":md
## Chat Pseudo Code

This approach guarantees that there's only one image cached at a time so that there's no need for cross attention masking.
This works because Llama3v is trained such that each token is only allowed to attend to the previous image and the rest are 
masked during training/finetuning. Since consecutive images are treated as one image for Llama3v, you can control the maximum
encoder sequence length by setting max_consecuitve here, as well as by settin max_num_tiles and max_resolution for the image input.

```python
model.eval()
model.setup_caches(1, torch.bf16)

with torch.no_grad():
    # Prefill system prompt
    toks, _ = transform(parse_prompt(system_prompt))
    model(toks) 
    while True:
        # Prefill user prompt split over images
        user_prompt = input(">>> ")
        toks, imgs = transform(parse_prompt(user_prompt))
        for i, tok in enumerate(split(toks, image_token, max_consecutive=1)):
            img = None
            if imgs is not None:
                img = imgs[i]
                reset_attn_cache(model)
            logits = model(tok, img)

        # Decode assitant response
        tok = sample_tok(logits) # only ouptput single token logits when model.cache_enabled=True
    	while tok != EOS:
    		logits = model(tok) 
    		tok = sample_tok(logits)
    		sys.stdout.buffer.write(transform.decode(tok))
```
"""

""":py"""

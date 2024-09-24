# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import re
import sys
import glob
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import safetensors.torch
import shutil

from torchchat.model import TransformerArgs

# support running without installing as a package
wd = Path(__file__).parent.parent
sys.path.append(str(wd.resolve()))
sys.path.append(str((wd / "build").resolve()))

from torchchat.model import ModelArgs

def remap_llava_checkpoint(llava_ckpt):
    def _translate_state_dict_for_vision_model(hf_state_dict) -> Dict[str, Any]:
        translated_state_dict = {}
        hf_weight_prefix = "vision_model."
        name_mapping = {
            f"{hf_weight_prefix}embeddings.class_embedding": "encoder.cls_token_embedding.weight",
            f"{hf_weight_prefix}embeddings.position_embedding.weight": "encoder.token_pos_embedding.positional_embedding",
            f"{hf_weight_prefix}embeddings.patch_embedding.weight": "encoder.conv.weight",
            f"{hf_weight_prefix}pre_layrnorm.weight": "encoder.ln_pre.weight",
            f"{hf_weight_prefix}pre_layrnorm.bias": "encoder.ln_pre.bias",
            f"{hf_weight_prefix}post_layernorm.weight": "encoder.ln_post.weight",
            f"{hf_weight_prefix}post_layernorm.bias": "encoder.ln_post.bias",
        }
        patterns = [
            (
                rf"{hf_weight_prefix}encoder\.layers\.([0-9]+)\.self_attn\.(k|q|v)_proj\.(weight|bias)",
                lambda match: f"encoder.layers.{match.group(1)}.attn.{match.group(2)}_proj.{match.group(3)}",
            ),
            (
                rf"{hf_weight_prefix}encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.(weight|bias)",
                lambda match: f"encoder.layers.{match.group(1)}.attn.output_proj.{match.group(2)}",
            ),
            (
                rf"{hf_weight_prefix}encoder\.layers\.([0-9]+)\.mlp\.fc(1|2)\.(weight|bias)",
                lambda match: f"encoder.layers.{match.group(1)}.mlp.w{match.group(2)}.{match.group(3)}",
            ),
            (
                rf"{hf_weight_prefix}encoder\.layers\.([0-9]+)\.layer_norm1\.(weight|bias)",
                lambda match: f"encoder.layers.{match.group(1)}.sa_norm.{match.group(2)}",
            ),
            (
                rf"{hf_weight_prefix}encoder\.layers\.([0-9]+)\.layer_norm2\.(weight|bias)",
                lambda match: f"encoder.layers.{match.group(1)}.mlp_norm.{match.group(2)}",
            ),
        ]
        for pattern, replacement in patterns:
            for key in list(hf_state_dict.keys()):
                if re.match(pattern, key):
                    new_key = re.sub(pattern, replacement, key)
                    name_mapping[key] = new_key
        temp_state_dict = {}
        for k, v in hf_state_dict.items():
            new_k = name_mapping.get(k, k)
            if "in_proj_weight" in new_k or "in_proj_bias" in new_k:
                if new_k not in temp_state_dict:
                    temp_state_dict[new_k] = {"q": None, "k": None, "v": None}
                if "q_proj" in k:
                    temp_state_dict[new_k]["q"] = v
                elif "k_proj" in k:
                    temp_state_dict[new_k]["k"] = v
                elif "v_proj" in k:
                    temp_state_dict[new_k]["v"] = v
            else:
                temp_state_dict[new_k] = v
        for k, v in temp_state_dict.items():
            if isinstance(v, dict):
                translated_state_dict[k] = torch.cat([v["q"], v["k"], v["v"]], dim=0)
            else:
                translated_state_dict[k] = v
        return translated_state_dict

    def _translate_state_dict_for_text_model(hf_state_dict) -> Dict[str, Any]:
        key_map = {
            r"model.layers.([0-9]+).self_attn.q_proj.": r"decoder.layers.\1.attention.wq.",
            r"model.layers.([0-9]+).self_attn.k_proj.": r"decoder.layers.\1.attention.wk.",
            r"model.layers.([0-9]+).self_attn.v_proj.": r"decoder.layers.\1.attention.wv.",
            r"model.layers.([0-9]+).self_attn.o_proj.": r"decoder.layers.\1.attention.wo.",
            r"model.layers.([0-9]+).input_layernorm.": r"decoder.layers.\1.attention_norm.",
            r"model.layers.([0-9]+).mlp.gate_proj.": r"decoder.layers.\1.feed_forward.w1.",
            r"model.layers.([0-9]+).mlp.down_proj.": r"decoder.layers.\1.feed_forward.w2.",
            r"model.layers.([0-9]+).mlp.up_proj.": r"decoder.layers.\1.feed_forward.w3.",
            r"model.layers.([0-9]+).post_attention_layernorm.": r"decoder.layers.\1.ffn_norm.",
            r"model.norm.": r"decoder.norm.",
            # r"model.embed_tokens.": r"tok_embeddings.", # load separately
            r"lm_head.": r"decoder.output.",
        }
        new_state_dict = {}
        def get_new_key(old_key: str) -> str:
            for old_pattern, replacement in key_map.items():
                if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                    return new_key
            return old_key
        for old_key in hf_state_dict.keys():
            new_key = get_new_key(old_key)
            new_state_dict[new_key] = hf_state_dict[old_key]
        return new_state_dict
    
    def _translate_state_dict_for_mm_projector_model(hf_state_dict) -> Dict[str, Any]:
        new_state_dict = {}
        for old_key in hf_state_dict.keys():
            new_key = "mm_projector." + old_key
            new_state_dict[new_key] = hf_state_dict[old_key]
        return new_state_dict
    
    def split_checkpoint(llava_ckpt):
        language_model_ckpt = {}
        multi_modal_ckpt = {}
        vision_tower_ckpt = {}
        for key, value in llava_ckpt.items():
            if key.startswith("language_model"):
                language_model_ckpt[key[len("language_model") + 1:]] = value
            elif key.startswith("multi_modal_projector"):
                multi_modal_ckpt[key[len("multi_modal_projector") + 1:]] = value
            elif key.startswith("vision_tower"):
                vision_tower_ckpt[key[len("vision_tower") + 1:]] = value
        return language_model_ckpt, multi_modal_ckpt, vision_tower_ckpt
    language_model_ckpt, multi_modal_ckpt, vision_tower_ckpt = split_checkpoint(llava_ckpt)
    remapped_state_dict = {
        "tok_embeddings.weight": language_model_ckpt.pop("model.embed_tokens.weight"),
    }
    remapped_state_dict.update(_translate_state_dict_for_text_model(language_model_ckpt))
    remapped_state_dict.update(_translate_state_dict_for_vision_model(vision_tower_ckpt))
    remapped_state_dict.update(_translate_state_dict_for_mm_projector_model(multi_modal_ckpt))
    return remapped_state_dict

    
@torch.inference_mode
def convert_llava_checkpoint(    
    *,
    model_dir: Optional[Path] = None,
) -> None:
    
    """
    Process safetensor files from a specific directory structure and save the remapped model.
    
    Args:
        model_dir (str): Base directory containing the model subdirectories.
    """

    def _get_llava_files_with_pattern(pattern):
        pattern = os.path.join(model_dir, f"models--llava-hf--llava-1.5-7b-hf/snapshots/*/{pattern}")
        return glob.glob(pattern)

    # get all safetensor files in the model directory
    safetensor_files = _get_llava_files_with_pattern("*.safetensors")
    
    if not safetensor_files:
        raise ValueError("No safetensor files found.")
    
    merged_weights = {}
    
    # Merge safetensor files into a whole
    for file in safetensor_files:
        # Load weights from the current file
        part_weights = safetensors.torch.load_file(file)
        
        # Iterate over each weight in the current file
        for key, value in part_weights.items():
            if key in merged_weights:
                # If the key already exists, concatenate tensors
                merged_weights[key] = torch.cat((merged_weights[key], value), dim=0)
            else:
                # If the key does not exist, add it to the dictionary
                merged_weights[key] = value
    
    # Remap the checkpoint and save it as pth
    remapped_weights = remap_llava_checkpoint(merged_weights)
    model_path = model_dir / "model.pth"
    torch.save(remapped_weights, model_path)

    # copy tokenizer
    tokenizer_files = _get_llava_files_with_pattern("tokenizer.model")
    assert len(tokenizer_files) == 1, "Should get only one tokenizer file, but got {}".format(tokenizer_files)

    tokenizer_path = model_dir / "tokenizer.model"
    shutil.copy(tokenizer_files[0], tokenizer_path)


@torch.inference_mode()
def convert_text_only_hf_checkpoint(
    *,
    model_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    remove_bin_files: bool = False,
) -> None:
    if model_dir is None:
        model_dir = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf")
    if model_name is None:
        model_name = model_dir.name

    config_args = ModelArgs.from_name(model_name).transformer_args['text']
    config = TransformerArgs.from_params(config_args)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json = model_dir / "pytorch_model.bin.index.json"

    # If there is no weight mapping, check for a consolidated model and
    # tokenizer we can move. Llama 2 and Mistral have weight mappings, while
    # Llama 3 has a consolidated model and tokenizer.
    # Otherwise raise an error.
    if not model_map_json.is_file():
        consolidated_pth = model_dir / "original" / "consolidated.00.pth"
        tokenizer_pth = model_dir / "original" / "tokenizer.model"
        if consolidated_pth.is_file() and tokenizer_pth.is_file():
            # Confirm we can load it
            loaded_result = torch.load(
                str(consolidated_pth), map_location="cpu", mmap=True, weights_only=True
            )
            del loaded_result  # No longer needed
            print(f"Moving checkpoint to {model_dir / 'model.pth'}.")
            os.rename(consolidated_pth, model_dir / "model.pth")
            os.rename(tokenizer_pth, model_dir / "tokenizer.model")
            print("Done.")
            return
        else:
            raise RuntimeError(
                f"Could not find {model_map_json} or {consolidated_pth} plus {tokenizer_pth}"
            )

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    bin_files = {model_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_heads):
        dim = config.dim
        return (
            w.view(n_heads, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_heads, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = torch.load(
            str(file), map_location="cpu", mmap=True, weights_only=True
        )
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_heads)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    print(f"Saving checkpoint to {model_dir / 'model.pth'}. This may take a while.")
    torch.save(final_result, model_dir / "model.pth")
    print("Done.")

    if remove_bin_files:
        for file in bin_files:
            os.remove(file)


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    model_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    remove_bin_files: bool = False,
):
    print(model_name)
    print("***********************")
    if "llava" in model_name:
        convert_llava_checkpoint(model_dir=model_dir)
    else:
        convert_text_only_hf_checkpoint(model_dir=model_dir, model_name=model_name, remove_bin_files=remove_bin_files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Hugging Face checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"),
    )
    parser.add_argument("--model-name", type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        model_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# support running without installing as a package
wd = Path(__file__).parent.parent
sys.path.append(str(wd.resolve()))
sys.path.append(str((wd / "build").resolve()))


def convert_hf_checkpoint(
    *,
    model_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    remove_bin_files: bool = False,
) -> None:

    # Local imports to avoid expensive imports
    from torchchat.model import ModelArgs, TransformerArgs
    import torch

    if model_dir is None:
        model_dir = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf")
    if model_name is None:
        model_name = model_dir.name

    # TODO: This is an incongruent way of resolving config_args
    # See https://github.com/pytorch/torchchat/issues/1179
    config_args = ModelArgs.from_name(model_name).transformer_args['text']
    config = TransformerArgs.from_params(config_args)
    print(f"Model config {config.__dict__}")

    # Find all candidate weight mapping index files
    model_map_json_matches = [Path(m) for m in glob.glob(str(model_dir / "*.index.json"))]

    # If there is no weight mapping, check for a consolidated model and
    # tokenizer we can move. Llama 2 and Mistral have weight mappings, while
    # Llama 3 has a consolidated model and tokenizer.
    # Otherwise raise an error.
    if not model_map_json_matches:
        consolidated_pth = model_dir / "original" / "consolidated.00.pth"
        tokenizer_pth = model_dir / "original" / "tokenizer.model"
        if consolidated_pth.is_file() and tokenizer_pth.is_file():
            # Confirm we can load it
            with torch.inference_mode():
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
                f"Could not find a valid model weight map or {consolidated_pth} plus {tokenizer_pth}"
            )

    # Load the json file(s) containing weight mapping
    #
    # NOTE: If there are multiple index files, there are two possibilities:
    #   1. The files could be mapped to different weight format files (e.g. .bin
    #       vs .safetensors)
    #   2. The files could be split subsets of the mappings that need to be
    #       merged
    #
    # In either case, we can simply keep the mappings where the target file is
    # valid in the model dir.
    bin_index = {}
    for weight_map_file in model_map_json_matches:
        with open(weight_map_file, "r") as handle:
            weight_map = json.load(handle)
        valid_mappings = {
            k: model_dir / v
            for (k, v) in weight_map.get("weight_map", {}).items()
            if (model_dir / v).is_file()
        }
        bin_index.update(valid_mappings)
    bin_files = set(bin_index.values())

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.mlp.gate_proj.bias": "layers.{}.feed_forward.w1.bias",
        "model.layers.{}.mlp.up_proj.bias": "layers.{}.feed_forward.w3.bias",
        "model.layers.{}.mlp.down_proj.bias": "layers.{}.feed_forward.w2.bias",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    def permute(w, n_heads):
        return (
            w.view(n_heads, 2, config.head_dim // 2, *w.shape[1:])
            .transpose(1, 2)
            .reshape(w.shape)
        )

    merged_result = {}
    for file in sorted(bin_files):

        # The state_dict can be loaded from either a torch zip file or
        # safetensors. We take our best guess from the name and try all
        # possibilities
        load_pt_mmap = lambda: torch.load(
            str(file), map_location="cpu", mmap=True, weights_only=True
        )
        load_pt_no_mmap = lambda: torch.load(
            str(file), map_location="cpu", mmap=False, weights_only=True
        )
        def load_safetensors():
            import safetensors.torch
            with open(file, "rb") as handle:
                return safetensors.torch.load(handle.read())
        if "safetensors" in str(file):
            loaders = [load_safetensors, load_pt_mmap, load_pt_no_mmap]
        else:
            loaders = [load_pt_mmap, load_pt_no_mmap, load_safetensors]

        state_dict = None
        for loader in loaders:
            try:
                with torch.inference_mode():
                    state_dict = loader()
                break
            except Exception:
                continue
        assert state_dict is not None, f"Unable to load tensors from {file}"
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
        if "wq.weight" in key or "wq.bias" in key:
            wk_key = key.replace("wq", "wk")
            wv_key = key.replace("wq", "wv")
            q = final_result[key]
            k = final_result[wk_key]
            v = final_result[wv_key]
            q = permute(q, config.n_heads)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[wk_key]
            del final_result[wv_key]
    print(f"Saving checkpoint to {model_dir / 'model.pth'}. This may take a while.")
    torch.save(final_result, model_dir / "model.pth")
    print("Done.")

    if remove_bin_files:
        for file in bin_files:
            os.remove(file)


def convert_hf_checkpoint_to_tune(
    *,
    model_dir: Optional[Path] = None,
    model_name: str,
) -> None:
    assert model_dir is not None

    consolidated_pth = model_dir / "original" / "consolidated.pth"
    tokenizer_pth = model_dir / "original" / "tokenizer.model"
    if consolidated_pth.is_file() and tokenizer_pth.is_file():
        print(f"Moving checkpoint to {model_dir / 'model.pth'}.")
        os.rename(consolidated_pth, model_dir / "model.pth")
        print(f"Moving tokenizer to {model_dir / 'tokenizer.model'}.")
        os.rename(tokenizer_pth, model_dir / "tokenizer.model")
        print("Done.")
    else:
        raise RuntimeError(f"Could not find {consolidated_pth}")


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

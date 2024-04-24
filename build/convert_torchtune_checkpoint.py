# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent
sys.path.append(str(wd.resolve()))
sys.path.append(str((wd / "build").resolve()))

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "llama2_7B": {"num_heads": 32, "num_kv_heads": 32, "dim": 4096},
    "llama3_8B": {"num_heads": 32, "num_kv_heads": 8, "dim": 4096},
}

WEIGHT_MAP = {
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


def from_hf(
    merged_result: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
) -> Dict[str, torch.Tensor]:
    """
    Utility function which converts the given state_dict from the HF format
    to one that is compatible with torchchat. The HF-format model involve
    permuting the query and key tensors and this requires additional arguments
    such as num_heads, num_kv_heads and dim.
    """

    def permute(w, n_heads):
        head_dim = dim // n_heads
        return (
            w.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(head_dim * n_heads, dim)
        )

    # Replace the keys with the version compatible with torchchat
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = WEIGHT_MAP[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = WEIGHT_MAP[key]

        final_result[new_key] = value

    # torchchat expects a fused q,k and v matrix
    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, num_heads)
            k = permute(k, num_kv_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    return final_result


@torch.inference_mode()
def convert_torchtune_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_files: List[str],
    checkpoint_format: str,
    model_name: str,
) -> None:

    # Sanity check all for all of the params
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"{checkpoint_dir} is not a directory")

    if len(checkpoint_files) == 0:
        raise RuntimeError("No checkpoint files provided")

    for file in checkpoint_files:
        if not (Path.joinpath(checkpoint_dir, file)).is_file():
            raise RuntimeError(f"{checkpoint_dir / file} is not a file")

    # If the model is already in meta format, simply rename it
    if checkpoint_format == "meta":
        if len(checkpoint_files) > 1:
            raise RuntimeError("Multiple meta format checkpoint files not supported")

        checkpoint_path = Path.joinpath(checkpoint_dir, checkpoint_files[0])
        loaded_result = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=True
        )
        del loaded_result

        os.rename(checkpoint_path, Path.joinpath(checkpoint_dir, "model.pth"))

    # If the model is in HF format, merge all of the checkpoints and then convert
    elif checkpoint_format == "hf":
        merged_result = {}
        for file in checkpoint_files:
            state_dict = torch.load(
                Path.joinpath(checkpoint_dir, file),
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            merged_result.update(state_dict)

        model_config = MODEL_CONFIGS[model_name]
        final_result = from_hf(merged_result, **model_config)

        print(
            f"Saving checkpoint to {checkpoint_dir / 'model.pth'}. This may take a while."
        )
        torch.save(final_result, Path.joinpath(checkpoint_dir, "model.pth"))
        print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert torchtune checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint-files",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-format",
        type=str,
        required=True,
        choices=["meta", "hf"],
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["llama2_7B", "llama3_8B"],
    )

    args = parser.parse_args()
    convert_torchtune_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_files=args.checkpoint_files,
        checkpoint_format=args.checkpoint_format,
        model_name=args.model_name,
    )

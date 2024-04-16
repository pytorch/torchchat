# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from build.convert_hf_checkpoint import convert_hf_checkpoint
from pathlib import Path
from typing import Optional

from requests.exceptions import HTTPError


def hf_download(
        repo_id: Optional[str] = None, 
        model_dir: Optional[Path] = None,
        hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download

    if model_dir is None:
        model_dir = Path(".model-artifacts/{repo_id}")

    try:
        snapshot_download(
            repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns="*safetensors*")
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
        else:
            raise e


def main(args):
    model_dir = Path(args.model_directory) / args.model
    os.makedirs(model_dir, exist_ok=True)

    # Download and store the HF model artifacts.
    print(f"Downloading {args.model} from HuggingFace...")
    hf_download(args.model, model_dir, args.hf_token)

    # Convert the model to the torchchat format.
    print(f"Converting {args.model} to torchchat format...")
    convert_hf_checkpoint(
        model_dir=model_dir,
        model_name=Path(args.model),
        remove_bin_files=True)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from build.convert_hf_checkpoint import convert_hf_checkpoint
from build.model import model_aliases
from pathlib import Path
from typing import Optional

from requests.exceptions import HTTPError

def download_and_convert(
        model: str,
        models_dir: Path,
        hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download

    if model in model_aliases:
        model = model_aliases[model]

    model_dir = models_dir / model
    os.makedirs(model_dir, exist_ok=True)

    # Download and store the HF model artifacts.
    print(f"Downloading {model} from HuggingFace...")
    try:
        snapshot_download(
            model,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns="*safetensors*")
    except HTTPError as e:
        if e.response.status_code == 401:
            raise RuntimeError("You need to pass a valid `--hf_token=...` to download private checkpoints.")
        else:
            raise e

    # Convert the model to the torchchat format.
    print(f"Converting {model} to torchchat format...")
    convert_hf_checkpoint(
        model_dir=model_dir,
        model_name=Path(model),
        remove_bin_files=True)

def is_model_downloaded(
        model: str, 
        models_dir: Path) -> bool:
    if model in model_aliases:
        model = model_aliases[model]

    model_dir = models_dir / model

    # TODO Can we be more thorough here?
    return os.path.isdir(model_dir)


def main(args):
    download_and_convert(args.model, args.model_directory, args.hf_token)

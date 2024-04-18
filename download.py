# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from build.convert_hf_checkpoint import convert_hf_checkpoint
from build.model import ModelConfig, ModelDistributionChannel, resolve_model_config

from requests.exceptions import HTTPError


def _download_and_convert_hf_snapshot(
    model: str, models_dir: Path, hf_token: Optional[str]
):
    model_dir = models_dir / model
    os.makedirs(model_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    # Download and store the HF model artifacts.
    print(f"Downloading {model} from HuggingFace...")
    try:
        snapshot_download(
            model,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns="*safetensors*",
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            raise RuntimeError(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e

    # Convert the model to the torchchat format.
    print(f"Converting {model} to torchchat format...")
    convert_hf_checkpoint(model_dir=model_dir, model_name=model, remove_bin_files=True)


def _download_direct(
    model: str,
    urls: Sequence[str],
    models_dir: Path,
):
    model_dir = models_dir / model
    os.makedirs(model_dir, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        local_path = model_dir / filename
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, str(local_path.absolute()))


def download_and_convert(
    model: str, models_dir: Path, hf_token: Optional[str] = None
) -> None:
    model_config, model_name = resolve_model_config(model)

    if (
        model_config.distribution_channel
        == ModelDistributionChannel.HuggingFaceSnapshot
    ):
        _download_and_convert_hf_snapshot(model_name, models_dir, hf_token)
    elif model_config.distribution_channel == ModelDistributionChannel.DirectDownload:
        _download_direct(model_name, model_config.distribution_path, models_dir)
    else:
        raise RuntimeError(
            f"Unknown distribution channel {model_config.distribution_channel}."
        )


def is_model_downloaded(model: str, models_dir: Path) -> bool:
    _, model_name = resolve_model_config(model)

    model_dir = models_dir / model_name
    return os.path.isdir(model_dir)


def main(args):
    download_and_convert(args.model, args.model_directory, args.hf_token)

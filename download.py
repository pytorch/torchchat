# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from build.convert_hf_checkpoint import convert_hf_checkpoint
from config.model_config import (
    ModelConfig,
    ModelDistributionChannel,
    load_model_configs,
    resolve_model_config,
)


def _download_hf_snapshot(
    model_config: ModelConfig, artifact_dir: Path, hf_token: Optional[str]
):
    from huggingface_hub import snapshot_download
    from requests.exceptions import HTTPError

    # Download and store the HF model artifacts.
    print(f"Downloading {model_config.name} from HuggingFace...")
    try:
        snapshot_download(
            model_config.distribution_path,
            local_dir=artifact_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns="*safetensors*",
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            raise RuntimeError(
                "Access denied. Run huggingface-cli login to authenticate."
            )
        else:
            raise e

    # Convert the model to the torchchat format.
    print(f"Converting {model_config.name} to torchchat format...")
    convert_hf_checkpoint(
        model_dir=artifact_dir, model_name=model_config.name, remove_bin_files=True
    )


def _download_direct(
    model_config: ModelConfig,
    artifact_dir: Path,
):
    for url in model_config.distribution_path:
        filename = url.split("/")[-1]
        local_path = artifact_dir / filename
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, str(local_path.absolute()))


def download_and_convert(
    model: str, models_dir: Path, hf_token: Optional[str] = None
) -> None:
    model_config = resolve_model_config(model)
    model_dir = models_dir / model_config.name

    # Download into a temporary directory. We'll move to the final location once
    # the download and conversion is complete. This allows recovery in the event
    # that the download or conversion fails unexpectedly.
    temp_dir = models_dir / "downloads" / model_config.name
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        if (
            model_config.distribution_channel
            == ModelDistributionChannel.HuggingFaceSnapshot
        ):
            _download_hf_snapshot(model_config, temp_dir, hf_token)
        elif (
            model_config.distribution_channel == ModelDistributionChannel.DirectDownload
        ):
            _download_direct(model_config, temp_dir)
        else:
            raise RuntimeError(
                f"Unknown distribution channel {model_config.distribution_channel}."
            )

        # Move from the temporary directory to the intended location,
        # overwriting if necessary.
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        shutil.move(temp_dir, model_dir)

    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


def is_model_downloaded(model: str, models_dir: Path) -> bool:
    model_config = resolve_model_config(model)

    # Check if the model directory exists and is not empty.
    model_dir = models_dir / model_config.name
    return os.path.isdir(model_dir) and os.listdir(model_dir)


# Subcommand to list available models.
def list_main(args) -> None:
    # TODO It would be nice to have argparse validate this. However, we have
    # model as an optional named parameter for all subcommands, so we'd
    # probably need to move it to be registered per-command.
    if args.model:
        print("Usage: torchchat.py list")
        return

    model_configs = load_model_configs()

    # Build the table in-memory so that we can align the text nicely.
    name_col = []
    aliases_col = []
    installed_col = []

    for name, config in model_configs.items():
        is_downloaded = is_model_downloaded(name, args.model_directory)

        name_col.append(name)
        aliases_col.append(", ".join(config.aliases))
        installed_col.append('Yes' if is_downloaded else "")

    cols = {
        "Model": name_col,
        "Aliases": aliases_col,
        "Downloaded": installed_col
    }

    # Find the length of the longest value in each column.
    col_widths = {key:max(*[len(s) for s in vals], len(key)) + 1 for (key,vals) in cols.items()}

    # Display header.
    print()
    print(*[val.ljust(width) for (val, width) in col_widths.items()])
    print(*["-" * width for width in col_widths.values()])

    for i in range(len(name_col)):
        row = [col[i] for col in cols.values()]
        print(*[val.ljust(width) for (val, width) in zip(row, col_widths.values())])
    print()


# Subcommand to remove downloaded model artifacts.
def remove_main(args) -> None:
    # TODO It would be nice to have argparse validate this. However, we have
    # model as an optional named parameter for all subcommands, so we'd
    # probably need to move it to be registered per-command.
    if not args.model:
        print("Usage: torchchat.py remove <model-or-alias>")
        return

    model_config = resolve_model_config(args.model)
    model_dir = args.model_directory / model_config.name

    if not os.path.isdir(model_dir):
        print(f"Model {args.model} has no downloaded artifacts.")
        return

    print(f"Removing downloaded model artifacts for {args.model}...")
    shutil.rmtree(model_dir)
    print("Done.")


# Subcommand to download model artifacts.
def download_main(args) -> None:
    download_and_convert(args.model, args.model_directory, args.hf_token)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

from torchchat.cli.convert_hf_checkpoint import (
    convert_hf_checkpoint,
    convert_hf_checkpoint_to_tune,
)
from torchchat.model_config.model_config import (
    load_model_configs,
    ModelConfig,
    ModelDistributionChannel,
    resolve_model_config,
)

# By default, download models from HuggingFace to the Hugginface hub directory.
# Both $HF_HOME and $HUGGINGFACE_HUB_CACHE are valid environment variables for the same directory.
HUGGINGFACE_HOME_PATH = Path(
    os.environ.get(
        "HF_HOME",
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
        ),
    )
)

if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", None) is None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# Previously, all models were stored in the torchchat models directory (by default ~/.torchchat/model-cache)
# For Hugging Face models, we now store them in the HuggingFace cache directory.
# This function will delete all model artifacts in the old directory for each model with the Hugging Face distribution path.
def _cleanup_hf_models_from_torchchat_dir(models_dir: Path):
    for model_config in load_model_configs().values():
        if (
            model_config.distribution_channel
            == ModelDistributionChannel.HuggingFaceSnapshot
        ):
            if os.path.exists(models_dir / model_config.name):
                print(
                    f"Cleaning up old model artifacts in {models_dir / model_config.name}. New artifacts will be downloaded to {HUGGINGFACE_HOME_PATH}"
                )
                shutil.rmtree(models_dir / model_config.name)


def _download_hf_snapshot(model_config: ModelConfig, hf_token: Optional[str]):
    from huggingface_hub import model_info, snapshot_download
    from requests.exceptions import HTTPError

    # Download and store the HF model artifacts.
    model_dir = get_model_dir(model_config, None)
    print(
        f"Downloading {model_config.name} from Hugging Face to {model_dir}",
        file=sys.stderr,
        flush=True,
    )
    try:
        # Fetch the info about the model's repo
        model_info = model_info(model_config.distribution_path, token=hf_token)
        model_fnames = [f.rfilename for f in model_info.siblings]

        # Check the model config for preference between safetensors and pth
        has_pth = any(f.endswith(".pth") for f in model_fnames)
        has_safetensors = any(f.endswith(".safetensors") for f in model_fnames)

        # If told to prefer safetensors, ignore pth files
        if model_config.prefer_safetensors:
            if not has_safetensors:
                print(
                    f"Model {model_config.name} does not have safetensors files, but prefer_safetensors is set to True. Using pth files instead.",
                    file=sys.stderr,
                )
                exit(1)
            ignore_patterns = "*.pth"

        # If the model has both, prefer pth files over safetensors
        elif has_pth and has_safetensors:
            ignore_patterns = "*safetensors*"

        # Otherwise, download everything
        else:
            ignore_patterns = None

        snapshot_download(
            model_config.distribution_path,
            token=hf_token,
            ignore_patterns=ignore_patterns,
        )
    except HTTPError as e:
        if e.response.status_code == 401:  # Missing HuggingFace CLI login.
            print(
                "Access denied. Create a HuggingFace account and run 'pip3 install huggingface_hub' and 'huggingface-cli login' to authenticate.",
                file=sys.stderr,
            )
            exit(1)
        elif e.response.status_code == 403:  # No access to the specific model.
            # The error message includes a link to request access to the given model. This prints nicely and does not include
            # a traceback.
            print(str(e), file=sys.stderr)
            exit(1)
        else:
            raise e

    # Update the model dir to include the snapshot we just downloaded.
    model_dir = get_model_dir(model_config, None)
    print("Model downloaded to", model_dir)

    # Convert the Multimodal Llama model to the torchtune format.
    if model_config.name in {
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
    }:
        print(f"Converting {model_config.name} to torchtune format...", file=sys.stderr)
        convert_hf_checkpoint_to_tune(model_dir=model_dir, model_name=model_config.name)

    else:
        # Convert the model to the torchchat format.
        print(f"Converting {model_config.name} to torchchat format...", file=sys.stderr)
        convert_hf_checkpoint(
            model_dir=model_dir, model_name=model_config.name, remove_bin_files=True
        )


def _download_direct(
    model_config: ModelConfig,
    artifact_dir: Path,
):
    for url in model_config.distribution_path:
        filename = url.split("/")[-1]
        local_path = artifact_dir / filename
        print(f"Downloading {url}...", file=sys.stderr)
        urllib.request.urlretrieve(url, str(local_path.absolute()))


def _get_hf_artifact_dir(model_config: ModelConfig) -> Path:
    """
    Returns the directory where the model artifacts are stored.

    This is the root folder with blobs, refs and snapshots
    """
    assert (
        model_config.distribution_channel
        == ModelDistributionChannel.HuggingFaceSnapshot
    )
    return (
        HUGGINGFACE_HOME_PATH
        / f"models--{model_config.distribution_path.replace('/', '--')}"
    )


def get_model_dir(model_config: ModelConfig, models_dir: Optional[Path]) -> Path:
    """
    Returns the directory where the model artifacts are expected to be stored.
    For Hugging Face artifacts, this will be the location of the "main" snapshot if it exists, or the expected model directory otherwise.
    For all other distribution channels, we use the models_dir.

    For CLI usage, pass in args.model_directory.
    """
    if (
        model_config.distribution_channel
        == ModelDistributionChannel.HuggingFaceSnapshot
    ):
        artifact_dir = _get_hf_artifact_dir(model_config)

        # If these paths doesn't exist, it means the model hasn't been downloaded yet.
        if not os.path.isdir(artifact_dir) and not os.path.isdir(
            artifact_dir / "snapshots"
        ):
            return artifact_dir
        snapshot = open(artifact_dir / "refs" / "main", "r").read().strip()
        return artifact_dir / "snapshots" / snapshot
    else:
        return models_dir / model_config.name


def download_and_convert(
    model: str, models_dir: Path, hf_token: Optional[str] = None
) -> None:
    model_config = resolve_model_config(model)
    model_dir = get_model_dir(model_config, models_dir)

    # HuggingFace download
    if (
        model_config.distribution_channel
        == ModelDistributionChannel.HuggingFaceSnapshot
    ):
        _download_hf_snapshot(model_config, hf_token)
        return

    # Direct download

    # Download into a temporary directory. We'll move to the final
    # location once the download and conversion is complete. This
    # allows recovery in the event that the download or conversion
    # fails unexpectedly.
    temp_dir = models_dir / "downloads" / model_config.name
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        if model_config.distribution_channel == ModelDistributionChannel.DirectDownload:
            _download_direct(model_config, temp_dir)
        else:
            raise RuntimeError(
                f"Unknown distribution channel {model_config.distribution_channel}."
            )

        # Move from the temporary directory to the intended location,
        # overwriting if necessary.
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        print(f"Moving model to {model_dir}.")
        shutil.move(temp_dir, model_dir)

    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


def is_model_downloaded(model: str, models_dir: Path) -> bool:
    model_config = resolve_model_config(model)

    # Check if the model directory exists and is not empty.
    model_dir = get_model_dir(model_config, models_dir)
    return os.path.isdir(model_dir) and os.listdir(model_dir)


# Subcommand to list available models.
def list_main(args) -> None:
    model_configs = load_model_configs()

    # Build the table in-memory so that we can align the text nicely.
    name_col = []
    aliases_col = []
    installed_col = []

    for name, config in model_configs.items():
        is_downloaded = is_model_downloaded(name, args.model_directory)

        name_col.append(name)
        aliases_col.append(", ".join(config.aliases))
        installed_col.append("Yes" if is_downloaded else "")

    cols = {"Model": name_col, "Aliases": aliases_col, "Downloaded": installed_col}

    # Find the length of the longest value in each column.
    col_widths = {
        key: max(*[len(s) for s in vals], len(key)) + 1 for (key, vals) in cols.items()
    }

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
    model_dir = get_model_dir(model_config, args.model_directory)

    if not os.path.isdir(model_dir):
        print(f"Model {args.model} has no downloaded artifacts in {model_dir}.")
        return
    if (
        model_config.distribution_channel
        == ModelDistributionChannel.HuggingFaceSnapshot
    ):
        # For HuggingFace models, we need to remove the entire root directory.
        model_dir = _get_hf_artifact_dir(model_config)

    print(f"Removing downloaded model artifacts for {args.model} at {model_dir}...")
    shutil.rmtree(model_dir)
    print("Done.")


# Subcommand to print downloaded model artifacts directory.
# Asking for location will/should trigger download of model if not available.
def where_main(args) -> None:
    # TODO It would be nice to have argparse validate this. However, we have
    # model as an optional named parameter for all subcommands, so we'd
    # probably need to move it to be registered per-command.
    if not args.model:
        print("Usage: torchchat.py where <model-or-alias>")
        return

    model_config = resolve_model_config(args.model)
    model_dir = get_model_dir(model_config, args.model_directory)

    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Model {args.model} has no downloaded artifacts in {model_dir}."
        )

    print(str(os.path.abspath(model_dir)))
    exit(0)


# Subcommand to download model artifacts.
def download_main(args) -> None:
    _cleanup_hf_models_from_torchchat_dir(args.model_directory)
    download_and_convert(args.model, args.model_directory, args.hf_token)

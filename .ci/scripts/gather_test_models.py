#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import os
from typing import Any


MODEL_REPOS = {
    "tinyllamas/stories15M": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt,https://github.com/karpathy/llama2.c/raw/master/tokenizer.model,https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
    # "tinyllamas/stories42M": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.pt,https://github.com/karpathy/llama2.c/raw/master/tokenizer.model,https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
    "tinyllamas/stories110M": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt,https://github.com/karpathy/llama2.c/raw/master/tokenizer.model,https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
    "openlm-research/open_llama_7b": "https://huggingface.co/openlm-research/open_llama_7b/resolve/main/config.json,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/generation_config.json,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/pytorch_model-00001-of-00002.bin,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/pytorch_model-00002-of-00002.bin,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/pytorch_model.bin.index.json,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/special_tokens_map.json,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/tokenizer.model,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/tokenizer.model,https://huggingface.co/openlm-research/open_llama_7b/resolve/main/tokenizer_config.json",
    "mistralai/Mistral-7B-v0.1": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/generation_config.json,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00002-of-00002.bin,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model.bin.index.json,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/special_tokens_map.json,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model,https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer_config.json",
    "mistralai/Mistral-7B-Instruct-v0.1": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/config.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/generation_config.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-00001-of-00002.bin,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-00002-of-00002.bin,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model.bin.index.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/special_tokens_map.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer_config.json",
    "mistralai/Mistral-7B-Instruct-v0.2": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/config.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/generation_config.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model-00001-of-00003.bin,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model-00002-of-00003.bin,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model-00003-of-00003.bin,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model.bin.index.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/special_tokens_map.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.json,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.model,https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer_config.json",
    # huggingface-cli prefixed Models will download using the huggingface-cli tool
    # TODO: Convert all of the MODEL_REPOS with a NamedTuple that includes the install_method
    "huggingface-cli/meta-llama/Meta-Llama-3-8B": "",
}

JOB_RUNNERS = {
    "cpu": {
        "8-core-ubuntu": "x86_64",
        # "macos-12": "x86_64", # not working for complie and ExecuTorch yet
        "macos-14": "aarch64",
    },
    "gpu": {
        "linux.g5.4xlarge.nvidia.gpu": "cuda",
    },
}


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Gather all models to test on CI for the target OS")
    parser.add_argument(
        "-e",
        "--event",
        type=str,
        choices=["pull_request", "push", "periodic"],
        required=True,
        help="GitHub CI Event. See https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#on",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=["cpu", "gpu"],
        required=True,
        help="Supported backends to run. ['cpu', 'gpu']",
    )

    return parser.parse_args()


def model_should_run_on_event(model: str, event: str, backend: str) -> bool:
    """
    A helper function to decide whether a model should be tested on an event (pull_request/push)
    We put higher priority and fast models to pull request and rest to push.
    """
    if event == "pull_request":
        return model in ["tinyllamas/stories15M"]
    elif event == "push":
        return model in []
    elif event == "periodic":
        # test llama3 on gpu only, see description in https://github.com/pytorch/torchchat/pull/399 for reasoning
        if backend == "gpu":
            return model in [
                "openlm-research/open_llama_7b",
                "huggingface-cli/meta-llama/Meta-Llama-3-8B",
            ]
        else:
            return model in ["openlm-research/open_llama_7b"]
    else:
        return False


def set_output(name: str, val: Any) -> None:
    """
    Set the GitHb output so that it can be accessed by other jobs
    """
    print(f"Setting {val} to GitHub output")

    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def export_models_for_ci() -> dict[str, dict]:
    """
    This gathers all the models that we want to test on GitHub OSS CI
    """

    args = parse_args()
    event = args.event
    backend = args.backend

    # This is the JSON syntax for configuration matrix used by GitHub
    # https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
    models = {"include": []}

    for repo_name, runner in itertools.product(
        MODEL_REPOS.keys(),
        JOB_RUNNERS[backend].items(),
    ):
        if not model_should_run_on_event(repo_name, event, backend):
            continue

        # This is mostly temporary to get this finished quickly while
        # doing minimal changes, see TODO at the top of the file to
        # see how this should probably be done
        install_method = "wget"
        final_repo_name = repo_name
        if repo_name.startswith("huggingface-cli"):
            install_method = "huggingface-cli"
            final_repo_name = repo_name.replace("huggingface-cli/", "")

        record = {
            "repo_name": final_repo_name,
            "model_name": final_repo_name.split("/")[-1],
            "resources": MODEL_REPOS[repo_name],
            "runner": runner[0],
            "platform": runner[1],
            "install_method": install_method,
            "timeout": 90,
        }

        models["include"].append(record)

    set_output("models", json.dumps(models))


if __name__ == "__main__":
    export_models_for_ci()

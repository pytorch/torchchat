#!/usr/bin/env python
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
}

JOB_RUNNERS = {
    "32-core-ubuntu": "linux x86",
    # "macos-13": "macos x86", # not working for ExecuTorch yet
    "macos-14": "macos M1",
}


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

    # This is the JSON syntax for configuration matrix used by GitHub
    # https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
    models = {"include": []}

    for repo_name, runner in itertools.product(
        MODEL_REPOS.keys(),
        JOB_RUNNERS.keys(),
    ):
        record = {
            "repo_name": repo_name,
            "resources": MODEL_REPOS[repo_name],
            "runner": runner,
            "platform": JOB_RUNNERS[runner],
            "timeout": 90,
        }

        models["include"].append(record)

    set_output("models", json.dumps(models))


if __name__ == "__main__":
    export_models_for_ci()

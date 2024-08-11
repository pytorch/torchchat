#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eou pipefail

# Install required python dependencies for developing
# Dependencies are defined in .pyproject.toml
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
if [[ -z ${CONDA_DEFAULT_ENV:-} ]] || [[ ${CONDA_DEFAULT_ENV:-} == "base" ]] || [[ ! -x "$(command -v python)" ]];
then
  PYTHON_EXECUTABLE=python3
fi

# Check python version. Expect 3.10.x or 3.11.x
printf "import sys\nif sys.version_info.major != 3 or sys.version_info.minor < 10 :\n\tprint('Please use Python >=3.10');sys.exit(1)\n" | $PYTHON_EXECUTABLE
if [[ $? -ne 0 ]]
then
  exit 1
fi

if [[ "$PYTHON_EXECUTABLE" == "python" ]];
then
  PIP_EXECUTABLE=pip
else
  PIP_EXECUTABLE=pip3
fi

#
# First install requirements in requirements.txt. Older torch may be
# installed from the dependency of other models. It will be overridden by
# newer version of torch nightly installed later in this script.
#

(
  set -x
  $PIP_EXECUTABLE install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121
)

# Since torchchat often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features. The NIGHTLY_VERSION value should
# agree with the third-party/pytorch pinned submodule commit.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
NIGHTLY_VERSION=dev20240728

# Uninstall triton, as nightly will depend on pytorch-triton, which is one and the same
(
  set -x
  $PIP_EXECUTABLE uninstall -y triton
)

# The pip repository that hosts nightly torch packages. cpu by default.
# If cuda is available, based on presence of nvidia-smi, install the pytorch nightly
# with cuda for faster execution on cuda GPUs.
if [[ -x "$(command -v nvidia-smi)" ]];
then
  TORCH_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/cu121"
else
  TORCH_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/cpu"
fi

# pip packages needed by exir.
REQUIREMENTS_TO_INSTALL=(
  torch=="2.5.0.${NIGHTLY_VERSION}"
)

# Install the requirements. `--extra-index-url` tells pip to look for package
# versions on the provided URL if they aren't available on the default URL.
(
  set -x
  $PIP_EXECUTABLE install --extra-index-url "${TORCH_NIGHTLY_URL}" \
    "${REQUIREMENTS_TO_INSTALL[@]}"
)

# For torchao need to install from github since nightly build doesn't have macos build.
# TODO: Remove this and install nightly build, once it supports macos
(
  set -x
  $PIP_EXECUTABLE install git+https://github.com/pytorch/ao.git@d477c0e59b458b5617dcb3e999290a87df3070d8
)
if [[ -x "$(command -v nvidia-smi)" ]]; then
  (
    set -x
    $PYTHON_EXECUTABLE scripts/patch_triton.py
  )
fi

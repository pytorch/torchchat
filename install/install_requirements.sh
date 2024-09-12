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
# First install requirements in install/requirements.txt. Older torch may be
# installed from the dependency of other models. It will be overridden by
# newer version of torch nightly installed later in this script.
#

(
  set -x
  $PIP_EXECUTABLE install -r install/requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121
)

# Since torchchat often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features. The PYTORCH_NIGHTLY_VERSION value should
# agree with the third-party/pytorch pinned submodule commit.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# PYTORCH_NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
PYTORCH_NIGHTLY_VERSION=dev20240814

# Nightly version for torchvision
VISION_NIGHTLY_VERSION=dev20240814

# Nightly version for torchao
AO_NIGHTLY_VERSION=dev20240905

# Nightly version for torchtune
TUNE_NIGHTLY_VERSION=dev20240910


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
  torch=="2.5.0.${PYTORCH_NIGHTLY_VERSION}"
  torchvision=="0.20.0.${VISION_NIGHTLY_VERSION}"
)

LINUX_REQUIREMENTS_TO_INSTALL=(
  torchao=="0.5.0.${AO_NIGHTLY_VERSION}"
  torchtune=="0.3.0.${TUNE_NIGHTLY_VERSION}"
)

# Install the requirements. --extra-index-url tells pip to look for package
# versions on the provided URL if they aren't available on the default URL.
(
  set -x
  $PIP_EXECUTABLE install --extra-index-url "${TORCH_NIGHTLY_URL}" \
    "${REQUIREMENTS_TO_INSTALL[@]}"
)

PLATFORM=$(uname -s)

# Install torchtune and torchao requirements for Linux systems using nightly.
# For non-Linux systems (e.g., macOS), install torchao from GitHub since nightly
# build doesn't have macOS build.
# TODO: Remove this and install nightly build, once it supports macOS
if [ "$PLATFORM" == "Linux" ];
then
  (
    set -x
    $PIP_EXECUTABLE install --pre --extra-index-url "${TORCH_NIGHTLY_URL}" --no-cache-dir \
      "${LINUX_REQUIREMENTS_TO_INSTALL[@]}"
  )
else
  # For torchao need to install from github since nightly build doesn't have macos build.
  # TODO: Remove this and install nightly build, once it supports macos
  (
    set -x
    $PIP_EXECUTABLE install git+https://github.com/pytorch/ao.git@e11201a62669f582d81cdb33e031a07fb8dfc4f3
  )
fi

if [[ -x "$(command -v nvidia-smi)" ]]; then
  (
    set -x
    $PYTHON_EXECUTABLE torchchat/utils/scripts/patch_triton.py
  )
fi

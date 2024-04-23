#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install required python dependencies for developing
# Dependencies are defined in .pyproject.toml
if [[ -z $PYTHON_EXECUTABLE ]];
then
  if [[ -z $CONDA_DEFAULT_ENV ]] || [[ $CONDA_DEFAULT_ENV == "base" ]] || [[ ! -x "$(command -v python)" ]];
  then
    PYTHON_EXECUTABLE=python3
  else
    PYTHON_EXECUTABLE=python
  fi
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

$PIP_EXECUTABLE install -r requirements.txt

# Since torchchat often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features. The NIGHTLY_VERSION value should
# agree with the third-party/pytorch pinned submodule commit.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
NIGHTLY_VERSION=dev20240415

# The pip repository that hosts nightly torch packages.
TORCH_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/cpu"

# pip packages needed by exir.
REQUIREMENTS_TO_INSTALL=(
  torch=="2.4.0.${NIGHTLY_VERSION}"
)

# Install the requirements. `--extra-index-url` tells pip to look for package
# versions on the provided URL if they aren't available on the default URL.
$PIP_EXECUTABLE install --extra-index-url "${TORCH_NIGHTLY_URL}" \
    "${REQUIREMENTS_TO_INSTALL[@]}"

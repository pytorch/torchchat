#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# USE_CPP=1 indicates that the torchao experimental aten kernels will be built and loaded
# if on Mac with Apple Silicon

if [ -z "${PYTHON_EXECUTABLE:-}" ];
then
  if [[ -z ${CONDA_DEFAULT_ENV:-} ]] || [[ ${CONDA_DEFAULT_ENV:-} == "base" ]] || [[ ! -x "$(command -v python)" ]];
  then
    PYTHON_EXECUTABLE=python3
  else
    PYTHON_EXECUTABLE=python
  fi
fi
echo "Using python executable: $PYTHON_EXECUTABLE"

if [[ "$PYTHON_EXECUTABLE" == "python" ]];
then
  PIP_EXECUTABLE=pip
elif [[ "$PYTHON_EXECUTABLE" == "python3" ]];
then
  PIP_EXECUTABLE=pip3
else
  PIP_EXECUTABLE=pip${PYTHON_SYS_VERSION}
fi
echo "Using pip executable: $PIP_EXECUTABLE"

if [[ $(uname -s) == "Darwin" && $(uname -m) == "arm64" ]]; then
  echo "Building torchao experimental mps ops (Apple Silicon detected)"
  APPLE_SILICON_DETECTED=1
else
  echo "NOT building torchao experimental mps ops (Apple Silicon NOT detected)"
  APPLE_SILICON_DETECTED=0
fi

export TORCHAO_PIN=$(cat install/.pins/torchao-pin.txt)
(
  set -x
  USE_CPP=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=${APPLE_SILICON_DETECTED} $PIP_EXECUTABLE install git+https://github.com/pytorch/ao.git@${TORCHAO_PIN}
)

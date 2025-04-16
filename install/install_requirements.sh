#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eou pipefail

# Install required python dependencies for developing
# Dependencies are defined in .pyproject.toml
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

PYTHON_SYS_VERSION="$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
# Check python version. Expect at least 3.10.x
if ! $PYTHON_EXECUTABLE -c "
import sys
if sys.version_info < (3, 10):
    sys.exit(1)
";
then
  echo "Python version must be at least 3.10.x. Detected version: $PYTHON_SYS_VERSION"
  exit 1
fi

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

(
  set -x
  $PIP_EXECUTABLE install -r install/requirements.txt
)

bash install/install_torch.sh

(
  set -x
  $PIP_EXECUTABLE install evaluate=="0.4.3" lm-eval=="0.4.7" psutil=="6.0.0"
)

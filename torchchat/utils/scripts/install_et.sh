#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex pipefail

source "$(dirname "${BASH_SOURCE[0]}")/install_utils.sh"

if [ "${ET_BUILD_DIR}" == "" ]; then
  ET_BUILD_DIR="et-build"
fi

ENABLE_ET_PYBIND="${1:-true}"

pushd ${TORCHCHAT_ROOT}
find_cmake_prefix_path
clone_executorch
install_executorch_libs $ENABLE_ET_PYBIND
install_executorch_python_libs $ENABLE_ET_PYBIND
# TODO: figure out the root cause of 'AttributeError: module 'evaluate'
# has no attribute 'utils'' error from evaluate CI jobs and remove
# `import lm_eval` from torchchat.py since it requires a specific version
# of numpy.
pip install numpy=='1.21.3'
popd

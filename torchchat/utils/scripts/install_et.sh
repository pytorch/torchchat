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
popd

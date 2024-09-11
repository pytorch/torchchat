#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [ -z "${TORCHCHAT_ROOT}" ]; then
    # Get the absolute path of the current script
    SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
    # Get the absolute path of the parent directory
    TORCHCHAT_ROOT="$(dirname "$SCRIPT_PATH")"
fi

if [ -z "${TORCHAO_BUILD_DIR}" ]; then
    TORCHAO_BUILD_DIR="torchao-build"
fi

source "$TORCHCHAT_ROOT/scripts/install_utils.sh"

find_cmake_prefix_path
clone_torchao
install_torchao_custom_aten_ops

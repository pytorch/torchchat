#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

device=${1:-cpu}

if [[ "$device" != "cpu" && "$device" != "mps" ]]; then
  echo "Invalid argument: $device. Valid values are 'cpu' or 'mps'." >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/install_utils.sh"

if [ ! -d "${TORCHCHAT_ROOT}/torchao-build" ]; then
  echo "Directory ${TORCHCHAT_ROOT}/torchao-build does not exist."
  echo "Make sure you run bash torchchat/utils/scripts/clone_torchao.sh"
  exit 1
fi

pushd ${TORCHCHAT_ROOT}
find_cmake_prefix_path
install_torchao_aten_ops "$device"
popd

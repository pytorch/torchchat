#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Simple script to build native aoti and et runner
# Function to display a help message

set -ex

show_help() {
cat << EOF
Usage: ${0##*/} [-h|--help] aoti|et
This script builds native aoti and et runner for LLM.
    -h|--help  Display this help and exit
    aoti       Build native runner for aoti
    et         Build native runner for et
EOF
}
# Check if no arguments were passed
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    show_help
    exit 1
fi
while (( "$#" )); do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    aoti)
      echo "Building aoti native runner..."
      TARGET="aoti"
      shift
      ;;
    et)
      echo "Building et native runner..."
      TARGET="et"
      shift
      ;;
    *)
      echo "Invalid option: $1"
      show_help
      exit 1
      ;;
  esac
done

if [ -z "${TORCHCHAT_ROOT}" ]; then
    # Get the absolute path of the current script
    SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
    # Get the absolute path of the parent directory
    TORCHCHAT_ROOT="$(dirname "$SCRIPT_PATH")"
fi

if [ -z "${ET_BUILD_DIR}" ]; then
    ET_BUILD_DIR="et-build"
fi

source "$TORCHCHAT_ROOT/scripts/install_utils.sh"

pushd ${TORCHCHAT_ROOT}
git submodule update --init
git submodule sync
if [[ "$TARGET" == "et" ]]; then
    find_cmake_prefix_path
    install_pip_dependencies
    clone_executorch
    install_executorch_libs false
fi
popd

# CMake commands
if [[ "$TARGET" == "et" ]]; then
    cmake -S . -B ./cmake-out -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" -G Ninja
else
    cmake -S . -B ./cmake-out -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" -G Ninja
fi
cmake --build ./cmake-out --target "${TARGET}"_run

printf "Build finished. Please run: \n./cmake-out/${TARGET}_run model.<pte|so> -z tokenizer.model -l <llama version (2 or 3)> -i <prompt>\n"

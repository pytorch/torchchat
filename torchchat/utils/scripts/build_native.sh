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

source "$(dirname "${BASH_SOURCE[0]}")/install_utils.sh"

if [ -z "${ET_BUILD_DIR}" ]; then
    ET_BUILD_DIR="et-build"
fi


pushd ${TORCHCHAT_ROOT}
git submodule update --init
git submodule sync
if [[ "$TARGET" == "et" ]]; then
  if [ ! -d "${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install" ]; then
    echo "Directory ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install does not exist."
    echo "Make sure you run install_executorch_libs"
    exit 1
  fi
fi
popd

# CMake commands
if [[ "$TARGET" == "et" ]]; then
    cmake -S . -B ./cmake-out -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DET_USE_ADAPTIVE_THREADS=ON -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" -G Ninja
else
    cmake -S . -B ./cmake-out -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" -G Ninja
fi
cmake --build ./cmake-out --target "${TARGET}"_run

printf "Build finished. Please run: \n./cmake-out/${TARGET}_run model.<pte|so> -z tokenizer.model -l <llama version (2 or 3)> -i <prompt>\n"

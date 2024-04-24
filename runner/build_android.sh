#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/install_utils.sh"

if ["${ANDROID_NDK}" == ""]; then
  echo "Please set ANDROID_NDK enviornment variable."
  echo "For example it can be /Users/guest/Desktop/android-ndk-r26."
  echo "You can use setup_android_ndk function in scripts/android_example.sh"
  echo "to set up; or you can download from Android NDK website"
  exit 1
else
  echo "ANDROID_NDK set to ${ANDROID_NDK}"
fi

export CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
export ANDROID_ABI=arm64-v8a
export ANDROID_PLATFORM=android-23
export ET_BUILD_DIR="et-build-android"
export CMAKE_OUT_DIR="cmake-out-android"
export EXECUTORCH_BUILD_CUSTOM_OPS_AOT="OFF"
export EXECUTORCH_BUILD_CUSTOM="ON"
export CMAKE_OUT_DIR="cmake-out-android"
# export DCMAKE_INSTALL_PREFIX=cmake-out-android
#

build_runner_et() {
  rm -rf cmake-out-android
  echo "ET BUILD DIR IS ${ET_BUILD_DIR}"
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -S . -B cmake-out-android -G Ninja
  cmake --build cmake-out-android/ -j16 --config Release --target et_run
}

find_cmake_prefix_path
clone_executorch
install_executorch
build_runner_et

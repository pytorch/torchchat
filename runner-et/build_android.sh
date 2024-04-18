#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

if ["${ANDROID_NDK}" == ""]; then
  echo "Please set ANDROID_NDK enviornment variable."
  echo "For example it can be /Users/guest/Desktop/android-ndk-r26."
  echo "You can download NDK from android website"
  exit 1
else
  echo "ANDROID_NDK set to ${ANDROID_NDK}"
fi

export CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
export DANDROID_ABI=arm64-v8a
export DANDROID_PLATFORM=android-23 
export ET_BUILD_DIR="et-build-android"
export CMAKE_OUT_DIR="cmake-out-android"
# export DCMAKE_INSTALL_PREFIX=cmake-out-android
#

install_executorch() {
  echo "Cloning executorch to ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src"
  ET_BUILD_DIR="${TORCHCHAT_ROOT}/${ET_BUILD_DIR}"
  rm -rf ${ET_BUILD_DIR}
  mkdir -p ${ET_BUILD_DIR}/src
  pushd ${ET_BUILD_DIR}/src
  git clone https://github.com/pytorch/executorch.git
  cd executorch
  git checkout viable/strict
  echo "Install executorch: submodule update"
  git submodule sync
  git submodule update --init

  echo "Applying fixes"
  cp ${TORCHCHAT_ROOT}/scripts/fixes_et/module.cpp ${ET_BUILD_DIR}/src/executorch/extension/module/module.cpp # ET uses non-standard C++ that does not compile in GCC
  cp ${TORCHCHAT_ROOT}/scripts/fixes_et/managed_tensor.h ${ET_BUILD_DIR}/src/executorch/extension/runner_util/managed_tensor.h # ET is missing headers for vector/memory.  This causes downstream issues when building runner-et.

  CMAKE_OUT_DIR="cmake-out-android"
  echo "Building and installing C++ libraries"
  echo "Inside: ${PWD}"
  mkdir ${CMAKE_OUT_DIR}
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_INSTALL_PREFIX=cmake-out-android -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=Info -DEXECUTORCH_BUILD_OPTIMIZED=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=ON -S . -B ${CMAKE_OUT_DIR} -G Ninja
  cmake --build ${CMAKE_OUT_DIR}
  cmake --install ${CMAKE_OUT_DIR} --prefix ${ET_BUILD_DIR}/install
  popd
}

build_runner_et() {
  rm -rf build/cmake-out-android
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -S ./runner-et -B build/cmake-out-android -G Ninja
  cmake --build build/cmake-out-android/ -j16 --config Release
}

# install_executorch
build_runner_et

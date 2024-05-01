#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex pipefail

install_pip_dependencies() {
  echo "Intalling common pip packages"
  pip3 install wheel "cmake>=3.19" ninja zstd
  pushd ${TORCHCHAT_ROOT}
  pip3 install -r ./requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121
  popd
}

function find_cmake_prefix_path() {
  path=`python -c "from distutils.sysconfig import get_python_lib;print(get_python_lib())"`
  MY_CMAKE_PREFIX_PATH=$path
}

clone_executorch() {
  echo "Cloning executorch to ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src"
  rm -rf ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}
  mkdir -p ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src
  pushd ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src
  git clone https://github.com/pytorch/executorch.git
  cd executorch
  git checkout f0f4db877dd649c4e8ce951a4ccc3827841e9ad3
  echo "Install executorch: submodule update"
  git submodule sync
  git submodule update --init

  popd
}

install_executorch_python_libs() {
  if [ ! -d "${TORCHCHAT_ROOT}/${ET_BUILD_DIR}" ]; then
    echo "Directory ${TORCHCHAT_ROOT}/${ET_BUILD_DIR} does not exist."
    echo "Make sur eyou run clone_executorch"
    exit 1
  fi
  pushd ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src
  cd executorch

  echo "Building and installing python libraries"
  if [ "${ENABLE_ET_PYBIND}" = false ]; then
      echo "Not installing pybind"
      bash ./install_requirements.sh
  else
      echo "Installing pybind"
      bash ./install_requirements.sh --pybind xnnpack
  fi
  pip3 list
  popd
}

COMMON_CMAKE_ARGS="\
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info \
    -DEXECUTORCH_BUILD_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_QUANTIZED=ON"

install_executorch() {
  # AOT lib has to be build for model export
  # So by default it is built, and you can explicitly opt-out
  EXECUTORCH_BUILD_CUSTOM_OPS_AOT_VAR=OFF
  if [ "${EXECUTORCH_BUILD_CUSTOM_OPS_AOT}" == "" ]; then
    EXECUTORCH_BUILD_CUSTOM_OPS_AOT_VAR=ON
  fi

  # but for runner not
  EXECUTORCH_BUILD_CUSTOM_VAR=OFF
  if [ ! ["${EXECUTORCH_BUILD_CUSTOM}" == ""] ]; then
    EXECUTORCH_BUILD_CUSTOM_VAR=ON
  fi
  echo "${EXECUTORCH_BUILD_CUSTOM_OPS_AOT_VAR}"
  echo "${EXECUTORCH_BUILD_CUSTOM_VAR}"
  if [ ! -d "${TORCHCHAT_ROOT}/${ET_BUILD_DIR}" ]; then
    echo "Directory ${TORCHCHAT_ROOT}/${ET_BUILD_DIR} does not exist."
    echo "Make sure you run clone_executorch"
    exit 1
  fi
  pushd ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src
  cd executorch

  if [ "${CMAKE_OUT_DIR}" == "" ]; then
    CMAKE_OUT_DIR="cmake-out"
  fi

  CROSS_COMPILE_ARGS=""
  if [ "${CMAKE_OUT_DIR}" == "cmake-out-android" ]; then
    CROSS_COMPILE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DANDROID_ABI=${ANDROID_ABI} -DANDROID_PLATFORM=${ANDROID_PLATFORM}"
  fi

  echo "Building and installing C++ libraries"
  echo "Inside: ${PWD}"
  rm -rf ${CMAKE_OUT_DIR}
  mkdir ${CMAKE_OUT_DIR}
  cmake ${COMMON_CMAKE_ARGS} \
        -DCMAKE_PREFIX_PATH=${MY_CMAKE_PREFIX_PATH} \
        -DEXECUTORCH_BUILD_CUSTOM_OPS_AOT=${EXECUTORCH_BUILD_CUSTOM_OPS_AOT_VAR} \
        -DEXECUTORCH_BUILD_CUSTOM=${EXECUTORCH_BUILD_CUSTOM_VAR} \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        ${CROSS_COMPILE_ARGS} \
        -S . -B ${CMAKE_OUT_DIR} -G Ninja
  cmake --build ${CMAKE_OUT_DIR}
  cmake --install ${CMAKE_OUT_DIR} --prefix ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install
  popd
}

install_executorch_libs() {
  # Install executorch python and C++ libs
  export CMAKE_ARGS="\
    ${COMMON_CMAKE_ARGS} \
    -DCMAKE_PREFIX_PATH=${MY_CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install"
  export CMAKE_BUILD_ARGS="--target install"

  install_executorch_python_libs $1
}

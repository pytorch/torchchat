#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex pipefail

if [ -z "$TORCHCHAT_ROOT" ]; then
  # Get the absolute path of the current script
  SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
  TORCHCHAT_ROOT="$SCRIPT_PATH/../../.."
  echo "Defaulting TORCHCHAT_ROOT to $TORCHCHAT_ROOT since it is unset."
fi

install_pip_dependencies() {
  echo "Intalling common pip packages"
  pip3 install wheel "cmake>=3.19" ninja zstd
  pushd ${TORCHCHAT_ROOT}
  pip3 install -r install/requirements.txt
  popd
}

function find_cmake_prefix_path() {
  path=`python3 -c "from distutils.sysconfig import get_python_lib;print(get_python_lib())"`
  MY_CMAKE_PREFIX_PATH=$path
}

clone_executorch_internal() {
  rm -rf ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src

  mkdir -p ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src
  pushd ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src
  git clone https://github.com/pytorch/executorch.git
  cd executorch
  git checkout $(cat ${TORCHCHAT_ROOT}/install/.pins/et-pin.txt)
  echo "Install ExecuTorch: submodule update"
  git submodule sync
  git submodule update --init

  popd
}

clone_executorch() {
  echo "Cloning ExecuTorch to ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src"

  # Check if executorch is already cloned and has the correct version
  if [ -d "${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src/executorch" ]; then
    pushd ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src/executorch

    # Check if the repo is clean
    git_status=$(git status --porcelain)
    if [ -n "$git_status" ]; then
      echo "ExecuTorch repo is not clean. Removing and recloning."
      popd
      clone_executorch_internal
      return
    fi

    # Check if the version is the same
    current_version=$(git rev-parse HEAD)
    desired_version=$(cat ${TORCHCHAT_ROOT}/install/.pins/et-pin.txt)

    if [ "$current_version" == "$desired_version" ]; then
      echo "ExecuTorch is already cloned with the correct version. Skipping clone."
      popd
      return
    fi

    echo "ExecuTorch is already cloned but has the wrong version. Removing and recloning."
    popd
  fi

  clone_executorch_internal
}


install_executorch_python_libs() {
  if [ ! -d "${TORCHCHAT_ROOT}/${ET_BUILD_DIR}" ]; then
    echo "Directory ${TORCHCHAT_ROOT}/${ET_BUILD_DIR} does not exist."
    echo "Make sure you run clone_executorch"
    exit 1
  fi
  pushd ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src
  cd executorch

  echo "Building and installing python libraries"
  if [ "${ENABLE_ET_PYBIND}" = false ]; then
      echo "Not installing pybind"
      bash ./install_requirements.sh --pybind off
  else
      echo "Installing pybind"
      bash ./install_requirements.sh --pybind xnnpack
  fi

  # TODO: figure out the root cause of 'AttributeError: module 'evaluate'
  # has no attribute 'utils'' error from evaluate CI jobs and remove
  # `import lm_eval` from torchchat.py since it requires a specific version
  # of numpy.
  pip install numpy=='1.26.4'

  pip3 list
  popd
}

COMMON_CMAKE_ARGS="\
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON"

install_executorch_cpp_libs() {
  # AOT lib has to be build for model export
  # So by default it is built, and you can explicitly opt-out
  EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT_VAR=OFF
  if [ "${EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT}" == "" ]; then
    EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT_VAR=ON
  fi

  # but for runner not
  EXECUTORCH_BUILD_KERNELS_CUSTOM_VAR=OFF
  if [ ! ["${EXECUTORCH_BUILD_KERNELS_CUSTOM}" == ""] ]; then
    EXECUTORCH_BUILD_KERNELS_CUSTOM_VAR=ON
  fi
  echo "${EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT_VAR}"
  echo "${EXECUTORCH_BUILD_KERNELS_CUSTOM_VAR}"
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
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM_AOT=${EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT_VAR} \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=${EXECUTORCH_BUILD_KERNELS_CUSTOM_VAR} \
        ${CROSS_COMPILE_ARGS} \
        -S . -B ${CMAKE_OUT_DIR} -G Ninja
  cmake --build ${CMAKE_OUT_DIR} -j16
  cmake --install ${CMAKE_OUT_DIR} --prefix ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install
  popd
}

install_executorch_libs() {
  EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT_VAR=OFF
  EXECUTORCH_BUILD_KERNELS_CUSTOM_VAR=OFF
  install_executorch_cpp_libs
  install_executorch_python_libs $1
}

clone_torchao() {
  echo "Cloning torchao to ${TORCHCHAT_ROOT}/torchao-build/src"
  rm -rf ${TORCHCHAT_ROOT}/torchao-build
  mkdir -p ${TORCHCHAT_ROOT}/torchao-build/src
  pushd ${TORCHCHAT_ROOT}/torchao-build/src
  echo $pwd

  git clone https://github.com/pytorch/ao.git
  cd ao
  git checkout $(cat ${TORCHCHAT_ROOT}/install/.pins/torchao-pin.txt)

  popd
}

install_torchao_aten_ops() {
  local device=${1:-cpu}

  if [[ "$device" == "cpu" ]]; then
    echo "Building torchao custom ops for ATen"
    pushd ${TORCHCHAT_ROOT}/torchao-build/src/ao/torchao/experimental
  elif [[ "$device" == "mps" ]]; then
    echo "Building torchao mps custom ops for ATen"
    pushd ${TORCHCHAT_ROOT}/torchao-build/src/ao/torchao/experimental/ops/mps
  else
    echo "Invalid argument: $device. Valid values are 'cpu' or 'mps'." >&2
    return 1
  fi

  CMAKE_OUT_DIR=${TORCHCHAT_ROOT}/torchao-build/cmake-out
  cmake -DCMAKE_PREFIX_PATH=${MY_CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT_DIR} \
    -DCMAKE_BUILD_TYPE="Release" \
    -S . \
    -B ${CMAKE_OUT_DIR} -G Ninja
  cmake --build  ${CMAKE_OUT_DIR} --target install --config Release

  popd
}

install_torchao_executorch_ops() {
  echo "Building torchao custom ops for ExecuTorch"
  pushd ${TORCHCHAT_ROOT}/torchao-build/src/ao/torchao/experimental

  CMAKE_OUT_DIR="${TORCHCHAT_ROOT}/torchao-build/cmake-out"
  cmake -DCMAKE_PREFIX_PATH=${MY_CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT_DIR} \
    -DCMAKE_BUILD_TYPE="Release" \
    -DTORCHAO_BUILD_EXECUTORCH_OPS=ON \
    -DEXECUTORCH_INCLUDE_DIRS="${EXECUTORCH_INCLUDE_DIRS}" \
    -DEXECUTORCH_LIBRARIES="${EXECUTORCH_LIBRARIES}" \
    -S . \
    -B ${CMAKE_OUT_DIR} -G Ninja
  cmake --build  ${CMAKE_OUT_DIR} --target install --config Release

  popd
}

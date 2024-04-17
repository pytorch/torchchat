#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

install_pip_dependencies() {
  echo "Intalling common pip packages"

  pip install wheel
  pip install cmake
  pip install ninja
  pip install zstd
  pushd ${TORCHCHAT_ROOT}
  pip install -r ./requirements.txt
  popd
}

install_executorch() {
<<<<<<< HEAD
  echo "Cloning executorch to ${TORCHCHAT_ROOT}/build/src"
  rm -rf ${TORCHCHAT_ROOT}/build
  mkdir -p ${TORCHCHAT_ROOT}/build/src
  pushd ${TORCHCHAT_ROOT}/build/src
=======
  echo "Cloning executorch to ${TORCHCHAT_ROOT}/et-build/src"
  rm -rf ${TORCHCHAT_ROOT}/et-build
  mkdir -p ${TORCHCHAT_ROOT}/et-build/src
  pushd ${TORCHCHAT_ROOT}/et-build/src
>>>>>>> main
  git clone https://github.com/pytorch/executorch.git
  cd executorch
  echo "Install executorch: submodule update"
  git submodule sync
  git submodule update --init

  echo "Applying fixes"
<<<<<<< HEAD
  cp ${TORCHCHAT_ROOT}/scripts/fixes_et/module.cpp ${TORCHCHAT_ROOT}/build/src/executorch/extension/module/module.cpp # ET uses non-standard C++ that does not compile in GCC
  cp ${TORCHCHAT_ROOT}/scripts/fixes_et/managed_tensor.h ${TORCHCHAT_ROOT}/build/src/executorch/extension/runner_util/managed_tensor.h # ET is missing headers for vector/memory.  This causes downstream issues when building runner-et.
=======
  cp ${TORCHCHAT_ROOT}/scripts/fixes_et/module.cpp ${TORCHCHAT_ROOT}/et-build/src/executorch/extension/module/module.cpp # ET uses non-standard C++ that does not compile in GCC
  cp ${TORCHCHAT_ROOT}/scripts/fixes_et/managed_tensor.h ${TORCHCHAT_ROOT}/et-build/src/executorch/extension/runner_util/managed_tensor.h # ET is missing headers for vector/memory.  This causes downstream issues when building runner-et.
>>>>>>> main

  echo "Building and installing python libraries"
  echo "Building and installing python libraries"
  if [ "${ENABLE_ET_PYBIND}" = false ]; then
      echo "Not installing pybind"
      bash ./install_requirements.sh
  else
      echo "Installing pybind"
      bash ./install_requirements.sh --pybind xnnpack
  fi
  pip list

  echo "Building and installing C++ libraries"
  echo "Inside: ${PWD}"
  mkdir cmake-out
  cmake -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_OPTIMIZED=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=ON -S . -B cmake-out -G Ninja
  cmake --build cmake-out
<<<<<<< HEAD
  cmake --install cmake-out --prefix ${TORCHCHAT_ROOT}/build/install
=======
  cmake --install cmake-out --prefix ${TORCHCHAT_ROOT}/et-build/install
>>>>>>> main
  popd
}


ENABLE_ET_PYBIND="${1:-true}"

pushd ${TORCHCHAT_ROOT}
install_pip_dependencies
install_executorch $ENABLE_ET_PYBIND
popd

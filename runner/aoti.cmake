#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
IF(DEFINED ENV{TORCHCHAT_ROOT})
    set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
ELSE()
    set(TORCHCHAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
ENDIF()

find_package(CUDA)

find_package(Torch 2.4.0)
if(Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${TORCH_CXX_FLAGS} -fpermissive")

    add_executable(aoti_run runner/run.cpp)
    target_compile_options(aoti_run PUBLIC -D__AOTI_MODEL__)
    if(DEFINED TORCH_CUDA_LIBRARIES)
        target_compile_options(aoti_run PUBLIC -DUSE_CUDA)
    endif()
    target_include_directories(aoti_run PRIVATE ${TORCHCHAT_ROOT}/runner)
    target_link_libraries(aoti_run "${TORCH_LIBRARIES}" m)
    set_property(TARGET aoti_run PROPERTY CXX_STANDARD 17)
endif()

if (LINK_TORCHAO_OPS)
    target_link_libraries(aoti_run "${TORCHCHAT_ROOT}/torchao-build/cmake-out/lib/libtorchao_ops_aten${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

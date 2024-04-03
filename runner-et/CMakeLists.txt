cmake_minimum_required(VERSION 3.19)

project(llama_fast)

set(CMAKE_CXX_STANDARD 17)

set(ET_DIR "./executorch" CACHE STRING "Path to executorch.  Default value is ./executorch.")

# Before calling this, executorch must be built with cmake.  Calling the ./install_requirements.sh in
# the executorch repo is not enough.
# To build executorch with cmake, cd to $ET_DIR and do the following steps:
# 1. mkdir cmake-out
# 2. cmake -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_XNNPACK=ON -S . -B cmake-out; cmake --build cmake-out

include(CMakePrintHelpers)
cmake_print_variables(ET_DIR)

add_executable(runner_et
    run.cpp
    # For reasons, we cannot link ${EXECUTORCH_DIR}/cmake-out/kernels/portable/libportable_ops_lib.a as a library
    # without doing whole-archive linking.  Otherwise we get "Missing operator" errors.
    # But whole-archive linking seems to have different flags depending on platform, and I don't want to deal with it.
    # (It looks like OBJECT libraries are cmake's answer to this, but ET does not have these set up.)
    # But fortunately, including the following object file seems to make things work.
    ${ET_DIR}/cmake-out/kernels/portable/CMakeFiles/portable_ops_lib.dir/RegisterCodegenUnboxedKernelsEverything.cpp.o
)
target_include_directories(runner_et PRIVATE ${ET_DIR}/..)
target_link_libraries(
    runner_et PRIVATE
    ${ET_DIR}/cmake-out/extension/data_loader/libextension_data_loader.a
    ${ET_DIR}/cmake-out/kernels/portable/libportable_kernels.a
    ${ET_DIR}/cmake-out/libexecutorch.a
    ${ET_DIR}/cmake-out/extension/module/libextension_module.a
    ${ET_DIR}/cmake-out/backends/xnnpack/libxnnpack_backend.a
)

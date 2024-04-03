cmake_minimum_required(VERSION 3.24)

project(llama_fast)

set(CMAKE_CXX_STANDARD 17)

set(ET_DIR "./executorch" CACHE STRING "Path to executorch.  Default value is ./executorch.")

include(CMakePrintHelpers)
cmake_print_variables(ET_DIR)

add_executable(runner_et run.cpp)
target_include_directories(runner_et PRIVATE ${ET_DIR}/..)

# Link ET runtime + extensions
target_link_libraries(
    runner_et PRIVATE
    ${ET_DIR}/cmake-out/extension/data_loader/libextension_data_loader.a
    ${ET_DIR}/cmake-out/kernels/portable/libportable_kernels.a
    ${ET_DIR}/cmake-out/libexecutorch.a
    ${ET_DIR}/cmake-out/extension/module/libextension_module.a
)

# To register kernels + backends properly, we need to use whole-archive linking.  This requires cmake 3.24+.

# Link portable kernel library
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ET_DIR}/cmake-out/kernels/portable/libportable_ops_lib.a">)

# Link XNNPACK
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ET_DIR}/cmake-out/backends/xnnpack/libxnnpack_backend.a>")
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ET_DIR}/cmake-out/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a>")
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ET_DIR}/cmake-out/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a>")
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ET_DIR}/cmake-out/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a>")

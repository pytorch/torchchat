cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)

project(llama-fast)

include(CMakePrintHelpers)
set(LLAMA_FAST_ROOT $ENV{LLAMA_FAST_ROOT})
cmake_print_variables(LLAMA_FAST_ROOT)

find_package(executorch CONFIG REQUIRED PATHS ${LLAMA_FAST_ROOT}/build/install/lib/cmake/ExecuTorch)
set(_common_include_directories ${LLAMA_FAST_ROOT}/build/src)
cmake_print_variables(_common_include_directories)

target_include_directories(executorch INTERFACE ${_common_include_directories}) # Ideally ET installation process would do this
add_executable(runner_et run.cpp)

# Link ET runtime + extensions
target_link_libraries(
    runner_et PRIVATE
        executorch
        extension_module
        ${LLAMA_FAST_ROOT}/build/src/executorch/cmake-out/extension/data_loader/libextension_data_loader.a # This one does not get installed by ET
        optimized_kernels
        portable_kernels
        cpublas
        eigen_blas
)
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,optimized_native_cpu_ops_lib">)
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend">)
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,XNNPACK">)
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,pthreadpool">)
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,cpuinfo">)

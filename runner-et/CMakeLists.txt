cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)

project(Torchchat)

include(CMakePrintHelpers)
include(Utils.cmake)
set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
cmake_print_variables(TORCHCHAT_ROOT)

find_package(executorch CONFIG REQUIRED PATHS ${TORCHCHAT_ROOT}/et-build/install/lib/cmake/ExecuTorch)
set(_common_include_directories ${TORCHCHAT_ROOT}/et-build/src)
cmake_print_variables(_common_include_directories)

target_include_directories(executorch INTERFACE ${_common_include_directories}) # Ideally ExecuTorch installation process would do this
add_executable(runner_et run.cpp)

# Link ET runtime + extensions
target_link_libraries(
    runner_et PRIVATE
        executorch
        extension_module
        ${TORCHCHAT_ROOT}/et-build/src/executorch/cmake-out/extension/data_loader/libextension_data_loader.a # This one does not get installed by ExecuTorch
        optimized_kernels
        portable_kernels
        cpublas
        eigen_blas
        # The libraries below need to be whole-archived linked
        optimized_native_cpu_ops_lib
        xnnpack_backend
        XNNPACK
        pthreadpool
        cpuinfo
)
target_link_options_shared_lib(optimized_native_cpu_ops_lib)
target_link_options_shared_lib(xnnpack_backend)
target_link_options_shared_lib(XNNPACK)
target_link_options_shared_lib(pthreadpool)
target_link_options_shared_lib(cpuinfo)
target_link_options_shared_lib(executorch)

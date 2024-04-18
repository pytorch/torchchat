cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)

IF(DEFINED ENV{ET_BUILD_DIR})
  set(ET_BUILD_DIR $ENV{ET_BUILD_DIR})
ELSE()
  set(ET_BUILD_DIR "et-build")
ENDIF()

MESSAGE(STATUS "Using ET BUILD DIR: --[${ET_BUILD_DIR}]--")

IF(DEFINED ENV{CMAKE_OUT_DIR})
  set(CMAKE_OUT_DIR $ENV{CMAKE_OUT_DIR})
ELSE()
  set(CMAKE_OUT_DIR "cmake-out")
ENDIF()

MESSAGE(STATUS "Using ET BUILD DIR: --[${ET_BUILD_DIR}]--")

project(Torchchat)

include(CMakePrintHelpers)
include(Utils.cmake)
set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
cmake_print_variables(TORCHCHAT_ROOT)

MESSAGE(STATUS "Looking for excutorch in ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install/lib/cmake/ExecuTorch")
set(executorch_DIR ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install/lib/cmake/ExecuTorch)
find_package(executorch CONFIG REQUIRED PATHS ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/install/lib/cmake/ExecuTorch)
MESSAGE(STATUS "Found executorch cmake config.")

set(_common_include_directories ${TORCHCHAT_ROOT}/${ET_BUILD_DIR}/src)
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
        ${TORCHCHAT_ROOT}/et-build/src/executorch/cmake-out/examples/models/llama2/custom_ops/libcustom_ops.a # This one does not get installed by ExecuTorch
        # The libraries below need to be whole-archived linked
        optimized_native_cpu_ops_lib
        xnnpack_backend
        XNNPACK
        pthreadpool
        cpuinfo
        # ${TORCHCHAT_ROOT}/et-build/src/executorch/cmake-out/examples/models/llama2/custom_ops/libcustom_ops_lib.a # This one does not get installed by ExecuTorch
)
target_link_options_shared_lib(executorch)
target_link_options_shared_lib(optimized_native_cpu_ops_lib)
target_link_options_shared_lib(xnnpack_backend)
target_link_options_shared_lib(XNNPACK)
target_link_options_shared_lib(pthreadpool)
target_link_options_shared_lib(cpuinfo)
target_link_libraries(runner_et PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${TORCHCHAT_ROOT}/et-build/src/executorch/cmake-out/examples/models/llama2/custom_ops/libcustom_ops_lib.a>")

cmake_minimum_required(VERSION 3.17)
project(custom_linear_8da4w)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_library(custom_linear_8da4w SHARED custom_linear_8da4w.cpp)
target_compile_definitions(custom_linear_8da4w PRIVATE USE_XNNPACK=1 USE_PTHREADPOOL=1) # needed for torch include headers
target_include_directories(custom_linear_8da4w PRIVATE "${TORCHCHAT_ROOT}/..")
target_link_libraries(custom_linear_8da4w PRIVATE "${TORCH_LIBRARIES}")

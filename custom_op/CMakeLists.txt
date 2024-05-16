cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_linear_op)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
message("${TORCH_LIBRARIES}")

add_library(custom_linear_op SHARED cache.cpp op.cpp)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
target_include_directories(custom_linear_op PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(custom_linear_op PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(custom_linear_op PRIVATE cxx_range_for)

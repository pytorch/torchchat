cmake_minimum_required(VERSION 3.17)
project(custom_linear)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_linear SHARED custom_linear.cpp)
target_link_libraries(custom_linear "${TORCH_LIBRARIES}")

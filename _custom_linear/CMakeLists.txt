cmake_minimum_required(VERSION 3.17)
project(custom_linear)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(Torch REQUIRED)

add_library(custom_linear SHARED custom_linear.cpp)
target_include_directories(custom_linear PRIVATE "${TORCHCHAT_ROOT}/..")
target_link_libraries(custom_linear PRIVATE "${TORCH_LIBRARIES}")


add_executable(main main.cpp)
target_include_directories(main PRIVATE "${TORCHCHAT_ROOT}/..")
target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}")

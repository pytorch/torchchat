cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(llama2so LANGUAGES CXX)

find_package(CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${TORCH_CXX_FLAGS} -fpermissive")

add_executable(run run.cpp)
target_link_libraries(run "${TORCH_LIBRARIES}" m)
set_property(TARGET run PROPERTY CXX_STANDARD 17)

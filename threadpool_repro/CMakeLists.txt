cmake_minimum_required(VERSION 3.17)
project(repro)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_compile_definitions(main PRIVATE USE_PTHREADPOOL=1) # needed for torch include headers
target_include_directories(main PRIVATE "${TORCHCHAT_ROOT}/..")
target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}")

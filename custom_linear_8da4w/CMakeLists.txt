cmake_minimum_required(VERSION 3.17)
project(custom_linear_8da4w)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

# add_library(threadpool STATIC threadpool/threadpool.cpp threadpool/threadpool_guard.cpp)
# target_link_libraries(threadpool PRIVATE "${TORCHCHAT_ROOT}/custom_linear_8da4w/pthreadpool/build/install/lib/libpthreadpool.a")
# target_include_directories(threadpool PRIVATE "${TORCHCHAT_ROOT}/custom_linear_8da4w/pthreadpool/build/install/include")
# target_include_directories(threadpool PRIVATE "${TORCHCHAT_ROOT}/..")

add_library(custom_linear_8da4w SHARED custom_linear_8da4w.cpp)
target_include_directories(custom_linear_8da4w PRIVATE "${TORCHCHAT_ROOT}/..")
target_link_libraries(custom_linear_8da4w PRIVATE "${TORCH_LIBRARIES}")
# target_link_libraries(custom_linear_8da4w PRIVATE threadpool)

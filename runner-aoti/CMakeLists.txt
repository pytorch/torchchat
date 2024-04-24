cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(llama2so LANGUAGES CXX)
IF(DEFINED ENV{TORCHCHAT_ROOT})
    set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
ELSE()
    set(TORCHCHAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
ENDIF()

find_package(CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${TORCH_CXX_FLAGS} -fpermissive")

add_executable(run run.cpp ${TORCHCHAT_ROOT}/runner/bpe_tokenizer.cpp)
target_include_directories(run PRIVATE ${TORCHCHAT_ROOT}/runner)
target_link_libraries(run "${TORCH_LIBRARIES}" m)
set_property(TARGET run PROPERTY CXX_STANDARD 17)

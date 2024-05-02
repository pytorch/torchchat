cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
IF(DEFINED ENV{TORCHCHAT_ROOT})
    set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
ELSE()
    set(TORCHCHAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
ENDIF()

# build tokenizer library
add_library(
    tokenizer
    tokenizer.h
    sentencepiece.cpp
    tiktoken.cpp)

target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} third-party/sentencepiece/src)

# add RE2 as subdirectory
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag
${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(third-party/abseil-cpp)
add_subdirectory(third-party/re2)
add_subdirectory(third-party/sentencepiece)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

target_link_libraries(tokenizer PUBLIC re2::re2 sentencepiece-static)

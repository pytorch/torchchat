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
    tiktoken.cpp
    tokenizers.cpp)

target_include_directories(
    tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    third-party/sentencepiece/src
    third-party/json/single_include)

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

# Add tokenizer_tool binary for testing
OPTION(BUILD_TOKENIZE_TOOL "Build the tokenize_tool binary" OFF)
if (BUILD_TOKENIZE_TOOL)
    add_executable(tokenize_tool tokenize_tool.cpp)
    target_include_directories(tokenize_tool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(tokenize_tool tokenizer)
endif()
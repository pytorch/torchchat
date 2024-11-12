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
    hf_tokenizers.cpp
    pre_tokenizer.cpp
    # llama.cpp unicode
    unicode-data.cpp
    unicode.cpp)

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
set(binary_dir "${CMAKE_BINARY_DIR}/bin")
OPTION(BUILD_TOKENIZE_TOOL "Build the tokenize_tool binary" OFF)
if (BUILD_TOKENIZE_TOOL)
    add_executable(tokenize_tool tokenize_tool.cpp)
    set_target_properties(tokenize_tool PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${binary_dir})
    target_include_directories(tokenize_tool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(tokenize_tool tokenizer)
endif()

# Add unit tests
OPTION(BUILD_UNIT_TESTS "Enable unit testing" OFF)
if (BUILD_UNIT_TESTS)
    add_subdirectory(third-party/googletest)
    enable_testing()
    set(test_binary_dir "${binary_dir}/test")

    file(GLOB test_source_files tests/*_test.cpp)
    foreach(test_source_file ${test_source_files})
        get_filename_component(test_name ${test_source_file} NAME_WE)
        message(STATUS "Configuring unit test ${test_name}")
        add_executable(${test_name} tests/main.cpp ${test_source_file})
        set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${test_binary_dir})
        target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR})
        target_include_directories(${test_name} PRIVATE third-party/googletest/googletest/include)
        target_link_libraries(${test_name} gtest tokenizer)
        add_test(${test_name} "${test_binary_dir}/${test_name}")
    endforeach()
endif()
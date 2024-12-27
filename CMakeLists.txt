cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
IF(DEFINED ENV{TORCHCHAT_ROOT})
    set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
ELSE()
    set(TORCHCHAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
ENDIF()

project(Torchchat)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

# include tokenizer
add_subdirectory(runner/third-party/tokenizers)

# include et_run executable
include(runner/et.cmake)
if(TARGET et_run)
    target_link_libraries(et_run PUBLIC tokenizers microkernels-prod)
    target_include_directories(et_run PUBLIC runner/third-party/tokenizers/include)
endif()

# include aoti_run executable
include(runner/aoti.cmake)
if(TARGET aoti_run)
    target_link_libraries(aoti_run tokenizers)
    target_include_directories(aoti_run PUBLIC runner/third-party/tokenizers/include)
endif()

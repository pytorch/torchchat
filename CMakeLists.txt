cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
IF(DEFINED ENV{TORCHCHAT_ROOT})
    set(TORCHCHAT_ROOT $ENV{TORCHCHAT_ROOT})
ELSE()
    set(TORCHCHAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
ENDIF()

project(Torchchat)

# include tokenizer
add_subdirectory(tokenizer)

# include et_run executable
include(runner/et.cmake)
if(TARGET et_run)
    target_link_libraries(et_run PUBLIC tokenizer)
endif()

# include aoti_run executable
include(runner/aoti.cmake)
if(TARGET aoti_run)
    target_link_libraries(aoti_run tokenizer)
endif()

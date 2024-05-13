CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(googlebenchmark-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googlebenchmark
	URL https://github.com/google/benchmark/archive/v1.5.3.zip
	URL_HASH SHA256=bdefa4b03c32d1a27bd50e37ca466d8127c1688d834800c38f3c587a396188ee
	SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source"
	BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)

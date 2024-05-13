CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(googletest-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googletest
	URL https://github.com/google/googletest/archive/release-1.12.0.zip
	URL_HASH SHA256=ce7366fe57eb49928311189cb0e40e0a8bf3d3682fca89af30d884c25e983786
    SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source"
    BINARY_DIR "${CMAKE_BINARY_DIR}/googletest"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)

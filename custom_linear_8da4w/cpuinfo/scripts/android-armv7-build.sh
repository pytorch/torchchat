#!/usr/bin/env bash

set -e

if [ -z "$ANDROID_NDK" ]
then
  echo "ANDROID_NDK not set; please set it to the Android NDK directory"
  exit 1
fi

if [ ! -d "$ANDROID_NDK" ]
then
  echo "ANDROID_NDK not a directory; did you install it under ${ANDROID_NDK}?"
  exit 1
fi

mkdir -p build/android/armeabi-v7a

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]
then
  CMAKE_ARGS+=("-GNinja")
fi

CMAKE_ARGS+=("-DCPUINFO_LIBRARY_TYPE=static")
# CMakeLists for Google Benchmark is broken on Android
CMAKE_ARGS+=("-DCPUINFO_BUILD_BENCHMARKS=OFF")
CMAKE_ARGS+=("-DCPUINFO_BUILD_TOOLS=ON")
CMAKE_ARGS+=("-DCPUINFO_BUILD_UNIT_TESTS=ON")
CMAKE_ARGS+=("-DCPUINFO_BUILD_MOCK_TESTS=ON")

# Android-specific options
CMAKE_ARGS+=("-DANDROID_NDK=$ANDROID_NDK")
CMAKE_ARGS+=("-DANDROID_ABI=armeabi-v7a")
CMAKE_ARGS+=("-DANDROID_PLATFORM=android-14")
CMAKE_ARGS+=("-DANDROID_PIE=ON")
CMAKE_ARGS+=("-DANDROID_STL=c++_static")
CMAKE_ARGS+=("-DANDROID_CPP_FEATURES=exceptions")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/android/armeabi-v7a && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ "$(uname)" == "Darwin" ]
then
  cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
  cmake --build . -- "-j$(nproc)"
fi

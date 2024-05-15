#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

cd ${TORCHCHAT_ROOT}
echo "Inside: $TORCHCHAT_ROOT"

which curl

if [ "$(uname)" == "Darwin" -a "$(uname -m)" == "arm64" ]; then
  JAVA_URL="https://download.oracle.com/java/17/archive/jdk-17.0.10_macos-aarch64_bin.tar.gz"
  SDK_MANAGER_URL="https://dl.google.com/android/repository/commandlinetools-mac-11076708_latest.zip"
  ANDROID_ABI=arm64-v8a
elif [ "$(uname)" == "Linux" -a "$(uname -m)" == "x86_64" ]; then
  JAVA_URL="https://download.oracle.com/java/17/archive/jdk-17.0.10_linux-x64_bin.tar.gz"
  SDK_MANAGER_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
  ANDROID_ABI=x86_64
else
  echo "Unsupported platform $(uname) $(uname -m)"
  exit -1
fi

LLAMA_AAR_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.2/executorch-llama.aar"

LLAMA_AAR_SHASUM="09d17f7bc59589b581e45bb49511d19196d0297d"

mkdir -p ${TORCHCHAT_ROOT}/build/android

setup_java() {
  if [ -x "$(command -v javac )" ] && [[ $(javac -version | cut -d' ' -f2 | cut -d'.' -f1) -ge 17 ]]; then
    echo "Java 17 is set up"
    return
  fi
  pushd ${TORCHCHAT_ROOT}/build/android
  echo "Download Java 17"
  curl "${JAVA_URL}" -o jdk-17.0.10.tar.gz

  echo "Unzip Java 17"
  tar xf jdk-17.0.10.tar.gz

  if [ "$(uname)" == "Darwin" -a "$(uname -m)" == "arm64" ]; then
    export JAVA_HOME="$(pwd)"/jdk-17.0.10.jdk/Contents/Home
    export PATH="$JAVA_HOME/bin:$PATH"
  elif [ "$(uname)" == "Linux" -a "$(uname -m)" == "x86_64" ]; then
    export JAVA_HOME="$(pwd)"/jdk-17.0.10
    export PATH="$JAVA_HOME/bin:$PATH"
  fi
  popd
}

setup_android_sdk_manager() {
  if [ -x "$(command -v sdkmanager )" ]; then
    echo "Android sdkmanager is set up"
    return
  fi
  pushd ${TORCHCHAT_ROOT}/build/android
  mkdir -p sdk/cmdline-tools/latest

  echo "Download Android SDK Manager"
  curl "${SDK_MANAGER_URL}" -o commandlinetools.zip

  echo "Unzip Android SDK Manager"
  unzip commandlinetools.zip
  mv cmdline-tools/* sdk/cmdline-tools/latest
  export PATH="$(realpath sdk/cmdline-tools/latest/bin):$PATH"
  popd
}

setup_android_sdk() {
  sdkmanager "platforms;android-34"
  sdkmanager "platform-tools"
}

download_aar_library() {
  mkdir -p ${TORCHCHAT_ROOT}/build/android/libs
  curl "${LLAMA_AAR_URL}" -o ${TORCHCHAT_ROOT}/build/android/libs/executorch.aar
  echo "${LLAMA_AAR_SHASUM}  ${TORCHCHAT_ROOT}/build/android/libs/executorch.aar" | shasum --check --status
}

build_app() {
  pushd android/Torchchat
  mkdir -p app/libs
  cp ${TORCHCHAT_ROOT}/build/android/executorch.aar app/libs
  ./gradlew :app:build
  popd
}

setup_avd() {
  if adb devices | grep -q "device$"; then
    echo "adb device detected, skipping avd setup"
    return
  fi
  sdkmanager "emulator"
  sdkmanager "system-images;android-34;google_apis;${ANDROID_ABI}"
  if ! avdmanager list avd | grep -q "torchchat"; then
    avdmanager create avd --name "torchchat" --package "system-images;android-34;google_apis;${ANDROID_ABI}"
  fi
}

export_model() {
  python torchchat.py export stories15M --output-pte-path ./build/android/model.pte
  curl -fsSL https://github.com/karpathy/llama2.c/raw/master/tokenizer.model -o ./build/android/tokenizer.model
  python ./unsupported/llama2.c/runner-utils/tokenizer.py --tokenizer-model=./build/android/tokenizer.model
}

push_files_to_android() {
  echo "If you need to use emulator, please use a separate window and run"
  echo "sdk/emulator/emulator @torchchat > /dev/null 2>&1 &"
  adb wait-for-device
  adb shell mkdir -p /data/local/tmp/llama
  adb push build/android/model.pte /data/local/tmp/llama
  adb push build/android/tokenizer.bin /data/local/tmp/llama
  adb install -t android/Torchchat/app/build/outputs/apk/debug/app-debug.apk
}

run_android_instrumented_test() {
  pushd android/Torchchat
  ./gradlew connectedAndroidTest
  popd
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  setup_java
  setup_android_sdk_manager
  setup_android_sdk
  download_aar_library
  build_app
  setup_avd
  export_model
  push_files_to_android
  run_android_instrumented_test
fi

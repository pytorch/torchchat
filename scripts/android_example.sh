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

LLAMA_JNI_ARM64_URL="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/8812806770/artifact/arm64-v8a/libexecutorch_llama_jni.so"
LLAMA_JNI_X86_64_URL="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/8812806770/artifact/arm64-v8a/libexecutorch_llama_jni.so"
LLAMA_JAR_URL="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/8812806770/artifact/executorch.jar"

mkdir -p ${TORCHCHAT_ROOT}/build/android

setup_java() {
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
  pushd ${TORCHCHAT_ROOT}/build/android
  mkdir -p sdk/cmdline-tools/latest

  echo "Download Android SDK Manager"
  curl "${SDK_MANAGER_URL}" -o commandlinetools.zip

  echo "Unzip Android SDK Manager"
  unzip commandlinetools.zip
  mv cmdline-tools/* sdk/cmdline-tools/latest
  export PATH="$(realpath sdk/cmdline-tools/latest/bin):$PATH"
  export ANDROID_HOME="$(realpath ./sdk)"
  export ANDROID_SDK_ROOT="$ANDROID_HOME"
  popd
}

setup_android_sdk() {
  sdkmanager "platforms;android-34"
  sdkmanager "platform-tools"
}

setup_android_ndk() {
  sdkmanager "ndk;25.0.8775105"
  export ANDROID_NDK="$ANDROID_HOME/ndk/25.0.8775105"
}

download_jar_library() {
  mkdir -p ${TORCHCHAT_ROOT}/build/android
  curl "${LLAMA_JAR_URL}" -o ${TORCHCHAT_ROOT}/build/android/executorch.jar
}

download_jni_library() {
  mkdir -p ${TORCHCHAT_ROOT}/build/android/arm64-v8a
  mkdir -p ${TORCHCHAT_ROOT}/build/android/x86_64
  if [ ! -f ${TORCHCHAT_ROOT}/build/android/arm64-v8a/libexecutorch_llama_jni.so ]; then
    curl "${LLAMA_JNI_ARM64_URL}" -o ${TORCHCHAT_ROOT}/build/android/arm64-v8a/libexecutorch_llama_jni.so
  fi
  if [ ! -f ${TORCHCHAT_ROOT}/build/android/x86_64/libexecutorch_llama_jni.so ]; then
    curl "${LLAMA_JNI_X86_64_URL}" -o ${TORCHCHAT_ROOT}/build/android/x86_64/libexecutorch_llama_jni.so
  fi
}

build_app() {
  pushd build/src/executorch/examples/demo-apps/android/LlamaDemo
  mkdir -p app/src/main/jniLibs/arm64-v8a
  mkdir -p app/src/main/jniLibs/x86_64
  cp ${TORCHCHAT_ROOT}/build/android/arm64-v8a/libexecutorch_llama_jni.so app/src/main/jniLibs/arm64-v8a
  cp ${TORCHCHAT_ROOT}/build/android/x86_64/libexecutorch_llama_jni.so app/src/main/jniLibs/x86_64
  ./gradlew :app:build
  popd
}

setup_avd() {
  sdkmanager "emulator"
  sdkmanager "system-images;android-34;google_apis;${ANDROID_ABI}"

  avdmanager create avd --name "torchchat" --package "system-images;android-34;google_apis;${ANDROID_ABI}"
  sdk/emulator/emulator @torchchat &
}

push_files_to_android() {
  adb wait-for-device
  adb shell mkdir -p /data/local/tmp/llama
  adb push stories15M.pte /data/local/tmp/llama
  adb push checkpoints/stories15M/tokenizer.bin /data/local/tmp/llama
  adb install -t build/src/executorch/examples/demo-apps/android/LlamaDemo/app/build/outputs/apk/debug/app-debug.apk
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  setup_java
  setup_android_sdk_manager
  setup_android_sdk
  setup_android_ndk
  setup_avd
  download_jni_library
  build_app
  push_files_to_android
fi

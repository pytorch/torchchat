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

LLAMA_JNI_ARM64_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.2/arm64-v8a/libexecutorch_llama_jni.so"
LLAMA_JNI_X86_64_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.2/x86_64/libexecutorch_llama_jni.so"
LLAMA_JAR_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.2/executorch.jar"

LLAMA_JNI_ARM64_SHASUM="38788e2a3318075ccdd6e337e4c56f82bbbce06f"
LLAMA_JNI_X86_64_SHASUM="cdc98d468b8e48c8784408fcada8177e6bc5f981"
LLAMA_JAR_SHASUM="fb9ee00d028ef23a48cb8958638a5010ba849ccf"

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

download_jar_library() {
  mkdir -p ${TORCHCHAT_ROOT}/build/android/libs
  curl "${LLAMA_JAR_URL}" -o ${TORCHCHAT_ROOT}/build/android/libs/executorch.jar
  echo "${LLAMA_JAR_SHASUM}  ${TORCHCHAT_ROOT}/build/android/libs/executorch.jar" | shasum --check --status
}

download_jni_library() {
  mkdir -p ${TORCHCHAT_ROOT}/build/android/jni/arm64-v8a
  mkdir -p ${TORCHCHAT_ROOT}/build/android/jni/x86_64
  if [ ! -f ${TORCHCHAT_ROOT}/build/android/jni/arm64-v8a/libexecutorch_llama_jni.so ]; then
    curl "${LLAMA_JNI_ARM64_URL}" -o ${TORCHCHAT_ROOT}/build/android/jni/arm64-v8a/libexecutorch_llama_jni.so
    echo "${LLAMA_JNI_ARM64_SHASUM}  ${TORCHCHAT_ROOT}/build/android/jni/arm64-v8a/libexecutorch_llama_jni.so" | shasum --check --status
  fi
  if [ ! -f ${TORCHCHAT_ROOT}/build/android/jni/x86_64/libexecutorch_llama_jni.so ]; then
    curl "${LLAMA_JNI_X86_64_URL}" -o ${TORCHCHAT_ROOT}/build/android/jni/x86_64/libexecutorch_llama_jni.so
    echo "${LLAMA_JNI_X86_64_SHASUM}  ${TORCHCHAT_ROOT}/build/android/jni/x86_64/libexecutorch_llama_jni.so" | shasum --check --status
  fi
}

make_executorch_aar() {
  if [ -f ${TORCHCHAT_ROOT}/build/android/executorch.aar ]; then
    return
  fi
  pushd ${TORCHCHAT_ROOT}/build/android
  echo \<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" \
   package=\"org.pytorch.executorch\"\> \
   \<uses-sdk android:minSdkVersion=\"19\" /\> \
   \</manifest\> > AndroidManifest.xml
  zip -r executorch.aar libs jni AndroidManifest.xml
  popd
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
  sdk/emulator/emulator @torchchat > /dev/null 2>&1 &
}

push_files_to_android() {
  echo "If you need to use emulator, please use a separate window and run"
  echo "sdk/emulator/emulator @torchchat > /dev/null 2>&1 &"
  adb wait-for-device
  adb shell mkdir -p /data/local/tmp/llama
  adb push stories15M.pte /data/local/tmp/llama
  adb push checkpoints/stories15M/tokenizer.bin /data/local/tmp/llama
  adb install -t android/Torchchat/app/build/outputs/apk/debug/app-debug.apk
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  setup_java
  setup_android_sdk_manager
  setup_android_sdk
  download_jni_library
  download_jar_library
  make_executorch_aar
  build_app
  setup_avd
  push_files_to_android
fi

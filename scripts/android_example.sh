#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

cd ${LLAMA_FAST_ROOT}
echo "Inside: $LLAMA_FAST_ROOT"

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

mkdir -p ${LLAMA_FAST_ROOT}/build/android
pushd ${LLAMA_FAST_ROOT}/build/android

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

mkdir -p sdk/cmdline-tools/latest

echo "Download Android SDK Manager"
curl "${SDK_MANAGER_URL}" -o commandlinetools.zip

echo "Unzip Android SDK Manager"
unzip commandlinetools.zip
mv cmdline-tools/* sdk/cmdline-tools/latest
export PATH="$(realpath sdk/cmdline-tools/latest/bin):$PATH"


export ANDROID_HOME="$(realpath ./sdk)"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
sdkmanager "platforms;android-34"
sdkmanager "ndk;25.0.8775105"
sdkmanager "platform-tools"
export ANDROID_NDK="$ANDROID_HOME/ndk/25.0.8775105"
sdkmanager "emulator"
sdkmanager "system-images;android-34;google_apis;${ANDROID_ABI}"

popd

pushd build/src/executorch/examples/demo-apps/android/LlamaDemo
./gradlew :app:setup
./gradlew :app:build
popd

avdmanager create avd --name "llama-fast" --package "system-images;android-34;google_apis;${ANDROID_ABI}"
sdk/emulator/emulator @llama-fast &

adb wait-for-device
adb shell mkdir /data/local/tmp/llama
adb push stories15M.pte /data/local/tmp/llama
adb push checkpoints/stories15M/tokenizer.bin /data/local/tmp/llama
adb install -t build/src/executorch/examples/demo-apps/android/LlamaDemo/app/build/outputs/apk/debug/app-debug.apk

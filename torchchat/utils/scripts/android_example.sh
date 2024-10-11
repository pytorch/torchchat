#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

# In CI environment, we accept SDK license and return automatically
if [ "${1:-}" == "--ci" ]; then
  export CI_ENV=1
fi

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

LLAMA_AAR_URL="https://ossci-android.s3.amazonaws.com/executorch/release/executorch-241002/executorch.aar"
LLAMA_AAR_SHASUM_URL="https://ossci-android.s3.amazonaws.com/executorch/release/executorch-241002/executorch.aar.sha256sums"

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
  rm -r sdk/cmdline-tools/latest/* || true

  echo "Download Android SDK Manager"
  curl "${SDK_MANAGER_URL}" -o commandlinetools.zip

  echo "Unzip Android SDK Manager"
  unzip commandlinetools.zip
  mv cmdline-tools/* sdk/cmdline-tools/latest
  export PATH="$(realpath sdk/cmdline-tools/latest/bin):$PATH"
  export PATH="$(realpath sdk/platform-tools):$PATH"
  popd
}

setup_android_sdk() {
  if [ -z "${CI_ENV:-}" ]; then
    sdkmanager "platforms;android-34" "platform-tools"
  else
    yes | sdkmanager "platforms;android-34" "platform-tools"
  fi
  export ANDROID_HOME="$(realpath build/android/sdk)"
}

download_aar_library() {
  mkdir -p ${TORCHCHAT_ROOT}/android/torchchat/app/libs
  curl "${LLAMA_AAR_URL}" -O
  curl "${LLAMA_AAR_SHASUM_URL}" -O
  shasum --check --status executorch.aar.sha256sums
  mv executorch.aar ${TORCHCHAT_ROOT}/android/torchchat/app/libs/
}

build_app() {
  pushd torchchat/edge/android/torchchat
  ./gradlew :app:build
  popd
}

setup_avd() {
  if adb devices | grep -q "device$"; then
    echo "adb device detected, skipping avd setup"
    return
  fi
  if [ -z "${CI_ENV:-}" ]; then
    sdkmanager "emulator" \
        "system-images;android-34;google_apis;${ANDROID_ABI}"
  else
    yes | sdkmanager "emulator" \
        "system-images;android-34;google_apis;${ANDROID_ABI}"
  fi
  if ! avdmanager list avd | grep -q "torchchat"; then
    echo no | avdmanager create avd --name "torchchat" --package "system-images;android-34;google_apis;${ANDROID_ABI}"
  fi
  export ANDROID_SDK_ROOT=$(realpath ./build/android/)
  trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
  if [ -z "${CI_ENV:-}" ]; then
    ./build/android/sdk/emulator/emulator @torchchat &
  else
    ./build/android/sdk/emulator/emulator -no-audio -no-window -gpu swiftshader_indirect @torchchat &
  fi
}

export_model() {
  python torchchat.py export stories15M --output-pte-path ./build/android/model.pte
  curl -fsSL https://github.com/karpathy/llama2.c/raw/master/tokenizer.model -o ./build/android/tokenizer.model
  python ./et-build/src/executorch/examples/models/llama2/tokenizer/tokenizer.py -t ./build/android/tokenizer.model -o build/android/tokenizer.bin
}

push_files_to_android() {
  adb wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done; input keyevent 82'
  adb shell mkdir -p /data/local/tmp/llama
  adb push build/android/model.pte /data/local/tmp/llama
  adb push build/android/tokenizer.bin /data/local/tmp/llama
}

run_android_instrumented_test() {
  pushd torchchat/edge/android/torchchat
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

adb install -t torchchat/edge/android/torchchat/app/build/outputs/apk/debug/app-debug.apk

if [ -z "${CI_ENV:-}" ]; then
  read -p "Press enter to exit emulator and finish"
fi

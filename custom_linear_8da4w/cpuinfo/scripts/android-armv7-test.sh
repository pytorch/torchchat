#!/usr/bin/env bash

set -e

adb push build/android/armeabi-v7a/init-test /data/local/tmp/init-test
adb push build/android/armeabi-v7a/cache-test /data/local/tmp/cache-test
adb push build/android/armeabi-v7a/chipset-test /data/local/tmp/chipset-test
adb shell /data/local/tmp/init-test --gtest_color=yes
adb shell /data/local/tmp/cache-test --gtest_color=yes
adb shell /data/local/tmp/chipset-test --gtest_color=yes

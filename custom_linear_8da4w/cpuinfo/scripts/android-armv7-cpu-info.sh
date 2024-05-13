#!/usr/bin/env bash

set -e

adb push build/android/armeabi-v7a/cpu-info /data/local/tmp/cpu-info
adb shell /data/local/tmp/cpu-info

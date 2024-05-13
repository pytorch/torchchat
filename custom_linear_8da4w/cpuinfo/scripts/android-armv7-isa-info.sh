#!/usr/bin/env bash

set -e

adb push build/android/armeabi-v7a/isa-info /data/local/tmp/isa-info
adb shell /data/local/tmp/isa-info

#!/usr/bin/env bash

set -e

adb push build/android/x86/cpu-info /data/local/tmp/cpu-info
adb shell /data/local/tmp/cpu-info
